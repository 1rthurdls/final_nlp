"""
Preprocessing module for ABSA dataset.
Handles data loading, parsing, and preparation for model training.
"""

import ast
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import re


@dataclass
class AspectTerm:
    """Represents an aspect term with its properties."""
    term: str
    polarity: str
    start: int
    end: int


@dataclass
class AspectCategory:
    """Represents an aspect category with sentiment."""
    category: str
    polarity: str


@dataclass
class Review:
    """Represents a complete review with all annotations."""
    sentence_id: str
    text: str
    aspect_terms: List[AspectTerm]
    aspect_categories: List[AspectCategory]


class SemEvalDataLoader:
    """Load and parse SemEval 2014 dataset."""

    SENTIMENT_MAP = {
        'positive': 0,
        'negative': 1,
        'neutral': 2,
        'conflict': 2  # Treat conflict as neutral
    }

    CATEGORY_MAP = {
        'food': 0,
        'service': 1,
        'ambience': 2,
        'price': 3,
        'anecdotes/miscellaneous': 4
    }

    def __init__(self, file_path: str):
        """
        Initialize data loader.

        Args:
            file_path: Path to CSV file
        """
        self.file_path = file_path
        self.df = None
        self.reviews = []

    def load_data(self) -> pd.DataFrame:
        """Load CSV data."""
        self.df = pd.read_csv(self.file_path)
        return self.df

    def _parse_aspect_terms(self, aspect_terms_str: str) -> List[AspectTerm]:
        """Parse aspect terms from string representation."""
        if pd.isna(aspect_terms_str) or aspect_terms_str == '[]':
            return []

        try:
            terms_list = ast.literal_eval(aspect_terms_str)
            aspect_terms = []

            for term_dict in terms_list:
                aspect_term = AspectTerm(
                    term=term_dict['term'],
                    polarity=term_dict['polarity'],
                    start=int(term_dict['from']),
                    end=int(term_dict['to'])
                )
                aspect_terms.append(aspect_term)

            return aspect_terms
        except (ValueError, SyntaxError, KeyError) as e:
            print(f"Error parsing aspect terms: {aspect_terms_str}, {e}")
            return []

    def _parse_aspect_categories(self, aspect_categories_str: str) -> List[AspectCategory]:
        """Parse aspect categories from string representation."""
        if pd.isna(aspect_categories_str) or aspect_categories_str == '[]':
            return []

        try:
            categories_list = ast.literal_eval(aspect_categories_str)
            aspect_categories = []

            for cat_dict in categories_list:
                aspect_cat = AspectCategory(
                    category=cat_dict['category'],
                    polarity=cat_dict['polarity']
                )
                aspect_categories.append(aspect_cat)

            return aspect_categories
        except (ValueError, SyntaxError, KeyError) as e:
            print(f"Error parsing aspect categories: {aspect_categories_str}, {e}")
            return []

    def parse_reviews(self) -> List[Review]:
        """Parse all reviews from dataframe."""
        if self.df is None:
            self.load_data()

        self.reviews = []

        for idx, row in self.df.iterrows():
            aspect_terms = self._parse_aspect_terms(row['aspectTerms'])
            aspect_categories = self._parse_aspect_categories(row['aspectCategories'])

            review = Review(
                sentence_id=str(row['sentenceId']),
                text=row['text'],
                aspect_terms=aspect_terms,
                aspect_categories=aspect_categories
            )
            self.reviews.append(review)

        return self.reviews

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        if not self.reviews:
            self.parse_reviews()

        total_reviews = len(self.reviews)
        total_aspect_terms = sum(len(r.aspect_terms) for r in self.reviews)
        total_aspect_categories = sum(len(r.aspect_categories) for r in self.reviews)

        # Sentiment distribution
        sentiment_dist = {'positive': 0, 'negative': 0, 'neutral': 0}
        for review in self.reviews:
            for term in review.aspect_terms:
                if term.polarity in sentiment_dist:
                    sentiment_dist[term.polarity] += 1

        # Category distribution
        category_dist = {cat: 0 for cat in self.CATEGORY_MAP.keys()}
        for review in self.reviews:
            for cat in review.aspect_categories:
                if cat.category in category_dist:
                    category_dist[cat.category] += 1

        return {
            'total_reviews': total_reviews,
            'total_aspect_terms': total_aspect_terms,
            'total_aspect_categories': total_aspect_categories,
            'sentiment_distribution': sentiment_dist,
            'category_distribution': category_dist,
            'avg_aspects_per_review': total_aspect_terms / total_reviews if total_reviews > 0 else 0
        }


class ABSADataset(Dataset):
    """PyTorch Dataset for ABSA task."""

    def __init__(
        self,
        reviews: List[Review],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
        task: str = "joint"  # "joint", "aspect_extraction", "sentiment_classification"
    ):
        """
        Initialize ABSA dataset.

        Args:
            reviews: List of Review objects
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            task: Task type
        """
        self.reviews = reviews
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task

    def __len__(self) -> int:
        return len(self.reviews)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from dataset."""
        review = self.reviews[idx]

        # Tokenize text
        encoding = self.tokenizer(
            review.text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True
        )

        # Remove batch dimension
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        offset_mapping = encoding['offset_mapping'].squeeze(0)

        # Create aspect labels (BIO tagging)
        aspect_labels = self._create_aspect_labels(
            review.text,
            review.aspect_terms,
            offset_mapping
        )

        # Create sentiment labels for each aspect
        sentiment_labels = self._create_sentiment_labels(
            review.text,
            review.aspect_terms,
            offset_mapping
        )

        # Create category labels (multi-label classification)
        category_labels = self._create_category_labels(review.aspect_categories)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'aspect_labels': torch.tensor(aspect_labels, dtype=torch.long),
            'sentiment_labels': torch.tensor(sentiment_labels, dtype=torch.long),
            'category_labels': torch.tensor(category_labels, dtype=torch.float),
            'text': review.text,
            'sentence_id': review.sentence_id
        }

    def _create_aspect_labels(
        self,
        text: str,
        aspect_terms: List[AspectTerm],
        offset_mapping: torch.Tensor
    ) -> List[int]:
        """
        Create BIO labels for aspect term extraction.

        Labels: 0=O, 1=B-ASPECT, 2=I-ASPECT
        """
        labels = [0] * self.max_length  # O (Outside)

        for aspect in aspect_terms:
            for token_idx, (start, end) in enumerate(offset_mapping):
                if start == 0 and end == 0:  # Special tokens
                    continue

                # Check if token overlaps with aspect span
                if start >= aspect.start and end <= aspect.end:
                    if start == aspect.start or labels[token_idx] == 0:
                        labels[token_idx] = 1  # B-ASPECT
                    else:
                        labels[token_idx] = 2  # I-ASPECT

        return labels

    def _create_sentiment_labels(
        self,
        text: str,
        aspect_terms: List[AspectTerm],
        offset_mapping: torch.Tensor
    ) -> List[int]:
        """
        Create sentiment labels for each token.

        Labels: 0=positive, 1=negative, 2=neutral, -100=ignore
        """
        labels = [-100] * self.max_length  # Ignore by default

        for aspect in aspect_terms:
            sentiment_label = SemEvalDataLoader.SENTIMENT_MAP.get(aspect.polarity, 2)

            for token_idx, (start, end) in enumerate(offset_mapping):
                if start == 0 and end == 0:  # Special tokens
                    continue

                # Assign sentiment to aspect tokens
                if start >= aspect.start and end <= aspect.end:
                    labels[token_idx] = sentiment_label

        return labels

    def _create_category_labels(
        self,
        aspect_categories: List[AspectCategory]
    ) -> List[float]:
        """
        Create multi-label classification labels for categories.

        Returns binary vector indicating presence of each category.
        """
        num_categories = len(SemEvalDataLoader.CATEGORY_MAP)
        labels = [0.0] * num_categories

        for cat in aspect_categories:
            if cat.category in SemEvalDataLoader.CATEGORY_MAP:
                cat_idx = SemEvalDataLoader.CATEGORY_MAP[cat.category]
                labels[cat_idx] = 1.0

        return labels


def create_dataloaders(
    train_path: str,
    test_path: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 16,
    max_length: int = 128,
    num_workers: int = 0
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and test dataloaders.

    Args:
        train_path: Path to training data
        test_path: Path to test data
        tokenizer: Tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of workers for data loading

    Returns:
        train_loader, test_loader
    """
    # Load data
    train_loader_obj = SemEvalDataLoader(train_path)
    train_reviews = train_loader_obj.parse_reviews()

    test_loader_obj = SemEvalDataLoader(test_path)
    test_reviews = test_loader_obj.parse_reviews()

    # Create datasets
    train_dataset = ABSADataset(train_reviews, tokenizer, max_length)
    test_dataset = ABSADataset(test_reviews, tokenizer, max_length)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader


if __name__ == "__main__":
    # Test preprocessing
    loader = SemEvalDataLoader("data/semeval2014_restaurants_train.csv")
    reviews = loader.parse_reviews()
    stats = loader.get_statistics()

    print("Dataset Statistics:")
    print(json.dumps(stats, indent=2))

    print(f"\nFirst review:")
    print(f"Text: {reviews[0].text}")
    print(f"Aspect terms: {[(a.term, a.polarity) for a in reviews[0].aspect_terms]}")
    print(f"Categories: {[(c.category, c.polarity) for c in reviews[0].aspect_categories]}")
