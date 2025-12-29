"""
Data Extraction Module for ABSA Datasets

Supports extraction from:
- SemEval-2014 Task 4 (Laptop and Restaurant reviews)
- SemEval-2015 Task 12
- SemEval-2016 Task 5
- Multi-Domain Aspect Extraction dataset

Author: Data Extraction Module
Date: 2025
"""

import xml.etree.ElementTree as ET
import json
import csv
import os
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AspectTerm:
    """Represents an aspect term with its attributes."""
    term: str
    polarity: str
    from_index: Optional[int] = None
    to_index: Optional[int] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AspectCategory:
    """Represents an aspect category with its attributes."""
    category: str
    polarity: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Opinion:
    """Represents an opinion with target and expression."""
    target: Optional[str] = None
    target_from: Optional[int] = None
    target_to: Optional[int] = None
    category: Optional[str] = None
    polarity: Optional[str] = None
    expression: Optional[str] = None
    expression_from: Optional[int] = None
    expression_to: Optional[int] = None

    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class Review:
    """Represents a complete review with all annotations."""
    review_id: str
    text: str
    aspect_terms: List[AspectTerm]
    aspect_categories: List[AspectCategory]
    opinions: List[Opinion]
    domain: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'review_id': self.review_id,
            'text': self.text,
            'aspect_terms': [at.to_dict() for at in self.aspect_terms],
            'aspect_categories': [ac.to_dict() for ac in self.aspect_categories],
            'opinions': [op.to_dict() for op in self.opinions],
            'domain': self.domain
        }


class SemEval2014Extractor:
    """Extractor for SemEval-2014 Task 4 datasets (Laptop and Restaurant)."""

    def __init__(self, domain: str = "restaurant"):
        """
        Initialize the extractor.

        Args:
            domain: Either 'restaurant' or 'laptop'
        """
        self.domain = domain
        logger.info(f"Initialized SemEval-2014 extractor for {domain} domain")

    def extract_from_xml(self, xml_path: str) -> List[Review]:
        """
        Extract reviews from SemEval-2014 XML format.

        Args:
            xml_path: Path to the XML file

        Returns:
            List of Review objects
        """
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"File not found: {xml_path}")

        tree = ET.parse(xml_path)
        root = tree.getroot()
        reviews = []

        for sentence in root.findall('.//sentence'):
            review_id = sentence.get('id', '')
            text_elem = sentence.find('text')
            text = text_elem.text if text_elem is not None and text_elem.text else ""

            # Extract aspect terms
            aspect_terms = []
            aspect_terms_elem = sentence.find('aspectTerms')
            if aspect_terms_elem is not None:
                for aspect_term in aspect_terms_elem.findall('aspectTerm'):
                    term = aspect_term.get('term', '')
                    polarity = aspect_term.get('polarity', 'neutral')
                    from_idx = aspect_term.get('from')
                    to_idx = aspect_term.get('to')

                    aspect_terms.append(AspectTerm(
                        term=term,
                        polarity=polarity,
                        from_index=int(from_idx) if from_idx else None,
                        to_index=int(to_idx) if to_idx else None
                    ))

            # Extract aspect categories
            aspect_categories = []
            aspect_categories_elem = sentence.find('aspectCategories')
            if aspect_categories_elem is not None:
                for aspect_category in aspect_categories_elem.findall('aspectCategory'):
                    category = aspect_category.get('category', '')
                    polarity = aspect_category.get('polarity', 'neutral')

                    aspect_categories.append(AspectCategory(
                        category=category,
                        polarity=polarity
                    ))

            review = Review(
                review_id=review_id,
                text=text,
                aspect_terms=aspect_terms,
                aspect_categories=aspect_categories,
                opinions=[],
                domain=self.domain
            )
            reviews.append(review)

        logger.info(f"Extracted {len(reviews)} reviews from {xml_path}")
        return reviews


class SemEval2015Extractor:
    """Extractor for SemEval-2015 Task 12 datasets."""

    def __init__(self, domain: str = "restaurant"):
        """
        Initialize the extractor.

        Args:
            domain: Domain of the dataset (e.g., 'restaurant', 'laptop', 'hotel')
        """
        self.domain = domain
        logger.info(f"Initialized SemEval-2015 extractor for {domain} domain")

    def extract_from_xml(self, xml_path: str) -> List[Review]:
        """
        Extract reviews from SemEval-2015 XML format.

        Args:
            xml_path: Path to the XML file

        Returns:
            List of Review objects
        """
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"File not found: {xml_path}")

        tree = ET.parse(xml_path)
        root = tree.getroot()
        reviews = []

        # SemEval-2015 has a different structure with Review -> sentences
        for review_elem in root.findall('.//Review'):
            review_id = review_elem.get('rid', '')

            for sentence in review_elem.findall('.//sentence'):
                sent_id = sentence.get('id', review_id)
                text_elem = sentence.find('text')
                text = text_elem.text if text_elem is not None and text_elem.text else ""

                # Extract opinions (SemEval-2015 uses Opinion elements)
                opinions = []
                opinions_elem = sentence.find('Opinions')
                if opinions_elem is not None:
                    for opinion in opinions_elem.findall('Opinion'):
                        target = opinion.get('target', 'NULL')
                        category = opinion.get('category', '')
                        polarity = opinion.get('polarity', 'neutral')
                        from_idx = opinion.get('from')
                        to_idx = opinion.get('to')

                        opinions.append(Opinion(
                            target=target if target != 'NULL' else None,
                            target_from=int(from_idx) if from_idx else None,
                            target_to=int(to_idx) if to_idx else None,
                            category=category,
                            polarity=polarity
                        ))

                review = Review(
                    review_id=sent_id,
                    text=text,
                    aspect_terms=[],
                    aspect_categories=[],
                    opinions=opinions,
                    domain=self.domain
                )
                reviews.append(review)

        logger.info(f"Extracted {len(reviews)} reviews from {xml_path}")
        return reviews


class SemEval2016Extractor:
    """Extractor for SemEval-2016 Task 5 datasets."""

    def __init__(self, domain: str = "restaurant"):
        """
        Initialize the extractor.

        Args:
            domain: Domain of the dataset
        """
        self.domain = domain
        logger.info(f"Initialized SemEval-2016 extractor for {domain} domain")

    def extract_from_xml(self, xml_path: str) -> List[Review]:
        """
        Extract reviews from SemEval-2016 XML format.

        Args:
            xml_path: Path to the XML file

        Returns:
            List of Review objects
        """
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"File not found: {xml_path}")

        tree = ET.parse(xml_path)
        root = tree.getroot()
        reviews = []

        # SemEval-2016 structure similar to 2015 but with enhancements
        for review_elem in root.findall('.//Review'):
            review_id = review_elem.get('rid', '')

            for sentence in review_elem.findall('.//sentence'):
                sent_id = sentence.get('id', review_id)
                text_elem = sentence.find('text')
                text = text_elem.text if text_elem is not None and text_elem.text else ""

                # Extract opinions with target and category
                opinions = []
                opinions_elem = sentence.find('Opinions')
                if opinions_elem is not None:
                    for opinion in opinions_elem.findall('Opinion'):
                        # Target information
                        target = opinion.get('target', 'NULL')
                        target_from = opinion.get('from')
                        target_to = opinion.get('to')

                        # Category information (E#A format: Entity#Attribute)
                        category = opinion.get('category', '')
                        polarity = opinion.get('polarity', 'neutral')

                        opinions.append(Opinion(
                            target=target if target != 'NULL' else None,
                            target_from=int(target_from) if target_from else None,
                            target_to=int(target_to) if target_to else None,
                            category=category,
                            polarity=polarity
                        ))

                review = Review(
                    review_id=sent_id,
                    text=text,
                    aspect_terms=[],
                    aspect_categories=[],
                    opinions=opinions,
                    domain=self.domain
                )
                reviews.append(review)

        logger.info(f"Extracted {len(reviews)} reviews from {xml_path}")
        return reviews


class MultiDomainExtractor:
    """Extractor for Multi-Domain Aspect Extraction datasets."""

    def __init__(self):
        """Initialize the multi-domain extractor."""
        logger.info("Initialized Multi-Domain extractor")

    def extract_from_json(self, json_path: str, domain: Optional[str] = None) -> List[Review]:
        """
        Extract reviews from JSON format.

        Args:
            json_path: Path to the JSON file
            domain: Optional domain specification

        Returns:
            List of Review objects
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"File not found: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        reviews = []

        # Handle different JSON structures
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            # Try common keys
            items = data.get('reviews', data.get('data', data.get('sentences', [])))
        else:
            raise ValueError("Unsupported JSON structure")

        for idx, item in enumerate(items):
            review_id = item.get('id', item.get('sentence_id', f'review_{idx}'))
            text = item.get('text', item.get('sentence', ''))
            item_domain = item.get('domain', domain or 'unknown')

            # Extract aspects
            aspect_terms = []
            aspects = item.get('aspects', item.get('aspect_terms', []))
            for aspect in aspects:
                if isinstance(aspect, dict):
                    aspect_terms.append(AspectTerm(
                        term=aspect.get('term', aspect.get('aspect', '')),
                        polarity=aspect.get('polarity', aspect.get('sentiment', 'neutral')),
                        from_index=aspect.get('from', aspect.get('start')),
                        to_index=aspect.get('to', aspect.get('end'))
                    ))
                elif isinstance(aspect, str):
                    aspect_terms.append(AspectTerm(term=aspect, polarity='neutral'))

            # Extract categories
            aspect_categories = []
            categories = item.get('categories', item.get('aspect_categories', []))
            for category in categories:
                if isinstance(category, dict):
                    aspect_categories.append(AspectCategory(
                        category=category.get('category', ''),
                        polarity=category.get('polarity', 'neutral')
                    ))
                elif isinstance(category, str):
                    aspect_categories.append(AspectCategory(category=category, polarity='neutral'))

            review = Review(
                review_id=review_id,
                text=text,
                aspect_terms=aspect_terms,
                aspect_categories=aspect_categories,
                opinions=[],
                domain=item_domain
            )
            reviews.append(review)

        logger.info(f"Extracted {len(reviews)} reviews from {json_path}")
        return reviews

    def extract_from_csv(self, csv_path: str, domain: Optional[str] = None,
                        text_column: str = 'text',
                        aspect_column: Optional[str] = 'aspect',
                        polarity_column: Optional[str] = 'polarity') -> List[Review]:
        """
        Extract reviews from CSV format.

        Args:
            csv_path: Path to the CSV file
            domain: Optional domain specification
            text_column: Name of the text column
            aspect_column: Name of the aspect column (optional)
            polarity_column: Name of the polarity column (optional)

        Returns:
            List of Review objects
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File not found: {csv_path}")

        reviews = []

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for idx, row in enumerate(reader):
                review_id = row.get('id', row.get('review_id', f'review_{idx}'))
                text = row.get(text_column, '')

                aspect_terms = []
                if aspect_column and aspect_column in row and row[aspect_column]:
                    polarity = row.get(polarity_column, 'neutral') if polarity_column else 'neutral'
                    aspect_terms.append(AspectTerm(
                        term=row[aspect_column],
                        polarity=polarity
                    ))

                review = Review(
                    review_id=review_id,
                    text=text,
                    aspect_terms=aspect_terms,
                    aspect_categories=[],
                    opinions=[],
                    domain=domain or 'unknown'
                )
                reviews.append(review)

        logger.info(f"Extracted {len(reviews)} reviews from {csv_path}")
        return reviews


class DatasetExtractor:
    """Main extractor class that handles all dataset types."""

    def __init__(self):
        """Initialize the dataset extractor."""
        self.extractors = {
            'semeval2014': SemEval2014Extractor,
            'semeval2015': SemEval2015Extractor,
            'semeval2016': SemEval2016Extractor,
            'multidomain': MultiDomainExtractor
        }
        logger.info("Initialized DatasetExtractor")

    def extract(self, file_path: str, dataset_type: str, domain: Optional[str] = None,
                **kwargs) -> List[Review]:
        """
        Extract data from any supported dataset.

        Args:
            file_path: Path to the dataset file
            dataset_type: Type of dataset ('semeval2014', 'semeval2015', 'semeval2016', 'multidomain')
            domain: Domain of the dataset (e.g., 'restaurant', 'laptop')
            **kwargs: Additional arguments for specific extractors

        Returns:
            List of Review objects
        """
        if dataset_type not in self.extractors:
            raise ValueError(f"Unknown dataset type: {dataset_type}. "
                           f"Supported types: {list(self.extractors.keys())}")

        # Initialize appropriate extractor
        if dataset_type == 'multidomain':
            extractor = self.extractors[dataset_type]()
        else:
            extractor = self.extractors[dataset_type](domain=domain or 'restaurant')

        # Determine file type and extract
        file_ext = Path(file_path).suffix.lower()

        if file_ext == '.xml':
            return extractor.extract_from_xml(file_path)
        elif file_ext == '.json':
            if hasattr(extractor, 'extract_from_json'):
                return extractor.extract_from_json(file_path, domain=domain)
            else:
                raise ValueError(f"JSON format not supported for {dataset_type}")
        elif file_ext == '.csv':
            if hasattr(extractor, 'extract_from_csv'):
                return extractor.extract_from_csv(file_path, domain=domain, **kwargs)
            else:
                raise ValueError(f"CSV format not supported for {dataset_type}")
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

    def save_to_json(self, reviews: List[Review], output_path: str):
        """
        Save extracted reviews to JSON format.

        Args:
            reviews: List of Review objects
            output_path: Path to save the JSON file
        """
        data = [review.to_dict() for review in reviews]

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(reviews)} reviews to {output_path}")

    def get_statistics(self, reviews: List[Review]) -> Dict:
        """
        Get statistics about the extracted reviews.

        Args:
            reviews: List of Review objects

        Returns:
            Dictionary with statistics
        """
        total_reviews = len(reviews)
        total_aspect_terms = sum(len(r.aspect_terms) for r in reviews)
        total_aspect_categories = sum(len(r.aspect_categories) for r in reviews)
        total_opinions = sum(len(r.opinions) for r in reviews)

        domains = set(r.domain for r in reviews if r.domain)

        # Polarity distribution
        polarity_counts = {}
        for review in reviews:
            for aspect in review.aspect_terms:
                polarity_counts[aspect.polarity] = polarity_counts.get(aspect.polarity, 0) + 1
            for opinion in review.opinions:
                if opinion.polarity:
                    polarity_counts[opinion.polarity] = polarity_counts.get(opinion.polarity, 0) + 1

        stats = {
            'total_reviews': total_reviews,
            'total_aspect_terms': total_aspect_terms,
            'total_aspect_categories': total_aspect_categories,
            'total_opinions': total_opinions,
            'domains': list(domains),
            'polarity_distribution': polarity_counts,
            'avg_aspect_terms_per_review': total_aspect_terms / total_reviews if total_reviews > 0 else 0,
            'avg_opinions_per_review': total_opinions / total_reviews if total_reviews > 0 else 0
        }

        return stats


# Convenience functions
def extract_semeval2014(file_path: str, domain: str = 'restaurant') -> List[Review]:
    """
    Extract SemEval-2014 dataset.

    Args:
        file_path: Path to XML file
        domain: 'restaurant' or 'laptop'

    Returns:
        List of Review objects
    """
    extractor = SemEval2014Extractor(domain=domain)
    return extractor.extract_from_xml(file_path)


def extract_semeval2015(file_path: str, domain: str = 'restaurant') -> List[Review]:
    """
    Extract SemEval-2015 dataset.

    Args:
        file_path: Path to XML file
        domain: Domain name

    Returns:
        List of Review objects
    """
    extractor = SemEval2015Extractor(domain=domain)
    return extractor.extract_from_xml(file_path)


def extract_semeval2016(file_path: str, domain: str = 'restaurant') -> List[Review]:
    """
    Extract SemEval-2016 dataset.

    Args:
        file_path: Path to XML file
        domain: Domain name

    Returns:
        List of Review objects
    """
    extractor = SemEval2016Extractor(domain=domain)
    return extractor.extract_from_xml(file_path)


def extract_multidomain(file_path: str, domain: Optional[str] = None) -> List[Review]:
    """
    Extract Multi-Domain dataset.

    Args:
        file_path: Path to JSON or CSV file
        domain: Optional domain name

    Returns:
        List of Review objects
    """
    extractor = MultiDomainExtractor()
    file_ext = Path(file_path).suffix.lower()

    if file_ext == '.json':
        return extractor.extract_from_json(file_path, domain=domain)
    elif file_ext == '.csv':
        return extractor.extract_from_csv(file_path, domain=domain)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")


if __name__ == "__main__":
    # Example usage
    print("ABSA Data Extraction Module")
    print("=" * 50)
    print("\nSupported datasets:")
    print("- SemEval-2014 Task 4 (Laptop and Restaurant)")
    print("- SemEval-2015 Task 12")
    print("- SemEval-2016 Task 5")
    print("- Multi-Domain Aspect Extraction")
    print("\nExample usage:")
    print("""
    from data.extraction import DatasetExtractor

    # Initialize extractor
    extractor = DatasetExtractor()

    # Extract SemEval-2014 Restaurant data
    reviews = extractor.extract('data/semeval2014_restaurant_train.xml',
                               dataset_type='semeval2014',
                               domain='restaurant')

    # Get statistics
    stats = extractor.get_statistics(reviews)
    print(stats)

    # Save to JSON
    extractor.save_to_json(reviews, 'output/semeval2014_restaurant.json')
    """)
