"""
Evaluation module for ABSA models.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    classification_report,
    confusion_matrix
)
from seqeval.metrics import f1_score, precision_score, recall_score
import json
from collections import defaultdict


def compute_metrics(
    true_labels: List[np.ndarray],
    pred_labels: List[np.ndarray],
    prefix: str = "",
    label_names: List[str] = None
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        true_labels: List of true label arrays
        pred_labels: List of predicted label arrays
        prefix: Prefix for metric names
        label_names: Names of labels for detailed report

    Returns:
        Dictionary of metrics
    """
    # Flatten arrays
    if isinstance(true_labels[0], np.ndarray):
        true_flat = np.concatenate(true_labels)
        pred_flat = np.concatenate(pred_labels)
    else:
        true_flat = np.array(true_labels)
        pred_flat = np.array(pred_labels)

    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_flat,
        pred_flat,
        average='weighted',
        zero_division=0
    )

    accuracy = accuracy_score(true_flat, pred_flat)

    metrics = {
        f'{prefix}precision': precision,
        f'{prefix}recall': recall,
        f'{prefix}f1': f1,
        f'{prefix}accuracy': accuracy
    }

    return metrics


def compute_bio_metrics(
    true_labels: List[List[int]],
    pred_labels: List[List[int]],
    label_map: Dict[int, str] = None
) -> Dict[str, float]:
    """
    Compute metrics for BIO tagging.

    Args:
        true_labels: List of true label sequences
        pred_labels: List of predicted label sequences
        label_map: Mapping from label IDs to names

    Returns:
        Dictionary of metrics
    """
    if label_map is None:
        label_map = {0: 'O', 1: 'B-ASPECT', 2: 'I-ASPECT'}

    # Convert to label names
    true_tags = [[label_map.get(label, 'O') for label in seq] for seq in true_labels]
    pred_tags = [[label_map.get(label, 'O') for label in seq] for seq in pred_labels]

    # Compute metrics using seqeval
    precision = precision_score(true_tags, pred_tags)
    recall = recall_score(true_tags, pred_tags)
    f1 = f1_score(true_tags, pred_tags)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def extract_aspects_from_bio(
    tokens: List[str],
    labels: List[int],
    label_map: Dict[int, str] = None
) -> List[Tuple[str, int, int]]:
    """
    Extract aspect terms from BIO labels.

    Args:
        tokens: List of tokens
        labels: List of BIO labels
        label_map: Mapping from label IDs to names

    Returns:
        List of (aspect_term, start_idx, end_idx) tuples
    """
    if label_map is None:
        label_map = {0: 'O', 1: 'B-ASPECT', 2: 'I-ASPECT'}

    aspects = []
    current_aspect = []
    start_idx = -1

    for i, (token, label) in enumerate(zip(tokens, labels)):
        label_name = label_map.get(label, 'O')

        if label_name == 'B-ASPECT':
            # Save previous aspect if exists
            if current_aspect:
                aspects.append((' '.join(current_aspect), start_idx, i))

            # Start new aspect
            current_aspect = [token]
            start_idx = i

        elif label_name == 'I-ASPECT' and current_aspect:
            # Continue current aspect
            current_aspect.append(token)

        else:  # 'O' or start new sequence
            # Save previous aspect if exists
            if current_aspect:
                aspects.append((' '.join(current_aspect), start_idx, i))
                current_aspect = []
                start_idx = -1

    # Save last aspect if exists
    if current_aspect:
        aspects.append((' '.join(current_aspect), start_idx, len(tokens)))

    return aspects


class ErrorAnalyzer:
    """
    Analyze model errors for detailed insights.
    """

    def __init__(self):
        """Initialize error analyzer."""
        self.errors = []
        self.error_categories = defaultdict(int)

    def analyze_prediction(
        self,
        text: str,
        true_aspects: List[Dict],
        pred_aspects: List[Dict],
        true_sentiments: List[str],
        pred_sentiments: List[str]
    ):
        """
        Analyze a single prediction.

        Args:
            text: Input text
            true_aspects: Ground truth aspects
            pred_aspects: Predicted aspects
            true_sentiments: Ground truth sentiments
            pred_sentiments: Predicted sentiments
        """
        # Check for aspect extraction errors
        true_aspect_set = {(a['term'], a['start']) for a in true_aspects}
        pred_aspect_set = {(a['term'], a['start']) for a in pred_aspects}

        # False negatives (missed aspects)
        missed_aspects = true_aspect_set - pred_aspect_set
        for aspect, start in missed_aspects:
            self.errors.append({
                'text': text,
                'type': 'false_negative',
                'aspect': aspect,
                'position': start,
                'category': self._categorize_error(text, aspect, 'missed')
            })
            self.error_categories['false_negative'] += 1

        # False positives (spurious aspects)
        spurious_aspects = pred_aspect_set - true_aspect_set
        for aspect, start in spurious_aspects:
            self.errors.append({
                'text': text,
                'type': 'false_positive',
                'aspect': aspect,
                'position': start,
                'category': self._categorize_error(text, aspect, 'spurious')
            })
            self.error_categories['false_positive'] += 1

        # Sentiment classification errors
        for true_asp, pred_asp in zip(true_aspects, pred_aspects):
            if (true_asp['term'] == pred_asp['term'] and
                true_asp.get('sentiment') != pred_asp.get('sentiment')):
                self.errors.append({
                    'text': text,
                    'type': 'sentiment_error',
                    'aspect': true_asp['term'],
                    'true_sentiment': true_asp.get('sentiment'),
                    'pred_sentiment': pred_asp.get('sentiment'),
                    'category': self._categorize_sentiment_error(
                        text, true_asp['term'],
                        true_asp.get('sentiment'),
                        pred_asp.get('sentiment')
                    )
                })
                self.error_categories['sentiment_error'] += 1

    def _categorize_error(
        self,
        text: str,
        aspect: str,
        error_type: str
    ) -> str:
        """
        Categorize aspect extraction error.

        Args:
            text: Input text
            aspect: Aspect term
            error_type: Type of error

        Returns:
            Error category
        """
        text_lower = text.lower()
        aspect_lower = aspect.lower()

        # Check for implicit aspects
        if aspect_lower not in text_lower:
            return 'implicit_aspect'

        # Check for multi-word aspects
        if len(aspect.split()) > 1:
            return 'multi_word_aspect'

        # Check for aspects in complex sentences
        if ',' in text or ';' in text or ' but ' in text_lower or ' however ' in text_lower:
            return 'complex_sentence'

        # Check for negation
        negation_words = ['not', "n't", 'no', 'never', 'neither']
        if any(neg in text_lower.split() for neg in negation_words):
            return 'negation_present'

        return 'other'

    def _categorize_sentiment_error(
        self,
        text: str,
        aspect: str,
        true_sentiment: str,
        pred_sentiment: str
    ) -> str:
        """
        Categorize sentiment classification error.

        Args:
            text: Input text
            aspect: Aspect term
            true_sentiment: True sentiment
            pred_sentiment: Predicted sentiment

        Returns:
            Error category
        """
        text_lower = text.lower()

        # Check for negation
        negation_words = ['not', "n't", 'no', 'never', 'neither']
        if any(neg in text_lower for neg in negation_words):
            return 'negation_handling'

        # Check for contrast
        contrast_words = ['but', 'however', 'although', 'though', 'yet']
        if any(contrast in text_lower for contrast in contrast_words):
            return 'contrast_handling'

        # Check for sarcasm/irony indicators
        if '!' in text or text.endswith('...'):
            return 'sarcasm_or_emphasis'

        # Check for neutral confused with positive/negative
        if true_sentiment == 'neutral' or pred_sentiment == 'neutral':
            return 'neutral_boundary'

        # Check for opposite polarity confusion
        if (true_sentiment == 'positive' and pred_sentiment == 'negative') or \
           (true_sentiment == 'negative' and pred_sentiment == 'positive'):
            return 'polarity_reversal'

        return 'other_sentiment'

    def get_summary(self) -> Dict:
        """
        Get error analysis summary.

        Returns:
            Dictionary with error statistics
        """
        total_errors = len(self.errors)

        # Count error categories
        category_counts = defaultdict(int)
        for error in self.errors:
            category_counts[error['category']] += 1

        # Sort by frequency
        sorted_categories = sorted(
            category_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        summary = {
            'total_errors': total_errors,
            'error_types': dict(self.error_categories),
            'error_categories': dict(sorted_categories),
            'error_breakdown': {
                'false_negative_rate': self.error_categories['false_negative'] / total_errors if total_errors > 0 else 0,
                'false_positive_rate': self.error_categories['false_positive'] / total_errors if total_errors > 0 else 0,
                'sentiment_error_rate': self.error_categories['sentiment_error'] / total_errors if total_errors > 0 else 0
            }
        }

        return summary

    def get_error_examples(
        self,
        category: str = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Get error examples.

        Args:
            category: Filter by error category
            limit: Maximum number of examples

        Returns:
            List of error examples
        """
        if category:
            examples = [e for e in self.errors if e['category'] == category]
        else:
            examples = self.errors

        return examples[:limit]

    def save_analysis(self, output_path: str):
        """
        Save error analysis to file.

        Args:
            output_path: Path to save analysis
        """
        summary = self.get_summary()

        # Add example errors for each category
        summary['example_errors'] = {}
        for category in summary['error_categories'].keys():
            examples = self.get_error_examples(category, limit=5)
            summary['example_errors'][category] = examples

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Error analysis saved to {output_path}")


def evaluate_model(
    model,
    dataloader,
    device: str = 'cuda',
    return_predictions: bool = False
) -> Dict:
    """
    Evaluate model on dataset.

    Args:
        model: Model to evaluate
        dataloader: Data loader
        device: Device to use
        return_predictions: Whether to return predictions

    Returns:
        Dictionary with evaluation metrics and optionally predictions
    """
    model.eval()
    all_aspect_preds = []
    all_aspect_labels = []
    all_sentiment_preds = []
    all_sentiment_labels = []
    all_texts = []
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )

            # Get predictions
            aspect_logits = outputs['aspect_logits']
            sentiment_logits = outputs['sentiment_logits']

            aspect_preds = torch.argmax(aspect_logits, dim=-1)
            sentiment_preds = torch.argmax(sentiment_logits, dim=-1)

            # Collect predictions
            for i in range(batch['input_ids'].size(0)):
                mask = batch['attention_mask'][i].bool()

                aspect_pred = aspect_preds[i][mask].cpu().numpy()
                aspect_label = batch['aspect_labels'][i][mask].cpu().numpy()
                sentiment_pred = sentiment_preds[i][mask].cpu().numpy()
                sentiment_label = batch['sentiment_labels'][i][mask].cpu().numpy()

                all_aspect_preds.append(aspect_pred)
                all_aspect_labels.append(aspect_label)

                # Filter sentiment labels
                valid_idx = sentiment_label != -100
                if valid_idx.any():
                    all_sentiment_preds.append(sentiment_pred[valid_idx])
                    all_sentiment_labels.append(sentiment_label[valid_idx])

                if return_predictions:
                    all_texts.append(batch.get('text', [''])[i])
                    predictions.append({
                        'text': batch.get('text', [''])[i],
                        'aspect_pred': aspect_pred.tolist(),
                        'aspect_true': aspect_label.tolist(),
                        'sentiment_pred': sentiment_pred.tolist(),
                        'sentiment_true': sentiment_label.tolist()
                    })

    # Compute metrics
    aspect_metrics = compute_bio_metrics(all_aspect_labels, all_aspect_preds)

    sentiment_metrics = {}
    if all_sentiment_labels:
        sentiment_preds_flat = np.concatenate(all_sentiment_preds)
        sentiment_labels_flat = np.concatenate(all_sentiment_labels)
        sentiment_metrics = compute_metrics(
            [sentiment_labels_flat],
            [sentiment_preds_flat],
            prefix="sentiment_"
        )

    results = {
        **aspect_metrics,
        **sentiment_metrics
    }

    if return_predictions:
        results['predictions'] = predictions

    return results


if __name__ == "__main__":
    # Test error analyzer
    analyzer = ErrorAnalyzer()

    # Example prediction
    text = "The food was great but the service was terrible."
    true_aspects = [
        {'term': 'food', 'start': 4, 'sentiment': 'positive'},
        {'term': 'service', 'start': 30, 'sentiment': 'negative'}
    ]
    pred_aspects = [
        {'term': 'food', 'start': 4, 'sentiment': 'positive'},
        {'term': 'service', 'start': 30, 'sentiment': 'positive'}  # Wrong sentiment
    ]

    analyzer.analyze_prediction(text, true_aspects, pred_aspects, [], [])

    summary = analyzer.get_summary()
    print(json.dumps(summary, indent=2))
