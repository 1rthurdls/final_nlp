"""
Baseline models for ABSA.
Includes rule-based and BiLSTM approaches.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import re
from collections import defaultdict
import numpy as np


class RuleBasedABSA:
    """
    Rule-based baseline using pattern matching and sentiment lexicons.
    """

    def __init__(self):
        """Initialize rule-based model."""
        # Simple sentiment lexicon
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'delicious', 'perfect', 'best', 'love', 'loved', 'awesome',
            'superb', 'outstanding', 'incredible', 'brilliant', 'nice',
            'friendly', 'fresh', 'tasty', 'delightful', 'exceptional'
        }

        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'poor',
            'disappointing', 'disgusting', 'nasty', 'rude', 'slow',
            'cold', 'overpriced', 'bland', 'mediocre', 'unpleasant',
            'dirty', 'unfriendly', 'stale', 'greasy', 'tasteless'
        }

        # Aspect keywords
        self.aspect_keywords = {
            'food': {'food', 'dish', 'meal', 'pizza', 'pasta', 'salad', 'dessert',
                    'appetizer', 'entree', 'cuisine', 'menu', 'plate', 'rice',
                    'chicken', 'beef', 'fish', 'seafood', 'burger', 'sandwich'},
            'service': {'service', 'staff', 'waiter', 'waitress', 'server',
                       'manager', 'host', 'hostess', 'bartender', 'employee'},
            'ambience': {'ambience', 'atmosphere', 'decor', 'interior', 'music',
                        'noise', 'lighting', 'view', 'setting', 'environment'},
            'price': {'price', 'cost', 'expensive', 'cheap', 'value', 'bill',
                     'money', 'worth', 'overpriced', 'affordable', 'pricey'}
        }

        # Negation words
        self.negation_words = {'not', "n't", 'no', 'never', 'neither', 'nor', 'none'}

    def extract_aspects(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Extract aspect terms using keyword matching.

        Args:
            text: Input text

        Returns:
            List of (aspect_term, start, end) tuples
        """
        text_lower = text.lower()
        aspects = []

        for category, keywords in self.aspect_keywords.items():
            for keyword in keywords:
                # Find all occurrences of keyword
                pattern = r'\b' + re.escape(keyword) + r'\b'
                for match in re.finditer(pattern, text_lower):
                    start, end = match.span()
                    aspects.append((text[start:end], start, end))

        # Remove duplicates
        aspects = list(set(aspects))
        return aspects

    def classify_sentiment(self, text: str, aspect_start: int, aspect_end: int) -> str:
        """
        Classify sentiment for an aspect using window-based lexicon matching.

        Args:
            text: Input text
            aspect_start: Aspect start position
            aspect_end: Aspect end position

        Returns:
            Sentiment label: 'positive', 'negative', or 'neutral'
        """
        # Extract window around aspect (5 words before and after)
        words = text.split()
        aspect_words = text[aspect_start:aspect_end].split()

        # Find aspect position in word list
        try:
            aspect_idx = next(i for i, word in enumerate(words)
                            if aspect_words[0].lower() in word.lower())
        except StopIteration:
            aspect_idx = len(words) // 2

        # Extract window
        window_start = max(0, aspect_idx - 5)
        window_end = min(len(words), aspect_idx + 6)
        window = words[window_start:window_end]

        # Check for negation
        has_negation = any(neg in word.lower() for word in window
                          for neg in self.negation_words)

        # Count sentiment words
        pos_count = sum(1 for word in window
                       if word.lower().strip(',.!?') in self.positive_words)
        neg_count = sum(1 for word in window
                       if word.lower().strip(',.!?') in self.negative_words)

        # Apply negation
        if has_negation:
            pos_count, neg_count = neg_count, pos_count

        # Determine sentiment
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        else:
            return 'neutral'

    def predict(self, text: str) -> Dict:
        """
        Predict aspects and sentiments for text.

        Args:
            text: Input text

        Returns:
            Dictionary with predictions
        """
        aspects = self.extract_aspects(text)
        predictions = []

        for aspect_term, start, end in aspects:
            sentiment = self.classify_sentiment(text, start, end)
            predictions.append({
                'aspect': aspect_term,
                'sentiment': sentiment,
                'start': start,
                'end': end
            })

        return {'text': text, 'predictions': predictions}


class BiLSTMABSA(nn.Module):
    """
    BiLSTM baseline model for aspect extraction and sentiment classification.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
        num_aspect_labels: int = 3,  # O, B-ASPECT, I-ASPECT
        num_sentiment_labels: int = 3,  # positive, negative, neutral
        dropout: float = 0.3,
        pretrained_embeddings: torch.Tensor = None
    ):
        """
        Initialize BiLSTM model.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            hidden_dim: Hidden dimension for LSTM
            num_aspect_labels: Number of aspect labels (BIO tags)
            num_sentiment_labels: Number of sentiment labels
            dropout: Dropout rate
            pretrained_embeddings: Pretrained word embeddings
        """
        super(BiLSTMABSA, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)

        # BiLSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Aspect extraction head
        self.aspect_classifier = nn.Linear(hidden_dim * 2, num_aspect_labels)

        # Sentiment classification head
        self.sentiment_classifier = nn.Linear(hidden_dim * 2, num_sentiment_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            aspect_logits: [batch_size, seq_len, num_aspect_labels]
            sentiment_logits: [batch_size, seq_len, num_sentiment_labels]
        """
        # Embedding
        embedded = self.embedding(input_ids)  # [batch, seq_len, embed_dim]
        embedded = self.dropout(embedded)

        # BiLSTM
        lstm_out, _ = self.lstm(embedded)  # [batch, seq_len, hidden*2]
        lstm_out = self.dropout(lstm_out)

        # Classification
        aspect_logits = self.aspect_classifier(lstm_out)
        sentiment_logits = self.sentiment_classifier(lstm_out)

        return aspect_logits, sentiment_logits


class BiLSTMCRF(nn.Module):
    """
    BiLSTM-CRF model for better sequence labeling.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
        num_labels: int = 3,
        dropout: float = 0.3
    ):
        """
        Initialize BiLSTM-CRF model.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            hidden_dim: Hidden dimension
            num_labels: Number of labels
            dropout: Dropout rate
        """
        super(BiLSTMCRF, self).__init__()

        self.num_labels = num_labels

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # BiLSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Linear layer to get emissions
        self.hidden2tag = nn.Linear(hidden_dim * 2, num_labels)

        # CRF transition parameters
        self.transitions = nn.Parameter(torch.randn(num_labels, num_labels))

        # Constraints: cannot transition to START or from END
        self.transitions.data[:, 0] = -10000  # Cannot transition to START
        self.transitions.data[0, :] = -10000  # Cannot transition from START

    def _get_lstm_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get LSTM features."""
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)

        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)

        emissions = self.hidden2tag(lstm_out)
        return emissions

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor = None,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass (compute loss during training).

        Args:
            input_ids: Input token IDs
            labels: True labels
            attention_mask: Attention mask

        Returns:
            loss or emissions
        """
        emissions = self._get_lstm_features(input_ids)

        if labels is not None:
            # Compute CRF loss
            loss = self._crf_loss(emissions, labels, attention_mask)
            return loss
        else:
            return emissions

    def _crf_loss(
        self,
        emissions: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CRF negative log-likelihood loss.

        Args:
            emissions: [batch, seq_len, num_labels]
            labels: [batch, seq_len]
            mask: [batch, seq_len]

        Returns:
            loss
        """
        batch_size, seq_len = labels.shape

        # Score of the true path
        gold_score = self._score_sequence(emissions, labels, mask)

        # Partition function (log sum of all possible paths)
        forward_score = self._forward_algorithm(emissions, mask)

        # Loss = -log(exp(gold_score) / exp(forward_score))
        loss = forward_score - gold_score
        return loss.mean()

    def _score_sequence(
        self,
        emissions: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Score a sequence of labels."""
        batch_size, seq_len = labels.shape
        scores = torch.zeros(batch_size, device=emissions.device)

        for t in range(seq_len):
            if t == 0:
                # Emission score at first position
                scores += emissions[:, t].gather(1, labels[:, t].unsqueeze(1)).squeeze(1)
            else:
                # Emission + transition score
                emit_score = emissions[:, t].gather(1, labels[:, t].unsqueeze(1)).squeeze(1)
                trans_score = self.transitions[labels[:, t-1], labels[:, t]]
                scores += (emit_score + trans_score) * mask[:, t]

        return scores

    def _forward_algorithm(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward algorithm to compute partition function."""
        batch_size, seq_len, num_labels = emissions.shape

        # Initialize alpha
        alpha = emissions[:, 0]  # [batch, num_labels]

        for t in range(1, seq_len):
            # Broadcast and add
            emit_score = emissions[:, t].unsqueeze(1)  # [batch, 1, num_labels]
            trans_score = self.transitions.unsqueeze(0)  # [1, num_labels, num_labels]
            alpha_t = alpha.unsqueeze(2)  # [batch, num_labels, 1]

            # Combine scores
            scores = alpha_t + trans_score + emit_score  # [batch, num_labels, num_labels]

            # Log-sum-exp over previous states
            alpha = torch.logsumexp(scores, dim=1)  # [batch, num_labels]

        # Sum over final states
        return torch.logsumexp(alpha, dim=1)  # [batch]

    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> List[List[int]]:
        """
        Viterbi decoding to find best label sequence.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Best label sequences
        """
        emissions = self._get_lstm_features(input_ids)
        return self._viterbi_decode(emissions, attention_mask)

    def _viterbi_decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor
    ) -> List[List[int]]:
        """Viterbi algorithm for decoding."""
        batch_size, seq_len, num_labels = emissions.shape

        # Initialize
        viterbi = emissions[:, 0]  # [batch, num_labels]
        backpointers = []

        for t in range(1, seq_len):
            # Broadcast
            prev_viterbi = viterbi.unsqueeze(2)  # [batch, num_labels, 1]
            trans = self.transitions.unsqueeze(0)  # [1, num_labels, num_labels]

            # Compute scores
            scores = prev_viterbi + trans  # [batch, num_labels, num_labels]

            # Max over previous states
            best_scores, best_paths = scores.max(dim=1)  # [batch, num_labels]

            # Add emission scores
            viterbi = best_scores + emissions[:, t]

            backpointers.append(best_paths)

        # Backtrack
        best_paths = []
        for b in range(batch_size):
            # Find best final state
            best_last_tag = viterbi[b].argmax().item()

            # Backtrack
            path = [best_last_tag]
            for bp in reversed(backpointers):
                best_last_tag = bp[b, best_last_tag].item()
                path.append(best_last_tag)

            path.reverse()
            best_paths.append(path)

        return best_paths
