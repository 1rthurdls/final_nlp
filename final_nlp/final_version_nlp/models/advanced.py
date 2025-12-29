"""
Advanced BERT-based models for ABSA.
Includes joint learning and multi-task architectures.
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from typing import Optional, Tuple, Dict
import torch.nn.functional as F


class BertForABSA(BertPreTrainedModel):
    """
    BERT-based joint model for aspect extraction and sentiment classification.
    Multi-task learning approach.
    """

    def __init__(self, config, num_aspect_labels=3, num_sentiment_labels=3):
        """
        Initialize BERT ABSA model.

        Args:
            config: BERT configuration
            num_aspect_labels: Number of aspect labels (BIO tags)
            num_sentiment_labels: Number of sentiment labels
        """
        super().__init__(config)

        self.num_aspect_labels = num_aspect_labels
        self.num_sentiment_labels = num_sentiment_labels

        # BERT encoder
        self.bert = BertModel(config)

        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Aspect extraction head (sequence labeling)
        self.aspect_classifier = nn.Linear(config.hidden_size, num_aspect_labels)

        # Sentiment classification head (sequence labeling)
        self.sentiment_classifier = nn.Linear(config.hidden_size, num_sentiment_labels)

        # Initialize weights
        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        aspect_labels: Optional[torch.Tensor] = None,
        sentiment_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]
            aspect_labels: Aspect labels [batch_size, seq_len]
            sentiment_labels: Sentiment labels [batch_size, seq_len]
            return_dict: Whether to return dict

        Returns:
            Dictionary with logits and optionally loss
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        sequence_output = outputs[0]  # [batch, seq_len, hidden_size]
        sequence_output = self.dropout(sequence_output)

        # Aspect extraction logits
        aspect_logits = self.aspect_classifier(sequence_output)

        # Sentiment classification logits
        sentiment_logits = self.sentiment_classifier(sequence_output)

        # Compute losses
        total_loss = None
        aspect_loss = None
        sentiment_loss = None

        if aspect_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = aspect_logits.view(-1, self.num_aspect_labels)
                active_labels = torch.where(
                    active_loss,
                    aspect_labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(aspect_labels)
                )
                aspect_loss = loss_fct(active_logits, active_labels)
            else:
                aspect_loss = loss_fct(
                    aspect_logits.view(-1, self.num_aspect_labels),
                    aspect_labels.view(-1)
                )

        if sentiment_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            sentiment_loss = loss_fct(
                sentiment_logits.view(-1, self.num_sentiment_labels),
                sentiment_labels.view(-1)
            )

        if aspect_loss is not None and sentiment_loss is not None:
            # Multi-task learning: combine losses
            total_loss = aspect_loss + sentiment_loss

        return {
            'loss': total_loss,
            'aspect_loss': aspect_loss,
            'sentiment_loss': sentiment_loss,
            'aspect_logits': aspect_logits,
            'sentiment_logits': sentiment_logits,
            'hidden_states': sequence_output
        }


class BertForABSAWithCRF(BertPreTrainedModel):
    """
    BERT-based model with CRF layer for better sequence labeling.
    """

    def __init__(self, config, num_aspect_labels=3, num_sentiment_labels=3):
        """
        Initialize BERT-CRF ABSA model.

        Args:
            config: BERT configuration
            num_aspect_labels: Number of aspect labels
            num_sentiment_labels: Number of sentiment labels
        """
        super().__init__(config)

        self.num_aspect_labels = num_aspect_labels
        self.num_sentiment_labels = num_sentiment_labels

        # BERT encoder
        self.bert = BertModel(config)

        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Aspect extraction with CRF
        self.aspect_hidden = nn.Linear(config.hidden_size, num_aspect_labels)
        self.aspect_crf = CRF(num_aspect_labels)

        # Sentiment classification
        self.sentiment_classifier = nn.Linear(config.hidden_size, num_sentiment_labels)

        # Initialize weights
        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        aspect_labels: Optional[torch.Tensor] = None,
        sentiment_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        # Aspect extraction with CRF
        aspect_emissions = self.aspect_hidden(sequence_output)

        # Sentiment classification
        sentiment_logits = self.sentiment_classifier(sequence_output)

        total_loss = None
        aspect_loss = None
        sentiment_loss = None

        if aspect_labels is not None:
            # CRF loss
            aspect_loss = -self.aspect_crf(aspect_emissions, aspect_labels, attention_mask)

        if sentiment_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            sentiment_loss = loss_fct(
                sentiment_logits.view(-1, self.num_sentiment_labels),
                sentiment_labels.view(-1)
            )

        if aspect_loss is not None and sentiment_loss is not None:
            total_loss = aspect_loss + sentiment_loss

        # Decode aspect labels
        aspect_predictions = None
        if attention_mask is not None:
            aspect_predictions = self.aspect_crf.decode(aspect_emissions, attention_mask)

        return {
            'loss': total_loss,
            'aspect_loss': aspect_loss,
            'sentiment_loss': sentiment_loss,
            'aspect_predictions': aspect_predictions,
            'sentiment_logits': sentiment_logits,
            'hidden_states': sequence_output
        }


class CRF(nn.Module):
    """
    Conditional Random Field layer for sequence labeling.
    """

    def __init__(self, num_tags: int, batch_first: bool = True):
        """
        Initialize CRF.

        Args:
            num_tags: Number of tags
            batch_first: Whether batch is first dimension
        """
        super().__init__()

        self.num_tags = num_tags
        self.batch_first = batch_first

        # Transition parameters: transitions[i, j] = score of transitioning from j to i
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        # Start and end transitions
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.ByteTensor] = None,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood.

        Args:
            emissions: Emission scores [batch_size, seq_len, num_tags]
            tags: True tags [batch_size, seq_len]
            mask: Mask [batch_size, seq_len]
            reduction: Reduction method

        Returns:
            Negative log-likelihood
        """
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8, device=tags.device)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # Compute log likelihood
        numerator = self._compute_score(emissions, tags, mask)
        denominator = self._compute_normalizer(emissions, mask)
        llh = numerator - denominator

        if reduction == 'mean':
            return llh.mean()
        elif reduction == 'sum':
            return llh.sum()
        else:
            return llh

    def decode(
        self,
        emissions: torch.Tensor,
        mask: Optional[torch.ByteTensor] = None
    ) -> list:
        """
        Viterbi decoding.

        Args:
            emissions: Emission scores [batch_size, seq_len, num_tags]
            mask: Mask [batch_size, seq_len]

        Returns:
            Best tag sequences
        """
        if mask is None:
            mask = torch.ones(
                emissions.shape[:2],
                dtype=torch.uint8,
                device=emissions.device
            )

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _compute_score(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.ByteTensor
    ) -> torch.Tensor:
        """Compute score of a tag sequence."""
        seq_len, batch_size = tags.shape
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_len):
            score += self.transitions[tags[i], tags[i-1]] * mask[i]
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        seq_ends = mask.long().sum(dim=0) - 1
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(
        self,
        emissions: torch.Tensor,
        mask: torch.ByteTensor
    ) -> torch.Tensor:
        """Compute partition function using forward algorithm."""
        seq_len = emissions.size(0)

        # Initialize alpha with start transitions and first emissions
        alpha = self.start_transitions + emissions[0]

        for i in range(1, seq_len):
            # Broadcast
            emit_scores = emissions[i].unsqueeze(1)  # [batch, 1, num_tags]
            trans_scores = self.transitions.unsqueeze(0)  # [1, num_tags, num_tags]
            alpha_t = alpha.unsqueeze(2)  # [batch, num_tags, 1]

            # Combine
            scores = alpha_t + emit_scores + trans_scores

            # Log-sum-exp
            alpha = torch.logsumexp(scores, dim=1)

            # Apply mask
            alpha = torch.where(mask[i].unsqueeze(1), alpha, alpha)

        # Add end transitions
        return torch.logsumexp(alpha + self.end_transitions, dim=1)

    def _viterbi_decode(
        self,
        emissions: torch.Tensor,
        mask: torch.ByteTensor
    ) -> list:
        """Viterbi decoding algorithm."""
        seq_len, batch_size = emissions.shape[:2]

        # Initialize
        viterbi = self.start_transitions + emissions[0]
        backpointers = []

        # Forward pass
        for i in range(1, seq_len):
            # Broadcast
            viterbi_t = viterbi.unsqueeze(2)  # [batch, num_tags, 1]
            trans = self.transitions.unsqueeze(0)  # [1, num_tags, num_tags]

            # Compute scores
            scores = viterbi_t + trans

            # Max over previous tags
            best_scores, best_tags = scores.max(dim=1)

            # Add emissions
            viterbi = best_scores + emissions[i]

            backpointers.append(best_tags)

        # Add end transitions
        viterbi += self.end_transitions

        # Backtrack
        best_paths = []
        for b in range(batch_size):
            # Find best last tag
            seq_len_b = mask[:, b].sum()
            best_last_tag = viterbi[b].argmax()

            # Backtrack
            path = [best_last_tag.item()]
            for bp in reversed(backpointers[:seq_len_b-1]):
                best_last_tag = bp[b, best_last_tag]
                path.append(best_last_tag.item())

            path.reverse()
            best_paths.append(path)

        return best_paths


class BertForAspectCategoryDetection(BertPreTrainedModel):
    """
    BERT-based model for aspect category detection (multi-label classification).
    """

    def __init__(self, config, num_categories=5):
        """
        Initialize model.

        Args:
            config: BERT configuration
            num_categories: Number of aspect categories
        """
        super().__init__(config)

        self.num_categories = num_categories

        # BERT encoder
        self.bert = BertModel(config)

        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Classification head
        self.classifier = nn.Linear(config.hidden_size, num_categories)

        # Initialize weights
        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Use [CLS] token representation
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        # Multi-label classification
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs[0]
        }
