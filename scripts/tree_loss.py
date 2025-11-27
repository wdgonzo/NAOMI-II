"""
Tree Composition Loss Functions

Loss functions for structure-aware embedding training:
1. Composition Loss: Ensures similar parse trees have similar encodings
2. Distance Loss: Enforces pairwise distance constraints
3. Combined Loss: Weighted combination of both
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List


class CompositionLoss(nn.Module):
    """
    Loss for tree composition.

    Encourages similar tree structures to have similar encodings.
    Uses contrastive loss: similar trees should be close, different trees should be far.
    """

    def __init__(self, margin: float = 1.0):
        """
        Initialize composition loss.

        Args:
            margin: Margin for contrastive loss
        """
        super().__init__()
        self.margin = margin

    def forward(self, encoded_trees: torch.Tensor, indices: List[int]) -> torch.Tensor:
        """
        Compute composition loss for a batch of encoded trees.

        Args:
            encoded_trees: Tensor of shape (batch_size, embedding_dim)
            indices: List of sentence indices (for identifying pairs)

        Returns:
            Composition loss value
        """
        batch_size = encoded_trees.size(0)

        if batch_size < 2:
            # Need at least 2 samples for contrastive loss
            return torch.tensor(0.0, device=encoded_trees.device)

        # Compute pairwise distances
        # encoded_trees: (batch_size, embedding_dim)
        # Normalize for cosine similarity
        normalized = nn.functional.normalize(encoded_trees, dim=1)

        # Compute similarity matrix (cosine similarity)
        similarity = torch.mm(normalized, normalized.t())  # (batch_size, batch_size)

        # For now, we use a simple self-reconstruction loss
        # TODO: Use actual similar/dissimilar pairs from data
        # Target: Each tree should be similar to itself, dissimilar to others
        target = torch.eye(batch_size, device=encoded_trees.device)

        # MSE loss between similarity matrix and identity
        loss = nn.functional.mse_loss(similarity, target)

        return loss


class DistanceLoss(nn.Module):
    """
    Distance-based loss function for word embeddings.

    Enforces semantic distance constraints from knowledge graph.
    """

    def forward(self, embeddings: torch.Tensor, batch: Dict) -> torch.Tensor:
        """
        Compute distance loss.

        Args:
            embeddings: Word embeddings (vocab_size, embedding_dim)
            batch: Batch with source_id, target_id, target_distance, confidence

        Returns:
            Distance loss value
        """
        source_embeds = embeddings[batch['source_id']]
        target_embeds = embeddings[batch['target_id']]

        # Compute actual distances (normalized)
        diff = source_embeds - target_embeds
        actual_dists = torch.norm(diff, dim=1) / np.sqrt(embeddings.shape[1])

        # Compute error
        errors = (actual_dists - batch['target_distance']) ** 2
        weighted_errors = errors * batch['confidence']

        return weighted_errors.mean()


class RegularizationLoss(nn.Module):
    """
    Regularization losses for training stability.

    Includes:
    - L2 regularization on embeddings
    - Orthogonality constraint for anchor dimensions
    """

    def __init__(self, l2_weight: float = 0.0001, ortho_weight: float = 0.001,
                 num_anchors: int = 51):
        """
        Initialize regularization loss.

        Args:
            l2_weight: Weight for L2 regularization
            ortho_weight: Weight for orthogonality constraint
            num_anchors: Number of anchor dimensions
        """
        super().__init__()
        self.l2_weight = l2_weight
        self.ortho_weight = ortho_weight
        self.num_anchors = num_anchors

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute regularization loss.

        Args:
            embeddings: Word embeddings (vocab_size, embedding_dim)

        Returns:
            Regularization loss value
        """
        loss = torch.tensor(0.0, device=embeddings.device)

        # L2 regularization
        if self.l2_weight > 0:
            l2_loss = torch.norm(embeddings, p=2) ** 2 / embeddings.numel()
            loss = loss + self.l2_weight * l2_loss

        # Orthogonality constraint on anchor dimensions
        if self.ortho_weight > 0 and embeddings.size(1) >= self.num_anchors:
            anchor_dims = embeddings[:, :self.num_anchors]  # (vocab_size, num_anchors)
            # Compute gram matrix
            gram = torch.mm(anchor_dims.t(), anchor_dims)  # (num_anchors, num_anchors)
            # Target: diagonal matrix (orthogonal)
            identity = torch.eye(self.num_anchors, device=embeddings.device)
            identity = identity * gram.diag().mean()  # Scale by average norm
            ortho_loss = nn.functional.mse_loss(gram, identity)
            loss = loss + self.ortho_weight * ortho_loss

        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss for structure-aware training.

    Combines:
    1. Composition loss (tree structure)
    2. Distance loss (semantic constraints)
    3. Regularization loss (stability)
    """

    def __init__(self,
                 composition_weight: float = 0.7,
                 distance_weight: float = 0.3,
                 l2_weight: float = 0.0001,
                 ortho_weight: float = 0.001,
                 num_anchors: int = 51):
        """
        Initialize combined loss.

        Args:
            composition_weight: Weight for composition loss
            distance_weight: Weight for distance loss
            l2_weight: Weight for L2 regularization
            ortho_weight: Weight for orthogonality constraint
            num_anchors: Number of anchor dimensions
        """
        super().__init__()
        self.composition_weight = composition_weight
        self.distance_weight = distance_weight

        self.composition_loss = CompositionLoss()
        self.distance_loss = DistanceLoss()
        self.regularization_loss = RegularizationLoss(l2_weight, ortho_weight, num_anchors)

    def forward(self,
                encoded_trees: torch.Tensor,
                tree_indices: List[int],
                word_embeddings: torch.Tensor,
                distance_batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            encoded_trees: Encoded tree representations (batch_size, embedding_dim)
            tree_indices: Indices for tree samples
            word_embeddings: Word embedding matrix (vocab_size, embedding_dim)
            distance_batch: Batch of distance constraints

        Returns:
            Dict with 'total', 'composition', 'distance', 'regularization' losses
        """
        # Composition loss
        comp_loss = self.composition_loss(encoded_trees, tree_indices)

        # Distance loss
        dist_loss = self.distance_loss(word_embeddings, distance_batch)

        # Regularization loss
        reg_loss = self.regularization_loss(word_embeddings)

        # Combined
        total_loss = (self.composition_weight * comp_loss +
                     self.distance_weight * dist_loss +
                     reg_loss)

        return {
            'total': total_loss,
            'composition': comp_loss,
            'distance': dist_loss,
            'regularization': reg_loss
        }


class SimpleCompositionLoss(nn.Module):
    """
    Simplified composition loss for initial training.

    Ensures that tree encodings are well-formed and diverse.
    """

    def forward(self, encoded_trees: torch.Tensor) -> torch.Tensor:
        """
        Compute simple composition loss.

        Encourages:
        1. Unit norm (normalized vectors)
        2. Diversity (not all the same)

        Args:
            encoded_trees: Encoded tree representations (batch_size, embedding_dim)

        Returns:
            Composition loss value
        """
        # Normalize
        norms = torch.norm(encoded_trees, dim=1)
        norm_loss = torch.mean((norms - 1.0) ** 2)

        # Diversity: minimize average pairwise similarity
        if encoded_trees.size(0) > 1:
            normalized = nn.functional.normalize(encoded_trees, dim=1)
            similarity = torch.mm(normalized, normalized.t())
            # Zero out diagonal
            mask = 1.0 - torch.eye(encoded_trees.size(0), device=encoded_trees.device)
            similarity = similarity * mask
            # We want LOW similarity between different trees
            diversity_loss = -torch.mean(torch.abs(similarity))  # Negative to minimize
        else:
            diversity_loss = torch.tensor(0.0, device=encoded_trees.device)

        return norm_loss - 0.1 * diversity_loss
