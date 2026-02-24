"""
PyTorch Two-Stage Training with Polarity Constraints

Integrates polarity constraints into the existing PyTorch training pipeline.

Stage 1: Pre-train with distance constraints
Stage 2: Add polarity constraints for antonym pairs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple
import time

from src.embeddings.device import DeviceManager
from src.embeddings.anchors import AnchorDimensions


class DistanceConstraintDataset(Dataset):
    """Dataset of word pairs with target distance constraints."""

    def __init__(self, edges: List[Tuple], vocab_size: int, relation_to_distance: Dict):
        self.edges = edges
        self.vocab_size = vocab_size
        self.samples = []

        for edge in edges:
            if len(edge) == 4:
                source_id, relation_id, target_id, confidence = edge
            elif len(edge) == 3:
                source_id, relation_id, target_id = edge
                confidence = 1.0  # Default confidence
            else:
                continue

            if relation_id in relation_to_distance:
                target_dist = relation_to_distance[relation_id]
                self.samples.append((source_id, target_id, target_dist, confidence))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        source_id, target_id, target_dist, confidence = self.samples[idx]
        return {
            'source_id': source_id,
            'target_id': target_id,
            'target_distance': target_dist,
            'confidence': confidence
        }


class PolarityConstraintDataset(Dataset):
    """Dataset of antonym pairs for polarity constraints."""

    def __init__(self, antonym_pairs: List[Tuple[int, int]]):
        self.pairs = antonym_pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        source_id, target_id = self.pairs[idx]
        return {
            'source_id': source_id,
            'target_id': target_id
        }


def create_relation_distance_map():
    """Map relation types to target distance ranges."""
    return {
        # ===== WordNet Semantic Relations =====
        # Core synonymy and antonymy
        'SYNONYM': 0.1,              # Very close - same meaning
        'ANTONYM': 0.9,              # Far apart - opposite meaning (KEY FOR POLARITY!)

        # Hierarchical (is-a)
        'HYPERNYM': 0.3,             # Parent concept (dog → animal)
        'HYPONYM': 0.3,              # Child concept (animal → dog)

        # Part-whole relations (meronyms/holonyms)
        'PART_MERONYM': 0.4,         # Has-part (car has wheel)
        'SUBSTANCE_MERONYM': 0.4,    # Made-of (table made of wood)
        'MEMBER_MERONYM': 0.4,       # Has-member (forest has trees)
        'PART_HOLONYM': 0.4,         # Part-of (wheel part of car)
        'SUBSTANCE_HOLONYM': 0.4,    # Substance-of (wood substance of table)
        'MEMBER_HOLONYM': 0.4,       # Member-of (tree member of forest)

        # Verb implications
        'ENTAILMENT': 0.35,          # Verb implies another (snore → sleep)
        'CAUSE': 0.35,               # Verb causes another (kill → die)
        'VERB_GROUP': 0.2,           # Verbs in same semantic group

        # Similarity relations
        'SIMILAR_TO': 0.15,          # Similar adjectives (good → fine)
        'ALSO_SEE': 0.4,             # Related concepts

        # Cross-POS relations
        'ATTRIBUTE': 0.5,            # Adjective-noun (heavy → weight)
        'PERTAINYM': 0.3,            # Adjective-noun (facial → face)
        'DERIVATIONALLY_RELATED': 0.25,  # Same root (destruction → destroy)

        # ===== Parse-Based Contextual Relations =====
        'describes': 0.3,            # Descriptive relation
        'is-agent-of': 0.4,          # Agent of action
        'is-patient-of': 0.4,        # Patient of action
        'modifies-manner': 0.4,      # Manner modification
        'modifies-degree': 0.4,      # Degree modification
        'coordinates-with': 0.3,     # Coordination
        'related-to': 0.5,           # Generic relation
    }


class DistanceLoss(nn.Module):
    """Distance-based loss function."""

    def forward(self, embeddings: torch.Tensor, batch: Dict) -> torch.Tensor:
        source_embeds = embeddings[batch['source_id']]
        target_embeds = embeddings[batch['target_id']]

        diff = source_embeds - target_embeds
        actual_dists = torch.norm(diff, dim=1) / np.sqrt(embeddings.shape[1])

        errors = (actual_dists - batch['target_distance']) ** 2
        weighted_errors = errors * batch['confidence']

        return weighted_errors.mean()


class PolarityLoss(nn.Module):
    """
    Integrated polarity loss for antonym pairs.

    Applies to ALL dimensions - network naturally learns which dimensions
    to use for polarity structure (emergent polarity dimensions).
    """

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, embeddings: torch.Tensor, antonym_pairs: List[Tuple[int, int]]) -> torch.Tensor:
        """
        Args:
            embeddings: (vocab_size, embedding_dim) embedding matrix
            antonym_pairs: List of (id1, id2) antonym pairs in this batch's vocabulary

        Returns:
            Polarity loss averaged over all pairs and dimensions
        """
        if not antonym_pairs:
            return torch.tensor(0.0, device=embeddings.device)

        loss = torch.tensor(0.0, device=embeddings.device)

        for id1, id2 in antonym_pairs:
            emb1 = embeddings[id1]
            emb2 = embeddings[id2]

            # Sign product for each dimension
            sign_product = torch.sign(emb1) * torch.sign(emb2)

            # Same sign = BAD (penalty proportional to magnitudes)
            same_sign_penalty = torch.where(
                sign_product > 0,
                torch.abs(emb1) + torch.abs(emb2),
                torch.zeros_like(emb1)
            )

            # Opposite signs = GOOD (reward proportional to magnitudes)
            opposite_sign_reward = torch.where(
                sign_product < 0,
                torch.abs(emb1) + torch.abs(emb2),
                torch.zeros_like(emb1)
            )

            # One is zero = slight penalty
            zero_penalty = torch.where(
                sign_product == 0,
                torch.ones_like(emb1) * 0.1,
                torch.zeros_like(emb1)
            )

            # Total loss for this pair (sum over all dimensions)
            pair_loss = same_sign_penalty - 0.5 * opposite_sign_reward + zero_penalty
            loss += pair_loss.sum()

        # Average over number of pairs
        return (loss / len(antonym_pairs)) * self.weight


class SimpleEmbeddingModel(nn.Module):
    """Simple normalized embedding model."""

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Parameter(torch.randn(vocab_size, embedding_dim))

    def forward(self, normalize=True):
        if normalize:
            return nn.functional.normalize(self.embeddings, p=2, dim=1)
        return self.embeddings


def discover_polarity_dimensions(model: SimpleEmbeddingModel, antonym_pairs: List[Tuple[int, int]],
                                   top_k: int = 10, min_consistency: float = 0.6) -> List[int]:
    """
    Discover dimensions where antonyms have opposite signs.

    Args:
        model: Trained embedding model
        antonym_pairs: List of (word1_id, word2_id) antonym pairs
        top_k: Number of dimensions to return
        min_consistency: Minimum sign consistency threshold

    Returns:
        List of dimension indices
    """
    print(f"Discovering polarity dimensions from {len(antonym_pairs)} antonym pairs...")

    embeddings = model.forward(normalize=False).detach().cpu().numpy()
    embedding_dim = embeddings.shape[1]

    # Calculate signed differences for each dimension
    dim_signed_diffs = [[] for _ in range(embedding_dim)]

    for word1_id, word2_id in antonym_pairs:
        emb1 = embeddings[word1_id]
        emb2 = embeddings[word2_id]

        diff = emb1 - emb2

        for dim_idx in range(embedding_dim):
            dim_signed_diffs[dim_idx].append(diff[dim_idx])

    # Score each dimension
    polarity_scores = {}

    for dim_idx in range(embedding_dim):
        diffs = np.array(dim_signed_diffs[dim_idx])

        # Discriminative power
        discriminative_power = float(np.mean(np.abs(diffs)))

        # Sign consistency
        signs = np.sign(diffs)
        sign_consistency = float(np.abs(np.mean(signs)))

        if sign_consistency < min_consistency:
            continue

        # Combined score
        polarity_scores[dim_idx] = discriminative_power * sign_consistency

    # Return top K dimensions
    sorted_dims = sorted(polarity_scores.items(), key=lambda x: x[1], reverse=True)
    top_dims = [dim for dim, score in sorted_dims[:top_k]]

    print(f"  Found {len(top_dims)} polarity dimensions:")
    for i, dim_idx in enumerate(top_dims[:5], 1):
        score = polarity_scores[dim_idx]
        print(f"    {i}. Dimension {dim_idx}: score={score:.4f}")

    return top_dims


def extract_antonym_pairs(edges: List[Tuple]) -> List[Tuple[int, int]]:
    """Extract antonym pairs from edges (relation_id == 'ANTONYM')."""
    antonym_pairs = []

    for edge in edges:
        if len(edge) >= 3:
            source_id, relation_id, target_id = edge[0], edge[1], edge[2]
            if relation_id == 'ANTONYM':  # Antonym relation
                antonym_pairs.append((source_id, target_id))

    print(f"Extracted {len(antonym_pairs)} antonym pairs from edges")
    return antonym_pairs


def train_stage1(model, dataloader, device, epochs, lr):
    """Stage 1: Pre-train with distance constraints."""
    print("\n" + "=" * 70)
    print("STAGE 1: PRE-TRAINING WITH DISTANCE CONSTRAINTS")
    print("=" * 70)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    distance_loss_fn = DistanceLoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()

            # Forward pass
            embeddings = model(normalize=True)
            loss = distance_loss_fn(embeddings, batch)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            progress.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

    print("\nStage 1 complete!")


def train_stage2(model, distance_dataloader, polarity_dataloader, device, epochs, lr,
                 polarity_dims, polarity_weight):
    """Stage 2: Fine-tune with polarity constraints."""
    print("\n" + "=" * 70)
    print("STAGE 2: FINE-TUNING WITH POLARITY CONSTRAINTS")
    print("=" * 70)
    print(f"Polarity dimensions: {polarity_dims}")
    print(f"Polarity weight: {polarity_weight}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    distance_loss_fn = DistanceLoss()
    polarity_loss_fn = PolarityLoss(polarity_dims, weight=polarity_weight)

    for epoch in range(epochs):
        model.train()
        epoch_dist_loss = 0.0
        epoch_pol_loss = 0.0
        num_batches = 0

        # Interleave distance and polarity batches
        distance_iter = iter(distance_dataloader)
        polarity_iter = iter(polarity_dataloader)

        progress = tqdm(range(max(len(distance_dataloader), len(polarity_dataloader))),
                       desc=f"Epoch {epoch+1}/{epochs}")

        for _ in progress:
            optimizer.zero_grad()

            total_loss = 0.0

            # Distance loss
            try:
                dist_batch = next(distance_iter)
                dist_batch = {k: v.to(device) for k, v in dist_batch.items()}

                embeddings = model(normalize=True)
                dist_loss = distance_loss_fn(embeddings, dist_batch)
                total_loss += dist_loss
                epoch_dist_loss += dist_loss.item()
            except StopIteration:
                distance_iter = iter(distance_dataloader)

            # Polarity loss
            try:
                pol_batch = next(polarity_iter)
                pol_batch = {k: v.to(device) for k, v in pol_batch.items()}

                embeddings = model(normalize=False)  # Don't normalize for polarity
                pol_loss = polarity_loss_fn(embeddings, pol_batch)
                total_loss += pol_loss
                epoch_pol_loss += pol_loss.item()
            except StopIteration:
                polarity_iter = iter(polarity_dataloader)

            # Backward pass
            if total_loss > 0:
                total_loss.backward()
                optimizer.step()
                num_batches += 1

                progress.set_postfix({
                    'dist_loss': f'{dist_loss.item():.4f}' if 'dist_loss' in locals() else 'N/A',
                    'pol_loss': f'{pol_loss.item():.4f}' if 'pol_loss' in locals() else 'N/A'
                })

        avg_dist_loss = epoch_dist_loss / num_batches if num_batches > 0 else 0
        avg_pol_loss = epoch_pol_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1}: dist_loss={avg_dist_loss:.4f}, pol_loss={avg_pol_loss:.4f}")

    print("\nStage 2 complete!")


def train_integrated(model, dataloader, antonym_pairs, device, epochs, lr,
                     distance_weight=1.0, polarity_weight=0.1):
    """
    Single-stage integrated training with distance + polarity loss.

    Args:
        model: Embedding model
        dataloader: DataLoader with distance constraints
        antonym_pairs: List of (id1, id2) antonym pairs
        device: torch device
        epochs: Number of epochs
        lr: Learning rate
        distance_weight: Weight for distance loss (default: 1.0)
        polarity_weight: Weight for polarity loss (default: 0.1)
    """
    print("\n" + "=" * 70)
    print("INTEGRATED TRAINING (Distance + Polarity)")
    print("=" * 70)
    print(f"Antonym pairs: {len(antonym_pairs)}")
    print(f"Distance weight: {distance_weight}")
    print(f"Polarity weight: {polarity_weight}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print("=" * 70)
    print()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    distance_loss_fn = DistanceLoss()
    polarity_loss_fn = PolarityLoss(weight=polarity_weight)

    for epoch in range(epochs):
        model.train()
        epoch_dist_loss = 0.0
        epoch_pol_loss = 0.0
        num_batches = 0

        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()

            # Get embeddings
            embeddings = model(normalize=True)

            # 1. Distance loss (on batch)
            dist_loss = distance_loss_fn(embeddings, batch)

            # 2. Polarity loss (on all antonym pairs)
            # Use unnormalized embeddings for polarity
            embeddings_unnorm = model(normalize=False)
            pol_loss = polarity_loss_fn(embeddings_unnorm, antonym_pairs)

            # Combined loss
            total_loss = distance_weight * dist_loss + pol_loss

            # Backward pass
            total_loss.backward()
            optimizer.step()

            epoch_dist_loss += dist_loss.item()
            epoch_pol_loss += pol_loss.item()
            num_batches += 1

            progress.set_postfix({
                'dist_loss': f'{dist_loss.item():.4f}',
                'pol_loss': f'{pol_loss.item():.4f}',
                'total': f'{total_loss.item():.4f}'
            })

        avg_dist_loss = epoch_dist_loss / num_batches
        avg_pol_loss = epoch_pol_loss / num_batches
        print(f"Epoch {epoch+1}: "
              f"dist_loss={avg_dist_loss:.4f}, "
              f"pol_loss={avg_pol_loss:.4f}")

    print("\nIntegrated training complete!")


def analyze_emerged_polarity_dims(model, antonym_pairs, threshold=0.7):
    """
    Analyze which dimensions emerged as polarity dimensions after integrated training.

    Args:
        model: Trained embedding model
        antonym_pairs: List of (id1, id2) antonym pairs
        threshold: Minimum opposite-sign consistency (default: 0.7)

    Returns:
        List of dimension indices that show polarity structure
    """
    print("\n" + "=" * 70)
    print("ANALYZING EMERGED POLARITY DIMENSIONS")
    print("=" * 70)

    embeddings = model(normalize=False).detach().cpu().numpy()
    embedding_dim = embeddings.shape[1]

    # For each dimension, count opposite-sign consistency
    dim_stats = []

    for dim_idx in range(embedding_dim):
        opposite_count = 0
        same_count = 0
        total_magnitude = 0.0

        for id1, id2 in antonym_pairs:
            val1 = embeddings[id1, dim_idx]
            val2 = embeddings[id2, dim_idx]

            sign_product = np.sign(val1) * np.sign(val2)

            if sign_product < 0:
                opposite_count += 1
            elif sign_product > 0:
                same_count += 1

            total_magnitude += abs(val1) + abs(val2)

        consistency = opposite_count / len(antonym_pairs)
        avg_magnitude = total_magnitude / (2 * len(antonym_pairs))

        dim_stats.append({
            'dim': dim_idx,
            'consistency': consistency,
            'avg_magnitude': avg_magnitude,
            'opposite_count': opposite_count,
            'same_count': same_count
        })

    # Sort by consistency
    dim_stats.sort(key=lambda x: x['consistency'], reverse=True)

    # Filter by threshold
    polarity_dims = [d['dim'] for d in dim_stats if d['consistency'] >= threshold]

    print(f"\nFound {len(polarity_dims)} dimensions with {threshold*100:.0f}%+ opposite-sign consistency:")
    print()
    for i, stats in enumerate(dim_stats[:15], 1):
        marker = " <-- POLARITY!" if stats['dim'] in polarity_dims else ""
        print(f"  {i:2d}. Dim {stats['dim']:3d}: "
              f"consistency={stats['consistency']:.2%}, "
              f"avg_mag={stats['avg_magnitude']:.3f}, "
              f"opposite={stats['opposite_count']}/{len(antonym_pairs)}"
              f"{marker}")

    return polarity_dims


def main():
    parser = argparse.ArgumentParser(description='Integrated polarity-aware training')
    parser.add_argument('--data-dir', type=str, default='data/sense_graph_1k_full_wordnet',
                       help='Directory with training data')
    parser.add_argument('--embedding-dim', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--distance-weight', type=float, default=1.0,
                       help='Weight for distance loss (default: 1.0)')
    parser.add_argument('--polarity-weight', type=float, default=0.1,
                       help='Weight for polarity loss (default: 0.1, try 0.1-0.5)')
    parser.add_argument('--output-dir', type=str, default='checkpoints_integrated',
                       help='Output directory')

    args = parser.parse_args()

    print("=" * 70)
    print("INTEGRATED POLARITY-AWARE TRAINING (PyTorch)")
    print("=" * 70)
    print()

    # Setup device
    device_manager = DeviceManager()
    device = device_manager.device
    print(f"Using device: {device}")
    print()

    # Load data
    print("Loading training data...")
    data_dir = Path(args.data_dir)

    with open(data_dir / 'training_examples.pkl', 'rb') as f:
        edges = pickle.load(f)

    with open(data_dir / 'vocabulary.json', 'r') as f:
        vocab_data = json.load(f)

    vocab_size = len(vocab_data['word_to_id'])
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Training edges: {len(edges)}")
    print()

    # Extract antonym pairs
    antonym_pairs = extract_antonym_pairs(edges)

    if len(antonym_pairs) == 0:
        print("WARNING: No antonym pairs found! Training without polarity loss.")
        antonym_pairs = []

    # Create datasets
    relation_distance_map = create_relation_distance_map()

    distance_dataset = DistanceConstraintDataset(edges, vocab_size, relation_distance_map)
    distance_dataloader = DataLoader(
        distance_dataset, batch_size=args.batch_size, shuffle=True
    )

    print(f"Created distance dataset: {len(distance_dataset)} samples")
    print()

    # Initialize model
    print(f"Initializing model with {args.embedding_dim} dimensions...")
    model = SimpleEmbeddingModel(vocab_size, args.embedding_dim).to(device)
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # INTEGRATED TRAINING: Distance + Polarity together
    train_integrated(
        model, distance_dataloader, antonym_pairs, device,
        args.epochs, args.lr, args.distance_weight, args.polarity_weight
    )

    # Save checkpoint
    checkpoint_path = output_dir / "integrated_polarity.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'embedding_dim': args.embedding_dim,
        'antonym_pairs': antonym_pairs
    }, checkpoint_path)
    print(f"\nSaved checkpoint: {checkpoint_path}")

    # Analyze emerged polarity dimensions
    if len(antonym_pairs) > 0:
        polarity_dims = analyze_emerged_polarity_dims(model, antonym_pairs, threshold=0.7)

        # Save polarity dimensions
        with open(output_dir / "polarity_dimensions.json", 'w') as f:
            json.dump({'polarity_dimensions': polarity_dims}, f, indent=2)
    else:
        polarity_dims = []

    # Save embeddings in numpy format for analysis
    embeddings_np = model.forward(normalize=False).detach().cpu().numpy()
    np.save(output_dir / "embeddings.npy", embeddings_np)

    with open(output_dir / "vocabulary.json", 'w') as f:
        json.dump(vocab_data, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nCheckpoints saved to: {output_dir}")
    print()
    print("Next steps:")
    print(f"  1. Test polarity: python scripts/test_polarity_constraints.py --checkpoint {output_dir}")
    print(f"  2. Analyze dims: python scripts/analyze_dimensions.py --checkpoint-dir {output_dir}")
    print()


if __name__ == "__main__":
    main()
