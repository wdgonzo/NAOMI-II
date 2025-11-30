"""
WordNet Bootstrap Training Script - Phase 1

Trains transparent semantic embeddings on PURE WordNet relations to discover
interpretable polarity dimensions.

Key Differences from Regular Training:
1. **Fixed Polarity Loss** - Uses cosine similarity (magnitude-independent)
2. **Higher Polarity Weight** - 10.0 (vs 1.0) for strong signal
3. **Lower Sparsity Weight** - 0.0001 (vs 0.005) to allow polarity structure
4. **Anchor Initialization** - Can load pre-initialized anchor dimensions
5. **Polarity Discovery** - Lower threshold (0.15 vs 0.30) for early detection

Expected Outcome:
- Discover 10-20 interpretable semantic axes (morality, temperature, size, etc.)
- Each antonym pair opposes on 1-5 specific dimensions (selective polarity)
- Enable NOT(good) ≈ bad compositional semantics

Usage:
    # Step 1: Build WordNet graph
    python scripts/extract_wordnet_only_graph.py

    # Step 2 (optional): Initialize anchors
    python scripts/initialize_anchor_dimensions.py

    # Step 3: Train bootstrap
    python scripts/train_wordnet_bootstrap.py \\
        --graph-dir data/wordnet_only_graph \\
        --init-embeddings checkpoints/initialized_embeddings.npy \\
        --epochs 200 \\
        --embedding-dim 128
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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import time

from src.embeddings.anchors import AnchorDimensions


class SimpleEmbeddingModel(nn.Module):
    """Simple word embedding model."""

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embeddings = nn.Parameter(
            torch.randn(vocab_size, embedding_dim) * 0.01
        )
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

    def forward(self):
        return self.embeddings


class DistanceConstraintDataset(Dataset):
    """Dataset of word pairs with target distance constraints."""

    def __init__(self, edges: List[Tuple], vocab_size: int, relation_to_distance: Dict):
        self.edges = edges
        self.vocab_size = vocab_size
        self.samples = []

        for source_id, relation, target_id in edges:
            if relation in relation_to_distance:
                target_dist = relation_to_distance[relation]
                self.samples.append((source_id, target_id, target_dist, 1.0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        source_id, target_id, target_dist, confidence = self.samples[idx]
        return {
            'source_id': torch.tensor(source_id, dtype=torch.long),
            'target_id': torch.tensor(target_id, dtype=torch.long),
            'target_distance': torch.tensor(target_dist, dtype=torch.float32),
            'confidence': torch.tensor(confidence, dtype=torch.float32)
        }


class DistanceLoss(nn.Module):
    """Distance-based loss function."""

    def forward(self, embeddings: torch.Tensor, batch: Dict) -> torch.Tensor:
        source_embeds = embeddings[batch['source_id']]
        target_embeds = embeddings[batch['target_id']]

        diff = source_embeds - target_embeds
        actual_dists = torch.norm(diff, dim=1)

        errors = (actual_dists - batch['target_distance']) ** 2
        weighted_errors = errors * batch['confidence']

        return weighted_errors.mean()


class FixedPolarityLoss(nn.Module):
    """
    FIXED Polarity Loss - Magnitude-Independent

    Problem with old version: torch.abs(emb1) * torch.abs(emb2) vanishes for small values
    Solution: Use cosine similarity (sign-only, no magnitude dependence)

    Goal: Antonyms should have opposite signs (negative cosine similarity)
    """

    def __init__(self, antonym_pairs: List[Tuple[int, int]], weight: float = 10.0):
        super().__init__()
        self.weight = weight

        if len(antonym_pairs) == 0:
            self.antonym_indices = None
        else:
            self.antonym_indices = torch.tensor(antonym_pairs, dtype=torch.long)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.antonym_indices is None or len(self.antonym_indices) == 0:
            return torch.tensor(0.0, device=embeddings.device)

        # Move antonym indices to same device as embeddings
        if self.antonym_indices.device != embeddings.device:
            self.antonym_indices = self.antonym_indices.to(embeddings.device)

        # Get antonym pair embeddings
        emb1 = embeddings[self.antonym_indices[:, 0]]  # (num_pairs, dim)
        emb2 = embeddings[self.antonym_indices[:, 1]]  # (num_pairs, dim)

        # Use cosine similarity (sign-aware, magnitude-independent)
        # Goal: antonyms should have negative cosine similarity
        cosine_sim = F.cosine_similarity(emb1, emb2, dim=1)  # (num_pairs,)

        # Penalize positive cosine similarity (same direction)
        # Reward negative cosine similarity (opposite direction)
        polarity_loss = torch.mean(cosine_sim)  # Higher = worse (same direction)

        return self.weight * polarity_loss


class SparsityLoss(nn.Module):
    """L1 sparsity regularization."""

    def __init__(self, weight: float = 0.0001, target: float = 0.55):
        super().__init__()
        self.weight = weight
        self.target = target

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # L1 penalty
        l1_loss = torch.mean(torch.abs(embeddings))
        return self.weight * l1_loss


def create_relation_distance_map():
    """Map relation types to target distances."""
    return {
        'SYNONYM': 0.1,
        'HYPERNYM': 0.3,
        'HYPONYM': 0.3,
        'ANTONYM': 1.0,  # Far apart (opposite)
        'SIMILAR_TO': 0.2,
        'MERONYM': 0.5,
        'HOLONYM': 0.5,
        'ENTAILMENT': 0.4,
        'CAUSE': 0.4,
        'ALSO_SEE': 0.6,
        'ATTRIBUTE': 0.5,
        'VERB_GROUP': 0.3,
        'PERTAINYM': 0.4,
        'DERIVATIONALLY_RELATED': 0.3,
    }


def extract_antonym_pairs(edges: List[Tuple[int, str, int]]) -> List[Tuple[int, int]]:
    """Extract antonym pairs from edges."""
    antonym_pairs = []
    for source_id, relation, target_id in edges:
        if relation == 'ANTONYM':
            antonym_pairs.append((source_id, target_id))
    return antonym_pairs


def discover_polarity_dimensions(embeddings: np.ndarray,
                                  antonym_pairs: List[Tuple[int, int]],
                                  min_consistency: float = 0.15,
                                  top_k: int = 20) -> List[int]:
    """
    Discover dimensions showing consistent polarity structure.

    Args:
        embeddings: (vocab_size, embedding_dim)
        antonym_pairs: List of (word1_id, word2_id) antonym pairs
        min_consistency: Minimum sign consistency (lowered from 0.3 to 0.15)
        top_k: Number of top polarity dims to return

    Returns:
        List of dimension indices with strong polarity structure
    """
    if len(antonym_pairs) == 0:
        return []

    num_pairs = len(antonym_pairs)
    num_dims = embeddings.shape[1]

    # For each dimension, check if antonyms consistently have opposite signs
    dim_scores = []

    for dim_idx in range(num_dims):
        signs = []
        for word1_id, word2_id in antonym_pairs:
            val1 = embeddings[word1_id, dim_idx]
            val2 = embeddings[word2_id, dim_idx]

            # Check if opposite signs
            if val1 * val2 < 0:  # Opposite signs
                signs.append(1.0)
            else:
                signs.append(0.0)

        # Consistency = fraction of pairs with opposite signs
        consistency = np.mean(signs)

        if consistency >= min_consistency:
            # Discriminative power = average absolute difference
            diffs = []
            for word1_id, word2_id in antonym_pairs:
                val1 = embeddings[word1_id, dim_idx]
                val2 = embeddings[word2_id, dim_idx]
                diffs.append(abs(val1 - val2))

            discriminative_power = np.mean(diffs)
            score = consistency * discriminative_power

            dim_scores.append((dim_idx, score, consistency, discriminative_power))

    # Sort by score and return top K
    dim_scores.sort(key=lambda x: x[1], reverse=True)
    polarity_dims = [dim_idx for dim_idx, _, _, _ in dim_scores[:top_k]]

    return polarity_dims, dim_scores[:top_k]


def main():
    parser = argparse.ArgumentParser(description='Phase 1: WordNet Bootstrap Training')
    parser.add_argument('--graph-dir', type=str,
                       default='data/wordnet_only_graph',
                       help='Knowledge graph directory')
    parser.add_argument('--output-dir', type=str,
                       default='checkpoints/phase1_bootstrap',
                       help='Output directory for checkpoints')
    parser.add_argument('--init-embeddings', type=str, default=None,
                       help='Pre-initialized embeddings (optional)')
    parser.add_argument('--embedding-dim', type=int, default=128,
                       help='Embedding dimensions')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=1024,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--polarity-weight', type=float, default=10.0,
                       help='Polarity loss weight (high for bootstrap)')
    parser.add_argument('--sparsity-weight', type=float, default=0.0001,
                       help='Sparsity loss weight (low for bootstrap)')
    parser.add_argument('--preserve-anchors', action='store_true',
                       help='Freeze first 51 anchor dimensions')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto/cpu/cuda)')

    args = parser.parse_args()

    print("="*70)
    print("PHASE 1: WORDNET BOOTSTRAP TRAINING")
    print("Goal: Discover Transparent Polarity Dimensions")
    print("="*70)
    print()

    # Device selection
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    print()

    # Load vocabulary
    print("[1/6] Loading vocabulary...")
    graph_dir = Path(args.graph_dir)
    vocab_path = graph_dir / "vocabulary.json"

    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
        word_to_id = vocab_data['word_to_id']
        id_to_word = vocab_data['id_to_word']

    vocab_size = len(word_to_id)
    print(f"  Vocabulary size: {vocab_size:,} sense-tagged words")
    print()

    # Load training examples
    print("[2/6] Loading training data...")
    training_path = graph_dir / "training_examples.pkl"
    with open(training_path, 'rb') as f:
        edges = pickle.load(f)

    print(f"  Training examples: {len(edges):,}")

    # Extract antonym pairs
    antonym_pairs = extract_antonym_pairs(edges)
    print(f"  Antonym pairs: {len(antonym_pairs)} ← KEY FOR POLARITY!")
    print()

    # Create dataset
    print("[3/6] Creating dataset...")
    relation_distances = create_relation_distance_map()
    dataset = DistanceConstraintDataset(edges, vocab_size, relation_distances)
    print(f"  Dataset size: {len(dataset)} samples")

    # Split into train/validation
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    print(f"  Train: {len(train_dataset):,}, Validation: {len(val_dataset):,}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print()

    # Initialize model
    print("[4/6] Initializing embedding model...")
    anchors = AnchorDimensions()
    num_anchors = anchors.num_anchors()
    print(f"  Embedding dim: {args.embedding_dim} ({num_anchors} anchor + {args.embedding_dim - num_anchors} learned)")

    model = SimpleEmbeddingModel(vocab_size, args.embedding_dim).to(device)

    # Load pre-initialized embeddings if provided
    if args.init_embeddings:
        print(f"  Loading pre-initialized embeddings from {args.init_embeddings}...")
        init_embs = np.load(args.init_embeddings)
        model.embeddings.data = torch.from_numpy(init_embs).to(device)
        print(f"  ✓ Loaded initialized embeddings (shape: {init_embs.shape})")

    # Preserve anchor dimensions if requested
    if args.preserve_anchors:
        print(f"  Anchor preservation: ENABLED (first {num_anchors} dims frozen)")
        def preserve_anchors_hook(grad):
            grad_copy = grad.clone()
            grad_copy[:, :num_anchors] = 0.0
            return grad_copy
        model.embeddings.register_hook(preserve_anchors_hook)
    else:
        print(f"  Anchor preservation: DISABLED")

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Initialize optimizer and losses
    print("[5/6] Initializing optimizer and loss functions...")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    distance_criterion = DistanceLoss()
    sparsity_criterion = SparsityLoss(weight=args.sparsity_weight)
    polarity_criterion = FixedPolarityLoss(antonym_pairs, weight=args.polarity_weight)

    print(f"  Optimizer: Adam (lr={args.lr})")
    print(f"  Loss components:")
    print(f"    - Distance loss (WordNet relations)")
    print(f"    - FIXED Polarity loss (cosine similarity, weight={args.polarity_weight})")
    print(f"    - Sparsity loss (L1 penalty, weight={args.sparsity_weight})")
    print()

    # Training loop
    print("[6/6] Training...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        train_losses = {'distance': [], 'polarity': [], 'sparsity': [], 'total': []}

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()

            embeddings = model()

            # Compute losses
            distance_loss = distance_criterion(embeddings, batch)
            polarity_loss = polarity_criterion(embeddings)
            sparsity_loss = sparsity_criterion(embeddings)

            total_loss = distance_loss + polarity_loss + sparsity_loss

            total_loss.backward()
            optimizer.step()

            train_losses['distance'].append(distance_loss.item())
            train_losses['polarity'].append(polarity_loss.item())
            train_losses['sparsity'].append(sparsity_loss.item())
            train_losses['total'].append(total_loss.item())

        # Validation
        model.eval()
        val_losses = {'distance': [], 'polarity': [], 'sparsity': [], 'total': []}

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                embeddings = model()

                distance_loss = distance_criterion(embeddings, batch)
                polarity_loss = polarity_criterion(embeddings)
                sparsity_loss = sparsity_criterion(embeddings)
                total_loss = distance_loss + polarity_loss + sparsity_loss

                val_losses['distance'].append(distance_loss.item())
                val_losses['polarity'].append(polarity_loss.item())
                val_losses['sparsity'].append(sparsity_loss.item())
                val_losses['total'].append(total_loss.item())

        # Compute averages
        train_avg = {k: np.mean(v) for k, v in train_losses.items()}
        val_avg = {k: np.mean(v) for k, v in val_losses.items()}

        # Polarity dimension discovery (every 10 epochs)
        polarity_dims = []
        if epoch % 10 == 0:
            embeddings_np = model.embeddings.detach().cpu().numpy()
            polarity_dims, dim_scores = discover_polarity_dimensions(
                embeddings_np, antonym_pairs, min_consistency=0.15
            )

            if len(polarity_dims) > 0:
                print(f"\n  [POLARITY DISCOVERY] Found {len(polarity_dims)} polarity dimensions:")
                for dim_idx, score, consistency, disc_power in dim_scores[:5]:
                    print(f"    Dim {dim_idx}: score={score:.4f}, consistency={consistency:.2%}, power={disc_power:.4f}")
            else:
                print(f"\n  [POLARITY DISCOVERY] No polarity dimensions found yet (continue training...)")

        # Print progress
        print(f"\n  Epoch {epoch}/{args.epochs}:")
        print(f"    Train - Distance: {train_avg['distance']:.4f}, Polarity: {train_avg['polarity']:.4f}, Sparsity: {train_avg['sparsity']:.4f}, Total: {train_avg['total']:.4f}")
        print(f"    Val   - Distance: {val_avg['distance']:.4f}, Polarity: {val_avg['polarity']:.4f}, Sparsity: {val_avg['sparsity']:.4f}, Total: {val_avg['total']:.4f}")

        # Compute sparsity percentage
        embeddings_np = model.embeddings.detach().cpu().numpy()
        sparsity_pct = np.mean(np.abs(embeddings_np) < 0.01) * 100
        print(f"    Sparsity: {sparsity_pct:.1f}% (target: 40-70%)")

        # Save best model
        if val_avg['total'] < best_val_loss:
            best_val_loss = val_avg['total']
            patience_counter = 0

            checkpoint_path = output_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'polarity_dims': polarity_dims,
            }, checkpoint_path)

            # Save embeddings
            embeddings_path = output_dir / "embeddings_best.npy"
            np.save(embeddings_path, embeddings_np)

            print(f"    ✓ Saved best model (loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping (patience={patience})")
                break

        print()

    # Final dimension discovery
    print("="*70)
    print("FINAL POLARITY DIMENSION DISCOVERY")
    print("="*70)
    embeddings_np = model.embeddings.detach().cpu().numpy()
    polarity_dims, dim_scores = discover_polarity_dimensions(
        embeddings_np, antonym_pairs, min_consistency=0.15, top_k=20
    )

    if len(polarity_dims) > 0:
        print(f"\n✓ Discovered {len(polarity_dims)} polarity dimensions:")
        for dim_idx, score, consistency, disc_power in dim_scores:
            print(f"  Dim {dim_idx}: score={score:.4f}, consistency={consistency:.2%}, power={disc_power:.4f}")

        # Save polarity dimension mapping
        polarity_map_path = output_dir / "polarity_dimensions.json"
        with open(polarity_map_path, 'w') as f:
            json.dump({
                'polarity_dims': polarity_dims,
                'dim_scores': [(int(d), float(s), float(c), float(p)) for d, s, c, p in dim_scores]
            }, f, indent=2)
        print(f"\nSaved polarity dimension mapping: {polarity_map_path}")
    else:
        print("\n✗ No polarity dimensions discovered")
        print("  Try: Increase polarity weight, increase epochs, or lower min_consistency")

    print()
    print("="*70)
    print("PHASE 1 BOOTSTRAP COMPLETE")
    print("="*70)
    print(f"Best model saved to: {output_dir / 'best_model.pt'}")
    print(f"Embeddings saved to: {output_dir / 'embeddings_best.npy'}")
    print()
    print("Next step: Phase 2 - Wikipedia refinement")
    print("  python scripts/train_embeddings.py \\")
    print(f"    --resume-from-checkpoint {output_dir / 'best_model.pt'} \\")
    print("    --graph-dir data/wikipedia_100k_graph")


if __name__ == "__main__":
    main()
