"""
NAOMI-II Embedding Training Script

Trains semantic embeddings using distance-based constraints from the knowledge graph.

Architecture:
- Simple word embedding model with L2 normalization
- Distance loss: Enforces semantic relationships (synonyms close, antonyms far)
- Anchor dimensions: First 51 dims are predefined semantic/grammatical features
- Dynamic dimensions: Remaining dims learned from data

Usage:
    python scripts/train_embeddings.py [--epochs 50] [--lr 0.001]
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

        for source_id, relation_id, target_id, confidence in edges:
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


def create_relation_distance_map():
    """Map relation types to target distance ranges."""
    return {
        0: 0.1,   # synonym
        1: 0.3,   # hypernym
        2: 0.3,   # hyponym
        3: 0.9,   # antonym
        4: 0.4,   # parse relations
        5: 0.4,
        6: 0.3,
        7: 0.3,
        8: 0.4,
        9: 0.5,
        10: 0.5,
    }


class DistanceLoss(nn.Module):
    """Distance-based loss function."""

    def forward(self, embeddings: torch.Tensor, batch: Dict) -> torch.Tensor:
        source_embeds = embeddings[batch['source_id']]
        target_embeds = embeddings[batch['target_id']]

        diff = source_embeds - target_embeds
        # Compute L2 distance without normalization
        actual_dists = torch.norm(diff, dim=1)

        errors = (actual_dists - batch['target_distance']) ** 2
        weighted_errors = errors * batch['confidence']

        return weighted_errors.mean()


class SparsityLoss(nn.Module):
    """
    L1 sparsity regularization for PyTorch.

    Encourages embeddings to be sparse (most values near zero).
    Words should only activate dimensions relevant to their meaning.

    Target: 40-70% sparsity.
    """

    def __init__(self, l1_weight: float = 0.01):
        """
        Initialize sparsity loss.

        Args:
            l1_weight: L1 regularization strength (0.001-0.1 typical)
        """
        super().__init__()
        self.l1_weight = l1_weight

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute L1 sparsity loss.

        Args:
            embeddings: Word embedding matrix (vocab_size x embedding_dim)

        Returns:
            Sparsity loss (penalizes non-zero values)
        """
        return self.l1_weight * torch.mean(torch.abs(embeddings))


class SelectivePolarityLoss(nn.Module):
    """
    Selective polarity constraint for antonyms (PyTorch version).

    Different antonym types oppose on different specific dimensions:
    - good/bad oppose on dimension 0 (morality)
    - hot/cold oppose on dimension 1 (temperature)
    - big/small oppose on dimension 2 (size)

    Goal: Each pair opposes on 1-10 dims (not all 128!).
    """

    def __init__(self, antonym_dimension_map: Dict[Tuple[str, str], List[int]],
                 word_to_id: Dict[str, int],
                 polarity_weight: float = 1.0,
                 similarity_weight: float = 0.5):
        """
        Initialize selective polarity loss.

        Args:
            antonym_dimension_map: Maps (word1, word2) -> list of dimensions
            word_to_id: Maps words to embedding indices
            polarity_weight: Weight for opposition on assigned dimensions
            similarity_weight: Weight for similarity on other dimensions
        """
        super().__init__()
        self.polarity_weight = polarity_weight
        self.similarity_weight = similarity_weight
        self.word_to_id = word_to_id

        # Pre-process antonym pairs into tensors for efficient computation
        self.pair_ids = []
        self.assigned_dims_list = []

        for (word1, word2), dims in antonym_dimension_map.items():
            id1 = word_to_id.get(word1.lower())
            id2 = word_to_id.get(word2.lower())

            if id1 is not None and id2 is not None:
                self.pair_ids.append((id1, id2))
                self.assigned_dims_list.append(dims)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute selective polarity loss.

        Args:
            embeddings: Word embedding matrix (vocab_size x embedding_dim)

        Returns:
            Total selective polarity loss
        """
        if len(self.pair_ids) == 0:
            return torch.tensor(0.0, device=embeddings.device)

        total_loss = torch.tensor(0.0, device=embeddings.device)
        embedding_dim = embeddings.shape[1]

        for (id1, id2), assigned_dims in zip(self.pair_ids, self.assigned_dims_list):
            vec1 = embeddings[id1]
            vec2 = embeddings[id2]

            # 1. Polarity constraint on ASSIGNED dimensions
            polarity_loss = torch.tensor(0.0, device=embeddings.device)
            for dim in assigned_dims:
                val1 = vec1[dim]
                val2 = vec2[dim]

                sign_product = torch.sign(val1) * torch.sign(val2)

                # Same sign - penalty
                same_sign_penalty = torch.where(
                    sign_product > 0,
                    torch.abs(val1) + torch.abs(val2),
                    torch.tensor(0.0, device=embeddings.device)
                )

                # Opposite signs - reward
                opposite_sign_reward = torch.where(
                    sign_product < 0,
                    -(torch.abs(val1) + torch.abs(val2)) * 0.5,
                    torch.tensor(0.0, device=embeddings.device)
                )

                # One is zero - small penalty
                zero_penalty = torch.where(
                    sign_product == 0,
                    torch.tensor(0.1, device=embeddings.device),
                    torch.tensor(0.0, device=embeddings.device)
                )

                polarity_loss += same_sign_penalty + opposite_sign_reward + zero_penalty

            # 2. Similarity constraint on NON-ASSIGNED dimensions
            all_dims = torch.arange(embedding_dim, device=embeddings.device)
            assigned_dims_tensor = torch.tensor(assigned_dims, device=embeddings.device)

            # Create mask for non-assigned dimensions
            mask = torch.ones(embedding_dim, dtype=torch.bool, device=embeddings.device)
            mask[assigned_dims_tensor] = False

            if mask.any():
                diff = vec1[mask] - vec2[mask]
                similarity_loss = torch.mean(diff ** 2)
            else:
                similarity_loss = torch.tensor(0.0, device=embeddings.device)

            total_loss += self.polarity_weight * polarity_loss + self.similarity_weight * similarity_loss

        return total_loss / len(self.pair_ids)

    @staticmethod
    def from_config_files(antonym_types_path: str,
                         dimension_assignments_path: str,
                         word_to_id: Dict[str, int]) -> 'SelectivePolarityLoss':
        """
        Create SelectivePolarityLoss from config files.

        Args:
            antonym_types_path: Path to antonym_types.json
            dimension_assignments_path: Path to dimension_assignments.json
            word_to_id: Word to ID mapping

        Returns:
            Configured SelectivePolarityLoss instance
        """
        # Load antonym types
        with open(antonym_types_path, 'r') as f:
            antonym_types = json.load(f)

        # Load dimension assignments
        with open(dimension_assignments_path, 'r') as f:
            dimension_assignments = json.load(f)
            # Remove comments
            dimension_assignments = {k: v for k, v in dimension_assignments.items()
                                    if not k.startswith('_')}

        # Build antonym-dimension map
        antonym_dimension_map = {}
        for atype, pairs in antonym_types.items():
            if atype in dimension_assignments:
                dims = dimension_assignments[atype]
                for word1, word2 in pairs:
                    antonym_dimension_map[(word1, word2)] = dims

        return SelectivePolarityLoss(antonym_dimension_map, word_to_id)


class DimensionalConsistencyLoss(nn.Module):
    """
    Enforces dimensional consistency across all words (PyTorch version).

    Each dimension has consistent meaning for ALL words:
    - Dimension 0 = morality for EVERY word
    - Dimension 1 = gender for EVERY word
    - etc.

    Creates interpretable semantic space.
    """

    def __init__(self, semantic_clusters: Dict[int, Dict[str, List[str]]],
                 word_to_id: Dict[str, int],
                 consistency_weight: float = 0.5,
                 sparsity_weight: float = 0.1):
        """
        Initialize dimensional consistency loss.

        Args:
            semantic_clusters: Maps dimension index to cluster definition
            word_to_id: Maps words to embedding indices
            consistency_weight: Weight for clustering on target dimension
            sparsity_weight: Weight for being zero on other dimensions
        """
        super().__init__()
        self.consistency_weight = consistency_weight
        self.sparsity_weight = sparsity_weight
        self.word_to_id = word_to_id

        # Pre-process clusters into efficient structures
        self.positive_words = {}  # dim -> list of word IDs
        self.negative_words = {}
        self.neutral_words = {}

        for dim_idx, cluster_def in semantic_clusters.items():
            # Positive pole
            pos_ids = []
            for word in cluster_def.get('positive', []):
                word_id = word_to_id.get(word.lower())
                if word_id is not None:
                    pos_ids.append(word_id)
            if pos_ids:
                self.positive_words[dim_idx] = pos_ids

            # Negative pole
            neg_ids = []
            for word in cluster_def.get('negative', []):
                word_id = word_to_id.get(word.lower())
                if word_id is not None:
                    neg_ids.append(word_id)
            if neg_ids:
                self.negative_words[dim_idx] = neg_ids

            # Neutral
            neutral_ids = []
            for word in cluster_def.get('neutral', []):
                word_id = word_to_id.get(word.lower())
                if word_id is not None:
                    neutral_ids.append(word_id)
            if neutral_ids:
                self.neutral_words[dim_idx] = neutral_ids

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute dimensional consistency loss.

        Args:
            embeddings: Word embedding matrix (vocab_size x embedding_dim)

        Returns:
            Total consistency loss
        """
        total_loss = torch.tensor(0.0, device=embeddings.device)
        num_constraints = 0
        embedding_dim = embeddings.shape[1]

        for dim_idx in self.positive_words.keys() | self.negative_words.keys() | self.neutral_words.keys():
            # 1. Positive pole: should have positive values on this dim
            if dim_idx in self.positive_words:
                for word_id in self.positive_words[dim_idx]:
                    vec = embeddings[word_id]
                    target_val = vec[dim_idx]

                    # Penalty if not positive, small reward if positive
                    sign_loss = torch.where(
                        target_val <= 0,
                        torch.abs(target_val) + 0.1,
                        -target_val * 0.1
                    )
                    total_loss += sign_loss

                    # Sparsity on other dimensions
                    other_dims = [i for i in range(embedding_dim) if i != dim_idx]
                    if other_dims:
                        total_loss += self.sparsity_weight * torch.mean(torch.abs(vec[other_dims]))

                    num_constraints += 1

            # 2. Negative pole: should have negative values on this dim
            if dim_idx in self.negative_words:
                for word_id in self.negative_words[dim_idx]:
                    vec = embeddings[word_id]
                    target_val = vec[dim_idx]

                    # Penalty if not negative, small reward if negative
                    sign_loss = torch.where(
                        target_val >= 0,
                        torch.abs(target_val) + 0.1,
                        -torch.abs(target_val) * 0.1
                    )
                    total_loss += sign_loss

                    # Sparsity on other dimensions
                    other_dims = [i for i in range(embedding_dim) if i != dim_idx]
                    if other_dims:
                        total_loss += self.sparsity_weight * torch.mean(torch.abs(vec[other_dims]))

                    num_constraints += 1

            # 3. Neutral: should be near zero on this dim
            if dim_idx in self.neutral_words:
                for word_id in self.neutral_words[dim_idx]:
                    vec = embeddings[word_id]
                    target_val = vec[dim_idx]

                    # Strong penalty for being non-zero
                    total_loss += torch.abs(target_val) * 2.0
                    num_constraints += 1

        if num_constraints == 0:
            return torch.tensor(0.0, device=embeddings.device)

        return self.consistency_weight * (total_loss / num_constraints)

    @staticmethod
    def from_config_file(semantic_clusters_path: str,
                        word_to_id: Dict[str, int]) -> 'DimensionalConsistencyLoss':
        """
        Create DimensionalConsistencyLoss from config file.

        Args:
            semantic_clusters_path: Path to semantic_clusters.json
            word_to_id: Word to ID mapping

        Returns:
            Configured DimensionalConsistencyLoss instance
        """
        with open(semantic_clusters_path, 'r', encoding='utf-8') as f:
            clusters_str = json.load(f)

        # Convert string keys back to integers
        clusters = {int(k): v for k, v in clusters_str.items()}

        return DimensionalConsistencyLoss(clusters, word_to_id)


class SimpleEmbeddingModel(nn.Module):
    """Simple embedding model without L2 normalization for sparsity."""

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        # Initialize with small random values (no normalization)
        self.embeddings = nn.Parameter(torch.randn(vocab_size, embedding_dim) * 0.01)

    def forward(self):
        # Return embeddings directly without normalization
        # This allows sparsity to emerge naturally
        return self.embeddings


def train_epoch(model, dataloader, distance_criterion, sparsity_criterion, polarity_criterion, consistency_criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_distance_loss = 0.0
    total_sparsity_loss = 0.0
    total_polarity_loss = 0.0
    total_consistency_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        embeddings = model()

        # Compute all losses
        distance_loss = distance_criterion(embeddings, batch)
        sparsity_loss = sparsity_criterion(embeddings)

        # Combined loss
        loss = distance_loss + sparsity_loss

        # Add polarity loss if available
        if polarity_criterion is not None:
            polarity_loss = polarity_criterion(embeddings)
            loss = loss + polarity_loss
            total_polarity_loss += polarity_loss.item()
        else:
            polarity_loss = torch.tensor(0.0)

        # Add consistency loss if available
        if consistency_criterion is not None:
            consistency_loss = consistency_criterion(embeddings)
            loss = loss + consistency_loss
            total_consistency_loss += consistency_loss.item()
        else:
            consistency_loss = torch.tensor(0.0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_distance_loss += distance_loss.item()
        total_sparsity_loss += sparsity_loss.item()

        postfix = {
            'loss': f'{loss.item():.4f}',
            'dist': f'{distance_loss.item():.4f}',
            'spar': f'{sparsity_loss.item():.4f}'
        }
        if polarity_criterion is not None:
            postfix['polar'] = f'{polarity_loss.item():.4f}'
        if consistency_criterion is not None:
            postfix['cons'] = f'{consistency_loss.item():.4f}'
        pbar.set_postfix(postfix)

    result = {
        'total': total_loss / len(dataloader),
        'distance': total_distance_loss / len(dataloader),
        'sparsity': total_sparsity_loss / len(dataloader)
    }
    if polarity_criterion is not None:
        result['polarity'] = total_polarity_loss / len(dataloader)
    if consistency_criterion is not None:
        result['consistency'] = total_consistency_loss / len(dataloader)
    return result


def validate(model, dataloader, distance_criterion, sparsity_criterion, polarity_criterion, consistency_criterion, device):
    """Validate on dataset."""
    model.eval()
    total_loss = 0.0
    total_distance_loss = 0.0
    total_sparsity_loss = 0.0
    total_polarity_loss = 0.0
    total_consistency_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            embeddings = model()

            distance_loss = distance_criterion(embeddings, batch)
            sparsity_loss = sparsity_criterion(embeddings)
            loss = distance_loss + sparsity_loss

            # Add polarity loss if available
            if polarity_criterion is not None:
                polarity_loss = polarity_criterion(embeddings)
                loss = loss + polarity_loss
                total_polarity_loss += polarity_loss.item()

            # Add consistency loss if available
            if consistency_criterion is not None:
                consistency_loss = consistency_criterion(embeddings)
                loss = loss + consistency_loss
                total_consistency_loss += consistency_loss.item()

            total_loss += loss.item()
            total_distance_loss += distance_loss.item()
            total_sparsity_loss += sparsity_loss.item()

    result = {
        'total': total_loss / len(dataloader),
        'distance': total_distance_loss / len(dataloader),
        'sparsity': total_sparsity_loss / len(dataloader)
    }
    if polarity_criterion is not None:
        result['polarity'] = total_polarity_loss / len(dataloader)
    if consistency_criterion is not None:
        result['consistency'] = total_consistency_loss / len(dataloader)
    return result


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)


def main():
    parser = argparse.ArgumentParser(description='Train NAOMI-II semantic embeddings')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--embedding-dim', type=int, default=128, help='Embedding dimensionality')
    parser.add_argument('--unsupervised', action='store_true',
                       help='Skip manual dimension assignments (use distance + sparsity only)')

    args = parser.parse_args()

    print("=" * 70)
    print("NAOMI-II EMBEDDING TRAINING")
    print("=" * 70)
    print()

    # Configuration
    TRAINING_DATA_DIR = Path("data/training")
    OUTPUT_DIR = Path("checkpoints")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize device
    print("[1/7] Initializing device...")
    device_manager = DeviceManager(prefer_npu=False, verbose=True)
    device = device_manager.device
    print()

    # Load training data
    print("[2/7] Loading training data...")
    with open(TRAINING_DATA_DIR / "training_edges.pkl", 'rb') as f:
        edges = pickle.load(f)
    with open(TRAINING_DATA_DIR / "vocabulary.json", 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)

    vocab_size = vocab_data['vocab_size']
    word_to_id = vocab_data['word_to_id']
    id_to_word = vocab_data['id_to_word']

    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Training edges: {len(edges)}")
    print()

    # Create dataset
    print("[3/7] Creating dataset...")
    relation_distances = create_relation_distance_map()
    dataset = DistanceConstraintDataset(edges, vocab_size, relation_distances)
    print(f"  Dataset size: {len(dataset)} samples")

    # Split into train/validation
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    print(f"  Train: {len(train_dataset)}, Validation: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    print()

    # Initialize model
    print("[4/7] Initializing embedding model...")
    anchors = AnchorDimensions()
    num_anchors = anchors.num_anchors()
    print(f"  Embedding dim: {args.embedding_dim} ({num_anchors} anchor + {args.embedding_dim - num_anchors} learned)")

    model = SimpleEmbeddingModel(vocab_size, args.embedding_dim).to(device)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Initialize optimizer and loss
    print("[5/7] Initializing optimizer and loss functions...")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    distance_criterion = DistanceLoss()
    sparsity_criterion = SparsityLoss(l1_weight=0.01)

    # Load selective polarity loss from config files (unless --unsupervised)
    config_dir = Path("config")
    if args.unsupervised:
        polarity_criterion = None
        consistency_criterion = None
        print(f"  [UNSUPERVISED MODE] Skipping manual dimension assignments")
        print(f"  Training with distance + sparsity constraints only")
    else:
        if (config_dir / "antonym_types.json").exists() and (config_dir / "dimension_assignments.json").exists():
            polarity_criterion = SelectivePolarityLoss.from_config_files(
                str(config_dir / "antonym_types.json"),
                str(config_dir / "dimension_assignments.json"),
                word_to_id
            )
            print(f"  Loaded {len(polarity_criterion.pair_ids)} antonym pairs for selective polarity")
        else:
            polarity_criterion = None
            print(f"  No polarity config found - skipping polarity loss")

        # Load dimensional consistency loss from config file
        if (config_dir / "semantic_clusters.json").exists():
            consistency_criterion = DimensionalConsistencyLoss.from_config_file(
                str(config_dir / "semantic_clusters.json"),
                word_to_id
            )
            num_dims = len(consistency_criterion.positive_words) + len(consistency_criterion.negative_words) + len(consistency_criterion.neutral_words)
            print(f"  Loaded semantic clusters for {num_dims} dimensions")
        else:
            consistency_criterion = None
            print(f"  No semantic clusters found - skipping consistency loss")
            print(f"  (Run 'python scripts/bootstrap_semantic_clusters.py' to generate clusters)")

    print(f"  Optimizer: Adam (lr={args.lr})")
    loss_components = "Distance + Sparsity"
    if polarity_criterion:
        loss_components += " + Selective Polarity"
    if consistency_criterion:
        loss_components += " + Dimensional Consistency"
    print(f"  Loss components: {loss_components}")
    print()

    # Training loop
    print("[6/7] Training...")
    print()
    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_losses = train_epoch(model, train_loader, distance_criterion, sparsity_criterion, polarity_criterion, consistency_criterion, optimizer, device, epoch)
        val_losses = validate(model, val_loader, distance_criterion, sparsity_criterion, polarity_criterion, consistency_criterion, device)

        print(f"Epoch {epoch}/{args.epochs}")
        loss_str = f"  Train - Total: {train_losses['total']:.4f}, Distance: {train_losses['distance']:.4f}, Sparsity: {train_losses['sparsity']:.4f}"
        if 'polarity' in train_losses:
            loss_str += f", Polarity: {train_losses['polarity']:.4f}"
        if 'consistency' in train_losses:
            loss_str += f", Consistency: {train_losses['consistency']:.4f}"
        print(loss_str)

        loss_str = f"  Val   - Total: {val_losses['total']:.4f}, Distance: {val_losses['distance']:.4f}, Sparsity: {val_losses['sparsity']:.4f}"
        if 'polarity' in val_losses:
            loss_str += f", Polarity: {val_losses['polarity']:.4f}"
        if 'consistency' in val_losses:
            loss_str += f", Consistency: {val_losses['consistency']:.4f}"
        print(loss_str)

        # Compute and display sparsity percentage
        with torch.no_grad():
            embeddings_np = model().cpu().numpy()
            sparsity_pct = 100.0 * np.mean(np.abs(embeddings_np) < 0.01)
            print(f"  Sparsity: {sparsity_pct:.1f}% (target: 40-70%)")

        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, epoch, val_losses['total'],
                          OUTPUT_DIR / f"checkpoint_epoch_{epoch}.pt")
            print(f"  Saved checkpoint")

        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            save_checkpoint(model, optimizer, epoch, val_losses['total'], OUTPUT_DIR / "best_model.pt")
            print(f"  New best model! (val_loss: {val_losses['total']:.4f})")

        print()

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed/60:.1f} minutes")
    print()

    # Save final model
    print("[7/7] Saving final model...")
    save_checkpoint(model, optimizer, args.epochs, val_losses['total'], OUTPUT_DIR / "final_model.pt")

    final_embeddings = model().detach().cpu().numpy()
    np.save(OUTPUT_DIR / "embeddings.npy", final_embeddings)
    print(f"  Saved embeddings to {OUTPUT_DIR / 'embeddings.npy'}")

    with open(OUTPUT_DIR / "vocabulary.json", 'w', encoding='utf-8') as f:
        json.dump({
            'word_to_id': word_to_id,
            'id_to_word': id_to_word,
            'vocab_size': vocab_size,
            'embedding_dim': args.embedding_dim
        }, f, indent=2, ensure_ascii=False)
    print(f"  Saved vocabulary to {OUTPUT_DIR / 'vocabulary.json'}")
    print()

    print("=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final embeddings: {OUTPUT_DIR / 'embeddings.npy'}")
    print()


if __name__ == "__main__":
    main()
