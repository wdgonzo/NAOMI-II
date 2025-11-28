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
from typing import Dict, List, Tuple, Optional
import time

# Mixed precision training support
try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

from src.embeddings.device import DeviceManager
from src.embeddings.anchors import AnchorDimensions
from src.embeddings.dynamic_dimensions import adaptive_dimension_management


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
        # Numeric IDs (original parse-based relations)
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
        # String IDs (WordNet-based relations)
        'hypernym': 0.3,
        'hyponym': 0.3,
        'similar': 0.2,
        'meronym': 0.5,
        'holonym': 0.5,
        'antonym': 1.0,
        # WordNet tagged relations (cycled training)
        'wordnet:hypernym': 0.3,
        'wordnet:hyponym': 0.3,
        'wordnet:similar': 0.2,
        'wordnet:meronym': 0.5,
        'wordnet:holonym': 0.5,
        'wordnet:antonym': 1.0,
        # Wikipedia relations (cycled training)
        'wikipedia:co-occur': 0.4,        # Co-occurrence relations
        'wikipedia:syntax_co-occur': 0.4,  # Syntactic co-occurrence
        # Generic fallback for unknown Wikipedia relations
        'co-occur': 0.4,
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


def train_epoch(model, dataloader, distance_criterion, sparsity_criterion, polarity_criterion, consistency_criterion, optimizer, device, epoch, scaler=None, gradient_accumulation_steps=1):
    """Train for one epoch with optional mixed precision and gradient accumulation."""
    model.train()
    total_loss = 0.0
    total_distance_loss = 0.0
    total_sparsity_loss = 0.0
    total_polarity_loss = 0.0
    total_consistency_loss = 0.0

    # Zero gradients at start
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Mixed precision context
        use_amp = scaler is not None and torch.cuda.is_available()

        if use_amp:
            with autocast():
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
                else:
                    polarity_loss = torch.tensor(0.0)

                # Add consistency loss if available
                if consistency_criterion is not None:
                    consistency_loss = consistency_criterion(embeddings)
                    loss = loss + consistency_loss
                else:
                    consistency_loss = torch.tensor(0.0)

                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
        else:
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
            else:
                polarity_loss = torch.tensor(0.0)

            # Add consistency loss if available
            if consistency_criterion is not None:
                consistency_loss = consistency_criterion(embeddings)
                loss = loss + consistency_loss
            else:
                consistency_loss = torch.tensor(0.0)

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps

        # Backward pass with or without mixed precision
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step only at accumulation boundaries
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

        # Track losses (unscaled)
        total_loss += loss.item() * gradient_accumulation_steps
        total_distance_loss += distance_loss.item()
        total_sparsity_loss += sparsity_loss.item()
        if polarity_criterion is not None:
            total_polarity_loss += polarity_loss.item()
        if consistency_criterion is not None:
            total_consistency_loss += consistency_loss.item()

        postfix = {
            'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
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
    parser.add_argument('--embedding-dim', type=int, default=128, help='Starting embedding dimensionality')
    parser.add_argument('--training-data', type=str, default='data/training',
                       help='Directory containing training data')
    parser.add_argument('--unsupervised', action='store_true',
                       help='Skip manual dimension assignments (use distance + sparsity only)')
    parser.add_argument('--dynamic-dims', action='store_true',
                       help='Enable dynamic dimension expansion during training')
    parser.add_argument('--max-dims', type=int, default=512,
                       help='Maximum dimensions when using --dynamic-dims')
    parser.add_argument('--expand-interval', type=int, default=2,
                       help='Check for dimension expansion every N epochs')
    parser.add_argument('--expand-by', type=int, default=64,
                       help='Number of dimensions to add each expansion')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping: epochs to wait for improvement')
    parser.add_argument('--min-delta', type=float, default=0.0001,
                       help='Early stopping: minimum improvement to count')

    # Cycled training (WordNet + Wikipedia alternating)
    parser.add_argument('--cycled-training', action='store_true',
                       help='Enable cycled training (alternate WordNet/Wikipedia each epoch)')
    parser.add_argument('--training-data-wordnet', type=str, default=None,
                       help='Directory with WordNet training data (for cycled training)')
    parser.add_argument('--training-data-wikipedia', type=str, default=None,
                       help='Directory with Wikipedia training data (for cycled training)')
    parser.add_argument('--vocabulary', type=str, default=None,
                       help='Unified vocabulary file (for cycled training)')

    # A100 optimizations
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Enable mixed precision training (float16) for 2x speedup')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                       help='Gradient accumulation steps (simulate larger batches)')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='Number of data loading workers (use 16 for A100)')
    parser.add_argument('--prefetch-factor', type=int, default=2,
                       help='Number of batches to prefetch per worker')
    parser.add_argument('--lr-scheduler', type=str, default=None,
                       choices=[None, 'cosine', 'step', 'exponential'],
                       help='Learning rate scheduler type')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                       help='Number of warmup epochs for LR scheduler')

    args = parser.parse_args()

    print("=" * 70)
    print("NAOMI-II EMBEDDING TRAINING")
    print("=" * 70)
    print()

    # Configuration
    OUTPUT_DIR = Path("checkpoints")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize device
    print("[1/7] Initializing device...")
    device_manager = DeviceManager(prefer_npu=False, verbose=True)
    device = device_manager.device
    print()

    # Load training data
    print("[2/7] Loading training data...")

    # Check for cycled training mode
    if args.cycled_training:
        if not args.training_data_wordnet or not args.training_data_wikipedia:
            print("ERROR: --cycled-training requires --training-data-wordnet and --training-data-wikipedia")
            sys.exit(1)

        print("  [CYCLED TRAINING MODE]")
        print("  Loading WordNet dataset...")
        wordnet_edges_file = Path(args.training_data_wordnet)
        with open(wordnet_edges_file, 'rb') as f:
            wordnet_edges = pickle.load(f)
        print(f"    WordNet edges: {len(wordnet_edges):,}")

        print("  Loading Wikipedia dataset...")
        wikipedia_edges_file = Path(args.training_data_wikipedia)
        with open(wikipedia_edges_file, 'rb') as f:
            wikipedia_edges = pickle.load(f)
        print(f"    Wikipedia edges: {len(wikipedia_edges):,}")

        # Load unified vocabulary
        if args.vocabulary:
            vocab_file = Path(args.vocabulary)
        else:
            vocab_file = Path(args.training_data_wordnet).parent / "vocabulary.json"

        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)

        # Store both edge sets for alternating
        cycled_edges = {
            'wordnet': wordnet_edges,
            'wikipedia': wikipedia_edges
        }
        edges = wordnet_edges + wikipedia_edges  # Combined for initial dataset creation

    else:
        # Single dataset mode (original behavior)
        TRAINING_DATA_DIR = Path(args.training_data)
        with open(TRAINING_DATA_DIR / "training_edges.pkl", 'rb') as f:
            edges = pickle.load(f)
        with open(TRAINING_DATA_DIR / "vocabulary.json", 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)

        cycled_edges = None

    vocab_size = vocab_data['vocab_size']
    word_to_id = vocab_data['word_to_id']
    id_to_word = vocab_data['id_to_word']

    print(f"  Vocabulary size: {vocab_size:,}")
    if not args.cycled_training:
        print(f"  Training edges: {len(edges):,}")
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

    # Create dataloaders (will be recreated if batch size changes)
    current_batch_size = args.batch_size
    initial_batch_size = args.batch_size
    initial_dims = args.embedding_dim

    # DataLoader kwargs for A100 optimization
    dataloader_kwargs = {
        'num_workers': args.num_workers,
        'pin_memory': True if torch.cuda.is_available() else False,
    }
    if args.num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = args.prefetch_factor

    train_loader = DataLoader(
        train_dataset, batch_size=current_batch_size, shuffle=True, **dataloader_kwargs
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, **dataloader_kwargs
    )

    # Create separate loaders for cycled training if enabled
    if args.cycled_training:
        wordnet_dataset = DistanceConstraintDataset(cycled_edges['wordnet'], vocab_size, relation_distances)
        wikipedia_dataset = DistanceConstraintDataset(cycled_edges['wikipedia'], vocab_size, relation_distances)

        # Split each dataset
        wordnet_val_size = int(len(wordnet_dataset) * 0.1)
        wordnet_train_size = len(wordnet_dataset) - wordnet_val_size
        wordnet_train, wordnet_val = torch.utils.data.random_split(
            wordnet_dataset, [wordnet_train_size, wordnet_val_size]
        )

        wikipedia_val_size = int(len(wikipedia_dataset) * 0.1)
        wikipedia_train_size = len(wikipedia_dataset) - wikipedia_val_size
        wikipedia_train, wikipedia_val = torch.utils.data.random_split(
            wikipedia_dataset, [wikipedia_train_size, wikipedia_val_size]
        )

        wordnet_train_loader = DataLoader(wordnet_train, batch_size=current_batch_size, shuffle=True, **dataloader_kwargs)
        wordnet_val_loader = DataLoader(wordnet_val, batch_size=args.batch_size, shuffle=False, **dataloader_kwargs)
        wikipedia_train_loader = DataLoader(wikipedia_train, batch_size=current_batch_size, shuffle=True, **dataloader_kwargs)
        wikipedia_val_loader = DataLoader(wikipedia_val, batch_size=args.batch_size, shuffle=False, **dataloader_kwargs)

        cycled_loaders = {
            'wordnet': (wordnet_train_loader, wordnet_val_loader),
            'wikipedia': (wikipedia_train_loader, wikipedia_val_loader)
        }

        print(f"  Cycled training loaders created:")
        print(f"    WordNet - Train: {len(wordnet_train):,}, Val: {len(wordnet_val):,}")
        print(f"    Wikipedia - Train: {len(wikipedia_train):,}, Val: {len(wikipedia_val):,}")
    else:
        cycled_loaders = None

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

    # Initialize learning rate scheduler if requested
    scheduler = None
    if args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs - args.warmup_epochs,
            eta_min=args.lr * 0.01
        )
        print(f"  LR Scheduler: Cosine annealing (T_max={args.epochs - args.warmup_epochs})")
    elif args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        print(f"  LR Scheduler: Step (step_size=10, gamma=0.5)")
    elif args.lr_scheduler == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        print(f"  LR Scheduler: Exponential (gamma=0.95)")

    # Mixed precision scaler
    scaler = None
    if args.mixed_precision:
        if not AMP_AVAILABLE:
            print("  WARNING: Mixed precision requested but not available (requires PyTorch 1.6+)")
            print("  Falling back to FP32 training")
        elif not torch.cuda.is_available():
            print("  WARNING: Mixed precision requires CUDA, falling back to FP32")
        else:
            scaler = GradScaler()
            print(f"  Mixed precision enabled (FP16/FP32)")

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
    if args.dynamic_dims:
        print(f"  Dynamic dimensions enabled: {args.embedding_dim} -> max {args.max_dims}")
        print(f"  Expansion check interval: every {args.expand_interval} epochs")
        print(f"  Expansion amount: {args.expand_by} dims per expansion")
    print(f"  Early stopping enabled: patience={args.patience}, min_delta={args.min_delta}")
    if args.cycled_training:
        print(f"  Cycled training enabled: alternating WordNet/Wikipedia each epoch")
    if args.mixed_precision and scaler is not None:
        print(f"  Mixed precision training: FP16/FP32")
    if args.gradient_accumulation_steps > 1:
        print(f"  Gradient accumulation: {args.gradient_accumulation_steps} steps")
    if scheduler is not None:
        print(f"  Learning rate schedule: {args.lr_scheduler}")
    print()
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # Determine which dataset to use for cycled training
        if args.cycled_training:
            if epoch % 2 == 1:
                # Odd epochs: WordNet
                current_source = 'wordnet'
                train_loader_cycle, val_loader_cycle = cycled_loaders['wordnet']
            else:
                # Even epochs: Wikipedia
                current_source = 'wikipedia'
                train_loader_cycle, val_loader_cycle = cycled_loaders['wikipedia']
        else:
            current_source = None
            train_loader_cycle = train_loader
            val_loader_cycle = val_loader

        # Apply warmup to learning rate
        if scheduler is not None and epoch <= args.warmup_epochs:
            warmup_factor = epoch / args.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * warmup_factor

        train_losses = train_epoch(model, train_loader_cycle, distance_criterion, sparsity_criterion, polarity_criterion, consistency_criterion, optimizer, device, epoch, scaler=scaler, gradient_accumulation_steps=args.gradient_accumulation_steps)
        val_losses = validate(model, val_loader_cycle, distance_criterion, sparsity_criterion, polarity_criterion, consistency_criterion, device)

        # Display epoch info
        epoch_info = f"Epoch {epoch}/{args.epochs}"
        if args.cycled_training:
            epoch_info += f" [{current_source.upper()}]"
        print(epoch_info)

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

        # Display current learning rate if using scheduler
        if scheduler is not None:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Learning rate: {current_lr:.6f}")

        # Dynamic dimension management
        if args.dynamic_dims:
            with torch.no_grad():
                embeddings_np_current = model().cpu().numpy()
                embeddings_np_current, model, did_expand = adaptive_dimension_management(
                    model=model,
                    embeddings=embeddings_np_current,
                    epoch=epoch,
                    max_dims=args.max_dims,
                    check_interval=args.expand_interval,
                    sparsity_threshold=0.3,
                    expand_by=args.expand_by
                )

                # If dimensions were expanded, adapt batch size and reinitialize optimizer
                if did_expand:
                    current_dims = model.embedding_dim
                    new_batch_size = int(initial_batch_size * (initial_dims / current_dims))
                    new_batch_size = max(new_batch_size, 32768)  # minimum 32K batch

                    if new_batch_size != current_batch_size:
                        current_batch_size = new_batch_size
                        train_loader = DataLoader(
                            train_dataset, batch_size=current_batch_size, shuffle=True, num_workers=0
                        )
                        val_loader = DataLoader(
                            val_dataset, batch_size=current_batch_size, shuffle=False, num_workers=0
                        )
                        print(f"  Adapted batch size to {current_batch_size} for {current_dims} dimensions")

                    optimizer = optim.Adam(model.parameters(), lr=args.lr)
                    print(f"  Reinitialized optimizer with new dimensions")

        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, epoch, val_losses['total'],
                          OUTPUT_DIR / f"checkpoint_epoch_{epoch}.pt")
            print(f"  Saved checkpoint")

        # Early stopping check
        if val_losses['total'] < best_val_loss - args.min_delta:
            best_val_loss = val_losses['total']
            epochs_without_improvement = 0
            save_checkpoint(model, optimizer, epoch, val_losses['total'], OUTPUT_DIR / "best_model.pt")
            print(f"  New best model! (val_loss: {val_losses['total']:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epochs")

        if epochs_without_improvement >= args.patience:
            print(f"\n[Early Stopping] No improvement for {args.patience} epochs")
            print(f"[Early Stopping] Best validation loss: {best_val_loss:.4f}")
            print(f"[Early Stopping] Stopping training at epoch {epoch}")
            break

        # Step learning rate scheduler (after warmup)
        if scheduler is not None and epoch > args.warmup_epochs:
            scheduler.step()

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

    # Get final embedding dimension (may have changed if dynamic dims enabled)
    final_embedding_dim = final_embeddings.shape[1]

    with open(OUTPUT_DIR / "vocabulary.json", 'w', encoding='utf-8') as f:
        json.dump({
            'word_to_id': word_to_id,
            'id_to_word': id_to_word,
            'vocab_size': vocab_size,
            'embedding_dim': final_embedding_dim,
            'starting_dim': args.embedding_dim,
            'dynamic_dims_enabled': args.dynamic_dims
        }, f, indent=2, ensure_ascii=False)
    print(f"  Saved vocabulary to {OUTPUT_DIR / 'vocabulary.json'}")
    print(f"  Final embedding dimensions: {final_embedding_dim} (started with {args.embedding_dim})")
    print()

    print("=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final embeddings: {OUTPUT_DIR / 'embeddings.npy'}")
    print()


if __name__ == "__main__":
    main()
