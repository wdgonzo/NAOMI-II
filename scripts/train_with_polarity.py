"""
Two-Stage Training with Polarity Constraints

Stage 1: Pre-train with distance-based fuzzy constraints
Stage 2: Discover polarity dimensions and fine-tune with polarity constraints

This implements the key insight: Force antonyms to have opposite signs
on specific dimensions to enable compositional semantics (NOT(good) ≈ bad).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import numpy as np
from typing import List, Tuple

from src.embeddings.model import EmbeddingModel
from src.embeddings.constraints import ConstraintLoss
from src.embeddings.polarity_discovery import PolarityDimensionDiscovery
from src.embeddings.logical_operators import test_logical_operators
from src.graph.knowledge_graph import KnowledgeGraph


def get_antonym_pairs() -> List[Tuple[str, str]]:
    """Get common antonym pairs for polarity constraint training."""
    return [
        ("good", "bad"),
        ("hot", "cold"),
        ("big", "small"),
        ("fast", "slow"),
        ("happy", "sad"),
        ("right", "wrong"),
        ("light", "dark"),
        ("hard", "soft"),
        ("long", "short"),
        ("high", "low"),
        ("new", "old"),
        ("clean", "dirty"),
        ("rich", "poor"),
        ("strong", "weak"),
        ("easy", "difficult"),
        ("young", "old"),
        ("bright", "dim"),
        ("thick", "thin"),
        ("wide", "narrow"),
        ("deep", "shallow")
    ]


def stage1_pretrain(model: EmbeddingModel, constraint_loss: ConstraintLoss,
                     epochs: int, lr: float, batch_size: int):
    """
    Stage 1: Pre-train with distance-based fuzzy constraints.

    This establishes basic semantic relationships without polarity constraints.

    Args:
        model: EmbeddingModel to train
        constraint_loss: ConstraintLoss with fuzzy constraints loaded
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size for constraint sampling
    """
    print("=" * 70)
    print("STAGE 1: PRE-TRAINING WITH DISTANCE CONSTRAINTS")
    print("=" * 70)
    print()

    stats = constraint_loss.get_statistics()
    print(f"Training configuration:")
    print(f"  Total constraints: {stats['total_constraints']}")
    print(f"  Constraints by type:")
    for ctype, count in stats['constraints_by_type'].items():
        print(f"    {ctype}: {count}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Batch size: {batch_size}")
    print()

    print("Beginning training...")
    print()

    for epoch in range(epochs):
        # Sample constraints for mini-batch
        sampled_constraints = constraint_loss.sample_constraints(batch_size)

        # Compute loss and gradients
        loss, loss_stats = constraint_loss.compute_total_loss(model)
        gradients = constraint_loss.compute_gradients(model)

        # Update embeddings
        model.embeddings -= lr * gradients

        # Log progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"loss={loss:.4f}, "
                  f"satisfied={loss_stats['satisfaction_rate']*100:.1f}%")

    print()
    print("Stage 1 complete!")
    print()

    # Final statistics
    final_loss, final_stats = constraint_loss.compute_total_loss(model)
    print("Final statistics:")
    print(f"  Total loss: {final_loss:.4f}")
    print(f"  Satisfaction rate: {final_stats['satisfaction_rate']*100:.1f}%")
    print(f"  Constraints satisfied: {final_stats['num_satisfied']}/{final_stats['num_constraints']}")
    print()


def stage2_polarity_training(model: EmbeddingModel, constraint_loss: ConstraintLoss,
                               epochs: int, lr: float, batch_size: int,
                               num_polarity_dims: int = 10):
    """
    Stage 2: Discover polarity dimensions and fine-tune with polarity constraints.

    This forces antonyms to have opposite signs on key dimensions.

    Args:
        model: Pre-trained EmbeddingModel
        constraint_loss: ConstraintLoss with fuzzy constraints
        epochs: Number of fine-tuning epochs
        lr: Learning rate (typically lower than stage 1)
        batch_size: Batch size
        num_polarity_dims: Number of polarity dimensions to discover
    """
    print("=" * 70)
    print("STAGE 2: POLARITY CONSTRAINT FINE-TUNING")
    print("=" * 70)
    print()

    # Step 1: Discover polarity dimensions
    print("[1/3] Discovering polarity dimensions...")
    print("-" * 70)

    antonym_pairs = get_antonym_pairs()
    discovery = PolarityDimensionDiscovery(model)

    polarity_dims = discovery.discover_polarity_dimensions(
        antonym_pairs,
        top_k=num_polarity_dims,
        min_consistency=0.6
    )

    print(f"  Discovered {len(polarity_dims)} polarity dimensions:")
    for i, dim_idx in enumerate(polarity_dims, 1):
        print(f"    {i}. Dimension {dim_idx}")
    print()

    # Step 2: Add polarity constraints
    print("[2/3] Adding polarity constraints...")
    print("-" * 70)

    constraint_loss.set_polarity_dimensions(polarity_dims)

    for word1, word2 in antonym_pairs:
        constraint_loss.add_polarity_constraint(word1, word2, weight=2.0)

    stats = constraint_loss.get_statistics()
    print(f"  Added {stats['num_polarity_constraints']} polarity constraints")
    print(f"  Total constraints: {stats['total_constraints']} distance + {stats['num_polarity_constraints']} polarity")
    print()

    # Step 3: Fine-tune with polarity constraints
    print("[3/3] Fine-tuning with polarity constraints...")
    print("-" * 70)
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print()

    for epoch in range(epochs):
        # Compute loss with both distance and polarity constraints
        loss, loss_stats = constraint_loss.compute_total_loss(model)
        gradients = constraint_loss.compute_gradients(model)

        # Update embeddings
        model.embeddings -= lr * gradients

        # Log progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"total_loss={loss:.4f}, "
                  f"distance_loss={loss_stats['distance_loss']:.4f}, "
                  f"polarity_loss={loss_stats['polarity_loss']:.4f}, "
                  f"polarity_satisfied={loss_stats['polarity_satisfaction_rate']*100:.1f}%")

    print()
    print("Stage 2 complete!")
    print()

    # Final statistics
    final_loss, final_stats = constraint_loss.compute_total_loss(model)
    print("Final statistics:")
    print(f"  Total loss: {final_loss:.4f}")
    print(f"  Distance loss: {final_stats['distance_loss']:.4f}")
    print(f"  Polarity loss: {final_stats['polarity_loss']:.4f}")
    print(f"  Distance satisfaction: {final_stats['satisfaction_rate']*100:.1f}%")
    print(f"  Polarity satisfaction: {final_stats['polarity_satisfaction_rate']*100:.1f}%")
    print(f"  Overall satisfaction: {final_stats['overall_satisfaction_rate']*100:.1f}%")
    print()

    return polarity_dims


def main():
    parser = argparse.ArgumentParser(description='Two-stage training with polarity constraints')
    parser.add_argument('--knowledge-graph', type=str, required=True,
                       help='Path to knowledge graph file')
    parser.add_argument('--embedding-dim', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--num-anchor-dims', type=int, default=51,
                       help='Number of anchor dimensions')

    # Stage 1 parameters
    parser.add_argument('--stage1-epochs', type=int, default=100,
                       help='Stage 1 training epochs')
    parser.add_argument('--stage1-lr', type=float, default=0.001,
                       help='Stage 1 learning rate')
    parser.add_argument('--stage1-batch-size', type=int, default=32,
                       help='Stage 1 batch size')

    # Stage 2 parameters
    parser.add_argument('--stage2-epochs', type=int, default=50,
                       help='Stage 2 training epochs')
    parser.add_argument('--stage2-lr', type=float, default=0.0005,
                       help='Stage 2 learning rate (typically lower)')
    parser.add_argument('--stage2-batch-size', type=int, default=32,
                       help='Stage 2 batch size')
    parser.add_argument('--num-polarity-dims', type=int, default=10,
                       help='Number of polarity dimensions to discover')

    # Output parameters
    parser.add_argument('--output-dir', type=str, default='checkpoints_polarity',
                       help='Directory for saving checkpoints')
    parser.add_argument('--test-logical-ops', action='store_true',
                       help='Test logical operators after training')

    args = parser.parse_args()

    print("=" * 70)
    print("TWO-STAGE POLARITY-AWARE EMBEDDING TRAINING")
    print("=" * 70)
    print()

    # Load knowledge graph
    print("Loading knowledge graph...")
    graph = KnowledgeGraph.load(args.knowledge_graph)
    print(f"  Loaded {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    print()

    # Initialize model
    print("Initializing embedding model...")
    model = EmbeddingModel(
        vocab_size=len(graph.nodes),
        embedding_dim=args.embedding_dim,
        num_anchor_dims=args.num_anchor_dims
    )

    # Build vocabulary from graph
    model.build_vocabulary_from_graph(graph)
    print(f"  Vocabulary size: {len(model.word_to_id)}")
    print(f"  Embedding dimensions: {args.embedding_dim}")
    print(f"  Anchor dimensions: {args.num_anchor_dims}")
    print()

    # Initialize constraint loss
    constraint_loss = ConstraintLoss()
    constraint_loss.add_constraints_from_graph(graph)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # STAGE 1: Pre-train with distance constraints
    stage1_pretrain(
        model=model,
        constraint_loss=constraint_loss,
        epochs=args.stage1_epochs,
        lr=args.stage1_lr,
        batch_size=args.stage1_batch_size
    )

    # Save stage 1 checkpoint
    stage1_checkpoint = output_dir / "stage1_pretrain"
    model.save_checkpoint(str(stage1_checkpoint))
    print(f"Saved stage 1 checkpoint: {stage1_checkpoint}")
    print()

    # STAGE 2: Discover polarity dimensions and fine-tune
    polarity_dims = stage2_polarity_training(
        model=model,
        constraint_loss=constraint_loss,
        epochs=args.stage2_epochs,
        lr=args.stage2_lr,
        batch_size=args.stage2_batch_size,
        num_polarity_dims=args.num_polarity_dims
    )

    # Save stage 2 checkpoint
    stage2_checkpoint = output_dir / "stage2_polarity"
    model.save_checkpoint(str(stage2_checkpoint))
    print(f"Saved stage 2 checkpoint: {stage2_checkpoint}")
    print()

    # Save polarity dimensions
    polarity_file = output_dir / "polarity_dimensions.json"
    with open(polarity_file, 'w') as f:
        json.dump({'polarity_dimensions': polarity_dims}, f, indent=2)
    print(f"Saved polarity dimensions: {polarity_file}")
    print()

    # Test logical operators
    if args.test_logical_ops:
        print("Testing logical operators...")
        print()
        test_logical_operators(model, polarity_dims)

    print("=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print()
    print(f"Checkpoints saved to: {output_dir}")
    print()
    print("Next steps:")
    print("  1. Run dimensional analysis: python scripts/analyze_dimensions.py")
    print("  2. Test embedding quality: python scripts/test_embeddings.py")
    print("  3. Validate NOT operation: Check logical operator results above")
    print()


if __name__ == "__main__":
    main()
