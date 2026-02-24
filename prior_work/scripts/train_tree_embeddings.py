"""
NAOMI-II Structure-Aware Embedding Training

Trains semantic embeddings using Tree-LSTM composition with distance constraints.

Architecture:
- Tree-LSTM encoder for structure-aware composition
- Word embeddings trained from both tree structure AND distance constraints
- Combined loss: composition + distance + regularization

Usage:
    python scripts/train_tree_embeddings.py [--epochs 100] [--lr 0.001]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from tree_dataset import load_training_data, collate_tree_batch, collate_distance_batch
from tree_loss import CombinedLoss, SimpleCompositionLoss, DistanceLoss
from src.embeddings.tree_lstm import TreeLSTMEncoder
from src.embeddings.device import DeviceManager
from src.embeddings.anchors import AnchorDimensions


def train_epoch(model, tree_loader, distance_loader, criterion, optimizer, device, epoch):
    """
    Train for one epoch with both tree and distance data.

    Args:
        model: TreeLSTMEncoder
        tree_loader: DataLoader for tree structures
        distance_loader: DataLoader for distance constraints
        criterion: Combined loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number

    Returns:
        Average loss components
    """
    model.train()

    total_losses = {
        'total': 0.0,
        'composition': 0.0,
        'distance': 0.0,
        'regularization': 0.0
    }

    num_batches = 0

    # Create iterators
    tree_iter = iter(tree_loader)
    distance_iter = iter(distance_loader)

    # Progress bar
    pbar = tqdm(range(max(len(tree_loader), len(distance_loader))), desc=f"Epoch {epoch}")

    for _ in pbar:
        # Get tree batch
        try:
            tree_batch = next(tree_iter)
        except StopIteration:
            tree_iter = iter(tree_loader)
            tree_batch = next(tree_iter)

        # Get distance batch
        try:
            distance_batch = next(distance_iter)
        except StopIteration:
            distance_iter = iter(distance_loader)
            distance_batch = next(distance_iter)

        # Move distance batch to device
        distance_batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in distance_batch.items()
        }

        # Encode trees
        hypotheses = tree_batch['hypotheses']
        indices = tree_batch['indices']

        encoded_trees = []
        for hyp in hypotheses:
            try:
                encoded = model(hyp, model.word_to_id, device)
                encoded_trees.append(encoded)
            except Exception as e:
                # Skip failed encodings
                print(f"\nWarning: Failed to encode tree: {e}")
                continue

        if len(encoded_trees) == 0:
            continue

        encoded_trees = torch.cat(encoded_trees, dim=0)  # (batch_size, embedding_dim)

        # Get word embeddings
        word_embeddings = model.word_embeddings.weight

        # Compute loss
        losses = criterion(
            encoded_trees=encoded_trees,
            tree_indices=indices[:len(encoded_trees)],
            word_embeddings=word_embeddings,
            distance_batch=distance_batch
        )

        total_loss = losses['total']

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate losses
        for key in total_losses:
            total_losses[key] += losses[key].item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{total_loss.item():.4f}",
            'comp': f"{losses['composition'].item():.4f}",
            'dist': f"{losses['distance'].item():.4f}"
        })

    # Average losses
    for key in total_losses:
        total_losses[key] /= max(num_batches, 1)

    return total_losses


def validate(model, tree_loader, distance_loader, criterion, device):
    """
    Validate on both tree and distance data.

    Args:
        model: TreeLSTMEncoder
        tree_loader: DataLoader for tree structures
        distance_loader: DataLoader for distance constraints
        criterion: Combined loss function
        device: Device to validate on

    Returns:
        Average loss components
    """
    model.eval()

    total_losses = {
        'total': 0.0,
        'composition': 0.0,
        'distance': 0.0,
        'regularization': 0.0
    }

    num_batches = 0

    with torch.no_grad():
        tree_iter = iter(tree_loader)
        distance_iter = iter(distance_loader)

        for _ in range(min(len(tree_loader), len(distance_loader))):
            try:
                tree_batch = next(tree_iter)
                distance_batch = next(distance_iter)
            except StopIteration:
                break

            # Move distance batch to device
            distance_batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in distance_batch.items()
            }

            # Encode trees
            hypotheses = tree_batch['hypotheses']
            indices = tree_batch['indices']

            encoded_trees = []
            for hyp in hypotheses:
                try:
                    encoded = model(hyp, model.word_to_id, device)
                    encoded_trees.append(encoded)
                except:
                    continue

            if len(encoded_trees) == 0:
                continue

            encoded_trees = torch.cat(encoded_trees, dim=0)

            # Get word embeddings
            word_embeddings = model.word_embeddings.weight

            # Compute loss
            losses = criterion(
                encoded_trees=encoded_trees,
                tree_indices=indices[:len(encoded_trees)],
                word_embeddings=word_embeddings,
                distance_batch=distance_batch
            )

            # Accumulate
            for key in total_losses:
                total_losses[key] += losses[key].item()
            num_batches += 1

    # Average
    for key in total_losses:
        total_losses[key] /= max(num_batches, 1)

    return total_losses


def save_checkpoint(model, optimizer, epoch, losses, filepath, word_to_id, id_to_word):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'vocab_size': model.vocab_size,
        'embedding_dim': model.embedding_dim,
        'hidden_dim': model.hidden_dim,
        'word_to_id': word_to_id,
        'id_to_word': id_to_word
    }
    torch.save(checkpoint, filepath)


def main():
    parser = argparse.ArgumentParser(description='Train structure-aware embeddings')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--embedding-dim', type=int, default=128, help='Embedding dimensionality')
    parser.add_argument('--hidden-dim', type=int, default=128, help='LSTM hidden dimensionality')
    parser.add_argument('--comp-weight', type=float, default=0.7, help='Composition loss weight')
    parser.add_argument('--dist-weight', type=float, default=0.3, help='Distance loss weight')

    args = parser.parse_args()

    print("=" * 70)
    print("NAOMI-II STRUCTURE-AWARE EMBEDDING TRAINING")
    print("=" * 70)
    print()

    # Configuration
    TRAINING_DATA_DIR = Path(__file__).parent.parent / "data" / "training"
    OUTPUT_DIR = Path(__file__).parent.parent / "checkpoints"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize device
    print("[1/7] Initializing device...")
    device_manager = DeviceManager(prefer_npu=False, verbose=True)
    device = device_manager.device
    print()

    # Load training data
    print("[2/7] Loading training data...")
    tree_dataset, distance_dataset, word_to_id, id_to_word = load_training_data(
        TRAINING_DATA_DIR
    )

    vocab_size = len(word_to_id)
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Tree samples: {len(tree_dataset)}")
    print(f"  Distance constraints: {len(distance_dataset)}")
    print()

    # Split datasets
    print("[3/7] Creating data loaders...")

    # Tree dataset split
    tree_val_size = int(len(tree_dataset) * 0.1)
    tree_train_size = len(tree_dataset) - tree_val_size
    tree_train, tree_val = torch.utils.data.random_split(
        tree_dataset, [tree_train_size, tree_val_size]
    )

    # Distance dataset split
    dist_val_size = int(len(distance_dataset) * 0.1)
    dist_train_size = len(distance_dataset) - dist_val_size
    dist_train, dist_val = torch.utils.data.random_split(
        distance_dataset, [dist_train_size, dist_val_size]
    )

    # Create data loaders
    tree_train_loader = DataLoader(
        tree_train, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_tree_batch, num_workers=0
    )
    tree_val_loader = DataLoader(
        tree_val, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_tree_batch, num_workers=0
    )

    dist_train_loader = DataLoader(
        dist_train, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_distance_batch, num_workers=0
    )
    dist_val_loader = DataLoader(
        dist_val, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_distance_batch, num_workers=0
    )

    print(f"  Train: {tree_train_size} trees, {dist_train_size} constraints")
    print(f"  Val: {tree_val_size} trees, {dist_val_size} constraints")
    print()

    # Initialize model
    print("[4/7] Initializing Tree-LSTM model...")
    anchors = AnchorDimensions()
    num_anchors = anchors.num_anchors()
    print(f"  Embedding dim: {args.embedding_dim} ({num_anchors} anchor + {args.embedding_dim - num_anchors} learned)")

    model = TreeLSTMEncoder(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_edge_types=20,
        dropout=0.3
    ).to(device)

    # Store vocab in model for easy access during encoding
    model.word_to_id = word_to_id

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Initialize optimizer and loss
    print("[5/7] Initializing optimizer...")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = CombinedLoss(
        composition_weight=args.comp_weight,
        distance_weight=args.dist_weight,
        num_anchors=num_anchors
    )
    print(f"  Optimizer: Adam (lr={args.lr})")
    print(f"  Loss: Combined (comp={args.comp_weight}, dist={args.dist_weight})")
    print()

    # Training loop
    print("[6/7] Training...")
    print()
    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_losses = train_epoch(
            model, tree_train_loader, dist_train_loader,
            criterion, optimizer, device, epoch
        )

        val_losses = validate(
            model, tree_val_loader, dist_val_loader,
            criterion, device
        )

        print(f"Epoch {epoch}/{args.epochs}")
        print(f"  Train - Total: {train_losses['total']:.4f}, "
              f"Comp: {train_losses['composition']:.4f}, "
              f"Dist: {train_losses['distance']:.4f}, "
              f"Reg: {train_losses['regularization']:.4f}")
        print(f"  Val   - Total: {val_losses['total']:.4f}, "
              f"Comp: {val_losses['composition']:.4f}, "
              f"Dist: {val_losses['distance']:.4f}, "
              f"Reg: {val_losses['regularization']:.4f}")

        # Save checkpoints
        if epoch % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_losses,
                OUTPUT_DIR / f"tree_checkpoint_epoch_{epoch}.pt",
                word_to_id, id_to_word
            )
            print(f"  Saved checkpoint")

        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            save_checkpoint(
                model, optimizer, epoch, val_losses,
                OUTPUT_DIR / "tree_best_model.pt",
                word_to_id, id_to_word
            )
            print(f"  New best model! (val_loss: {val_losses['total']:.4f})")

        print()

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed/60:.1f} minutes")
    print()

    # Save final model
    print("[7/7] Saving final model...")
    save_checkpoint(
        model, optimizer, args.epochs, val_losses,
        OUTPUT_DIR / "tree_final_model.pt",
        word_to_id, id_to_word
    )

    # Extract and save embeddings
    final_embeddings = model.word_embeddings.weight.detach().cpu().numpy()
    np.save(OUTPUT_DIR / "tree_embeddings.npy", final_embeddings)
    print(f"  Saved embeddings to {OUTPUT_DIR / 'tree_embeddings.npy'}")

    import json
    with open(OUTPUT_DIR / "tree_vocabulary.json", 'w', encoding='utf-8') as f:
        json.dump({
            'word_to_id': word_to_id,
            'id_to_word': id_to_word,
            'vocab_size': vocab_size,
            'embedding_dim': args.embedding_dim,
            'hidden_dim': args.hidden_dim
        }, f, indent=2, ensure_ascii=False)
    print(f"  Saved vocabulary to {OUTPUT_DIR / 'tree_vocabulary.json'}")
    print()

    print("=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final embeddings: {OUTPUT_DIR / 'tree_embeddings.npy'}")
    print()


if __name__ == "__main__":
    main()
