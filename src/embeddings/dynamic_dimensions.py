"""
Dynamic Dimension Management

Enables the embedding model to expand dimensions during training as needed.

Key Principles:
1. Start with reasonable number of dimensions (128)
2. Monitor sparsity across all dimensions
3. If all dimensions are heavily utilized (>70% non-zero), add more
4. Initialize new dimensions with small random values
5. Continue training with expanded space

This prevents both:
- Under-capacity: Too few dimensions to capture semantic richness
- Over-capacity: Too many dimensions leading to overfitting

Usage:
    from src.embeddings.dynamic_dimensions import should_add_dimension, expand_embeddings

    if should_add_dimension(embeddings, threshold=0.3):
        embeddings, model = expand_embeddings(model, embeddings, num_new=16)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict


def compute_dimension_sparsity(embeddings: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Compute sparsity for each dimension across all words.

    Args:
        embeddings: (vocab_size, num_dims) array
        epsilon: Threshold for considering value as zero

    Returns:
        (num_dims,) array of sparsity values (0.0 = all zeros, 1.0 = all non-zero)
    """
    # Count non-zero values per dimension
    non_zero_counts = np.sum(np.abs(embeddings) > epsilon, axis=0)

    # Sparsity = fraction of non-zero values
    vocab_size = embeddings.shape[0]
    sparsity = non_zero_counts / vocab_size

    return sparsity


def should_add_dimension(embeddings: np.ndarray,
                         sparsity_threshold: float = 0.3,
                         min_saturated_dims: int = None) -> bool:
    """
    Determine if we need to add more dimensions.

    We add dimensions when most existing dimensions are heavily utilized,
    suggesting the model needs more capacity to represent semantic distinctions.

    Args:
        embeddings: (vocab_size, num_dims) array
        sparsity_threshold: Minimum sparsity (0.3 = 30% of words use this dim)
        min_saturated_dims: Minimum number of saturated dimensions before adding
                           (default: 80% of current dimensions)

    Returns:
        True if we should add dimensions, False otherwise
    """
    num_dims = embeddings.shape[1]

    if min_saturated_dims is None:
        min_saturated_dims = int(0.8 * num_dims)

    # Compute sparsity for all dimensions
    dim_sparsity = compute_dimension_sparsity(embeddings)

    # Count how many dimensions are saturated (heavily used)
    saturated_dims = np.sum(dim_sparsity >= sparsity_threshold)

    # Add dimensions if most are saturated
    should_add = saturated_dims >= min_saturated_dims

    return should_add


def expand_embeddings(model: nn.Module,
                     embeddings: np.ndarray,
                     num_new_dims: int = 16,
                     init_scale: float = 0.001) -> Tuple[np.ndarray, nn.Module]:
    """
    Expand embedding dimensions by adding new columns.

    Args:
        model: The embedding model (must have 'embeddings' parameter)
        embeddings: Current (vocab_size, num_dims) array
        num_new_dims: Number of dimensions to add
        init_scale: Scale for initializing new dimensions (small values)

    Returns:
        (new_embeddings, updated_model) tuple
    """
    vocab_size, current_dims = embeddings.shape

    # Create new dimensions with small random initialization
    new_dims = np.random.randn(vocab_size, num_new_dims) * init_scale

    # Concatenate to existing embeddings
    expanded_embeddings = np.concatenate([embeddings, new_dims], axis=1)

    # Update model's embedding parameter
    model.embeddings = nn.Parameter(
        torch.from_numpy(expanded_embeddings).float()
    )

    # Update model's embedding_dim attribute
    model.embedding_dim = current_dims + num_new_dims

    return expanded_embeddings, model


def get_dimension_statistics(embeddings: np.ndarray,
                             epsilon: float = 1e-6) -> Dict:
    """
    Get comprehensive statistics about dimension usage.

    Args:
        embeddings: (vocab_size, num_dims) array
        epsilon: Threshold for zero values

    Returns:
        Dictionary with statistics
    """
    num_dims = embeddings.shape[1]
    dim_sparsity = compute_dimension_sparsity(embeddings, epsilon)

    # Compute L1 norm per dimension (total magnitude)
    dim_l1_norm = np.sum(np.abs(embeddings), axis=0)

    # Compute variance per dimension
    dim_variance = np.var(embeddings, axis=0)

    # Categorize dimensions by usage
    unused_dims = np.sum(dim_sparsity < 0.1)  # < 10% of words use it
    sparse_dims = np.sum((dim_sparsity >= 0.1) & (dim_sparsity < 0.3))
    moderate_dims = np.sum((dim_sparsity >= 0.3) & (dim_sparsity < 0.7))
    saturated_dims = np.sum(dim_sparsity >= 0.7)  # >= 70% of words use it

    stats = {
        'num_dimensions': num_dims,
        'mean_sparsity': float(np.mean(dim_sparsity)),
        'median_sparsity': float(np.median(dim_sparsity)),
        'min_sparsity': float(np.min(dim_sparsity)),
        'max_sparsity': float(np.max(dim_sparsity)),
        'unused_dimensions': int(unused_dims),
        'sparse_dimensions': int(sparse_dims),
        'moderate_dimensions': int(moderate_dims),
        'saturated_dimensions': int(saturated_dims),
        'mean_l1_norm': float(np.mean(dim_l1_norm)),
        'mean_variance': float(np.mean(dim_variance)),
        'dimension_sparsity': dim_sparsity.tolist(),
        'dimension_l1_norm': dim_l1_norm.tolist(),
        'dimension_variance': dim_variance.tolist()
    }

    return stats


def prune_unused_dimensions(embeddings: np.ndarray,
                            model: nn.Module,
                            min_sparsity: float = 0.05,
                            epsilon: float = 1e-6) -> Tuple[np.ndarray, nn.Module, np.ndarray]:
    """
    Remove dimensions that are barely used (optional, for cleanup).

    Args:
        embeddings: (vocab_size, num_dims) array
        model: Embedding model
        min_sparsity: Minimum sparsity to keep a dimension
        epsilon: Zero threshold

    Returns:
        (pruned_embeddings, updated_model, kept_indices) tuple
    """
    dim_sparsity = compute_dimension_sparsity(embeddings, epsilon)

    # Find dimensions to keep (above minimum sparsity)
    kept_dims = dim_sparsity >= min_sparsity
    kept_indices = np.where(kept_dims)[0]

    # Prune embeddings
    pruned_embeddings = embeddings[:, kept_dims]

    # Update model
    model.embeddings = nn.Parameter(
        torch.from_numpy(pruned_embeddings).float()
    )
    model.embedding_dim = pruned_embeddings.shape[1]

    return pruned_embeddings, model, kept_indices


def log_dimension_info(embeddings: np.ndarray, epoch: int = None):
    """
    Log human-readable dimension statistics.

    Args:
        embeddings: (vocab_size, num_dims) array
        epoch: Optional epoch number for logging
    """
    stats = get_dimension_statistics(embeddings)

    prefix = f"[Epoch {epoch}] " if epoch is not None else ""

    print(f"\n{prefix}Dimension Statistics:")
    print(f"  Total dimensions: {stats['num_dimensions']}")
    print(f"  Mean sparsity: {stats['mean_sparsity']:.1%}")
    print(f"  Dimension usage distribution:")
    print(f"    Unused (<10%): {stats['unused_dimensions']}")
    print(f"    Sparse (10-30%): {stats['sparse_dimensions']}")
    print(f"    Moderate (30-70%): {stats['moderate_dimensions']}")
    print(f"    Saturated (>70%): {stats['saturated_dimensions']}")

    if stats['saturated_dimensions'] >= 0.8 * stats['num_dimensions']:
        print(f"  ⚠️  WARNING: {stats['saturated_dimensions']}/{stats['num_dimensions']} dimensions saturated!")
        print(f"      Consider adding more dimensions for better capacity.")
    elif stats['unused_dimensions'] >= 0.3 * stats['num_dimensions']:
        print(f"  ℹ️  INFO: {stats['unused_dimensions']}/{stats['num_dimensions']} dimensions unused.")
        print(f"      Model has sufficient capacity, may be able to prune.")


def adaptive_dimension_management(model: nn.Module,
                                 embeddings: np.ndarray,
                                 epoch: int,
                                 max_dims: int = 512,
                                 check_interval: int = 10,
                                 sparsity_threshold: float = 0.3,
                                 expand_by: int = 16) -> Tuple[np.ndarray, nn.Module, bool]:
    """
    Automatically manage dimensions during training.

    Call this every epoch. It will:
    1. Check if dimensions need expansion (every check_interval epochs)
    2. Expand if needed (up to max_dims)
    3. Log statistics

    Args:
        model: Embedding model
        embeddings: Current embeddings
        epoch: Current epoch number
        max_dims: Maximum allowed dimensions
        check_interval: How often to check for expansion (epochs)
        sparsity_threshold: Threshold for saturation
        expand_by: How many dimensions to add when expanding

    Returns:
        (embeddings, model, did_expand) tuple
    """
    did_expand = False

    # Check if it's time to evaluate expansion
    if epoch > 0 and epoch % check_interval == 0:
        current_dims = embeddings.shape[1]

        if current_dims < max_dims:
            # Check if we should add dimensions
            if should_add_dimension(embeddings, sparsity_threshold):
                # Calculate how many to add (don't exceed max)
                num_to_add = min(expand_by, max_dims - current_dims)

                print(f"\n[Epoch {epoch}] Adding {num_to_add} dimensions ({current_dims} → {current_dims + num_to_add})")

                embeddings, model = expand_embeddings(model, embeddings, num_to_add)
                did_expand = True

    # Log statistics (less frequently to avoid spam)
    if epoch % (check_interval * 2) == 0 or did_expand:
        log_dimension_info(embeddings, epoch)

    return embeddings, model, did_expand
