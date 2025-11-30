"""
Discovered Dimension Visualization

Visualizes transparent semantic dimensions discovered during training.

Generates:
1. **Dimension Heatmap** - Shows which words activate which dimensions
2. **t-SNE Plot** - 2D projection of embedding space colored by semantic groups
3. **Polarity Analysis** - Visualizes antonym opposition patterns
4. **Sparsity Distribution** - Histogram of per-dimension activation
5. **Semantic Axis Examples** - Top words for each discovered dimension

Usage:
    python scripts/visualize_discovered_dimensions.py \\
        --embeddings checkpoints/phase1_bootstrap/embeddings_best.npy \\
        --vocabulary checkpoints/phase1_bootstrap/vocabulary.json \\
        --polarity-dims checkpoints/phase1_bootstrap/polarity_dimensions.json \\
        --output-dir results/visualizations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Dict, Tuple
from collections import defaultdict

# Optional: t-SNE for 2D visualization
try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False
    print("Warning: scikit-learn not available, t-SNE plots will be skipped")


def load_data(embeddings_path: str, vocab_path: str, polarity_path: str = None):
    """Load embeddings, vocabulary, and polarity dimensions."""
    # Load embeddings
    embeddings = np.load(embeddings_path)

    # Load vocabulary
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
        word_to_id = vocab_data['word_to_id']
        id_to_word = vocab_data['id_to_word']
        id_to_word = {int(k): v for k, v in id_to_word.items()}

    # Load polarity dimensions
    polarity_dims = None
    dim_scores = None
    if polarity_path and Path(polarity_path).exists():
        with open(polarity_path, 'r') as f:
            polarity_data = json.load(f)
            polarity_dims = polarity_data.get('polarity_dims', [])
            dim_scores = polarity_data.get('dim_scores', [])

    return embeddings, word_to_id, id_to_word, polarity_dims, dim_scores


def plot_dimension_heatmap(embeddings: np.ndarray,
                            id_to_word: Dict[int, str],
                            dims_to_show: List[int],
                            output_path: str,
                            max_words: int = 50):
    """Plot heatmap showing which words activate which dimensions."""
    print(f"\n[1/5] Generating dimension heatmap...")

    # Select top words by variance on these dimensions
    variances = []
    for word_idx in range(min(len(id_to_word), 1000)):  # Sample 1000 words
        word_emb = embeddings[word_idx, dims_to_show]
        variances.append((word_idx, np.var(word_emb)))

    variances.sort(key=lambda x: x[1], reverse=True)
    top_word_indices = [idx for idx, _ in variances[:max_words]]

    # Create heatmap data
    heatmap_data = embeddings[top_word_indices][:, dims_to_show]

    # Get word labels
    word_labels = []
    for idx in top_word_indices:
        word = id_to_word[idx]
        word_display = word.split('_wn.')[0] if '_wn.' in word else word
        word_labels.append(word_display[:15])  # Truncate long words

    # Plot
    fig, ax = plt.subplots(figsize=(12, max_words // 2))
    im = ax.imshow(heatmap_data, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)

    ax.set_xticks(np.arange(len(dims_to_show)))
    ax.set_xticklabels([f"D{d}" for d in dims_to_show], rotation=45)
    ax.set_yticks(np.arange(len(word_labels)))
    ax.set_yticklabels(word_labels, fontsize=8)

    ax.set_xlabel("Dimension")
    ax.set_ylabel("Word")
    ax.set_title("Dimension Activation Heatmap (Top 50 Variant Words)")

    plt.colorbar(im, ax=ax, label="Activation Value")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def plot_tsne(embeddings: np.ndarray,
              id_to_word: Dict[int, str],
              word_to_id: Dict[str, int],
              output_path: str,
              max_points: int = 1000):
    """Plot t-SNE 2D projection of embeddings."""
    if not TSNE_AVAILABLE:
        print(f"\n[2/5] Skipping t-SNE (sklearn not available)")
        return

    print(f"\n[2/5] Generating t-SNE visualization...")

    # Sample words for visualization
    sample_indices = np.random.choice(len(id_to_word), min(max_points, len(id_to_word)), replace=False)
    sample_embeddings = embeddings[sample_indices]

    # Run t-SNE
    print("  Running t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(sample_indices) - 1))
    tsne_result = tsne.fit_transform(sample_embeddings)

    # Define semantic groups for coloring
    semantic_groups = {
        'positive': {'good', 'excellent', 'great', 'wonderful', 'beautiful', 'happy', 'joy'},
        'negative': {'bad', 'terrible', 'awful', 'horrible', 'ugly', 'sad', 'miserable'},
        'hot': {'hot', 'warm', 'heat', 'burning', 'fire'},
        'cold': {'cold', 'cool', 'freezing', 'ice', 'chilly'},
        'large': {'big', 'large', 'huge', 'enormous', 'giant'},
        'small': {'small', 'tiny', 'little', 'miniature', 'minute'},
    }

    # Assign colors
    colors = []
    labels = []
    for idx in sample_indices:
        word = id_to_word[idx]
        word_base = word.split('_wn.')[0] if '_wn.' in word else word

        assigned = False
        for group_name, group_words in semantic_groups.items():
            if word_base in group_words:
                colors.append(group_name)
                labels.append(word_base)
                assigned = True
                break

        if not assigned:
            colors.append('other')
            labels.append('')

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    color_map = {
        'positive': 'green',
        'negative': 'red',
        'hot': 'orange',
        'cold': 'blue',
        'large': 'purple',
        'small': 'brown',
        'other': 'lightgray'
    }

    for group_name in semantic_groups.keys():
        mask = np.array(colors) == group_name
        if np.any(mask):
            ax.scatter(tsne_result[mask, 0], tsne_result[mask, 1],
                      c=color_map[group_name], label=group_name, s=50, alpha=0.7)

    # Plot 'other' in background
    other_mask = np.array(colors) == 'other'
    ax.scatter(tsne_result[other_mask, 0], tsne_result[other_mask, 1],
              c='lightgray', s=10, alpha=0.3, label='other')

    # Annotate semantic group words
    for i, (x, y) in enumerate(tsne_result):
        if labels[i]:  # Only label semantic group words
            ax.annotate(labels[i], (x, y), fontsize=8, alpha=0.7)

    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_title("t-SNE Visualization of Embedding Space")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def plot_polarity_analysis(embeddings: np.ndarray,
                            id_to_word: Dict[int, str],
                            word_to_id: Dict[str, int],
                            polarity_dims: List[int],
                            output_path: str):
    """Visualize antonym opposition patterns."""
    print(f"\n[3/5] Generating polarity analysis...")

    # Test antonym pairs
    test_pairs = [
        ('good', 'bad'),
        ('hot', 'cold'),
        ('big', 'small'),
        ('fast', 'slow'),
        ('happy', 'sad'),
        ('light', 'dark'),
        ('high', 'low'),
        ('strong', 'weak'),
    ]

    # Find pairs that exist in vocabulary
    valid_pairs = []
    for word1, word2 in test_pairs:
        # Find indices (handles sense-tagged)
        idx1 = None
        idx2 = None

        for vocab_word, idx in word_to_id.items():
            if vocab_word.startswith(word1 + "_wn.") and idx1 is None:
                idx1 = idx
            if vocab_word.startswith(word2 + "_wn.") and idx2 is None:
                idx2 = idx

        if idx1 is not None and idx2 is not None:
            valid_pairs.append((word1, word2, idx1, idx2))

    if not valid_pairs:
        print("  No valid antonym pairs found in vocabulary")
        return

    # Plot opposition on each polarity dimension
    num_pairs = len(valid_pairs)
    num_dims = min(len(polarity_dims), 10) if polarity_dims else embeddings.shape[1]

    fig, axes = plt.subplots(num_dims, 1, figsize=(10, num_dims * 1.5))
    if num_dims == 1:
        axes = [axes]

    dims_to_plot = polarity_dims[:num_dims] if polarity_dims else list(range(num_dims))

    for ax_idx, dim in enumerate(dims_to_plot):
        ax = axes[ax_idx]

        # For each pair, plot their values on this dimension
        pair_positions = []
        for i, (word1, word2, idx1, idx2) in enumerate(valid_pairs):
            val1 = embeddings[idx1, dim]
            val2 = embeddings[idx2, dim]

            # Plot as opposing bars
            ax.barh(i * 2, val1, color='green' if val1 > 0 else 'red', alpha=0.7)
            ax.barh(i * 2 + 0.5, val2, color='green' if val2 > 0 else 'red', alpha=0.7)

            # Add labels
            ax.text(0, i * 2, word1, ha='center', va='center', fontsize=8, fontweight='bold')
            ax.text(0, i * 2 + 0.5, word2, ha='center', va='center', fontsize=8, fontweight='bold')

            pair_positions.append(i * 2 + 0.25)

        ax.axvline(x=0, color='black', linewidth=1, linestyle='-')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-0.5, num_pairs * 2)
        ax.set_yticks([])
        ax.set_xlabel("Dimension Value")
        ax.set_title(f"Dimension {dim} (Polarity Rank #{ax_idx + 1})")
        ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def plot_sparsity_distribution(embeddings: np.ndarray, output_path: str):
    """Plot histogram of per-dimension activation rates."""
    print(f"\n[4/5] Generating sparsity distribution...")

    # Compute activation rate per dimension
    activation_per_dim = np.mean(np.abs(embeddings) > 0.01, axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Histogram
    axes[0].hist(activation_per_dim, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0.4, color='green', linestyle='--', label='Target range (40-70%)')
    axes[0].axvline(x=0.7, color='green', linestyle='--')
    axes[0].set_xlabel("Activation Rate (% of words using dimension)")
    axes[0].set_ylabel("Number of Dimensions")
    axes[0].set_title("Dimension Activation Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Line plot (dimension index vs activation)
    axes[1].plot(activation_per_dim, linewidth=1, color='steelblue')
    axes[1].axhline(y=0.4, color='green', linestyle='--', alpha=0.5, label='Target range')
    axes[1].axhline(y=0.7, color='green', linestyle='--', alpha=0.5)
    axes[1].set_xlabel("Dimension Index")
    axes[1].set_ylabel("Activation Rate")
    axes[1].set_title("Per-Dimension Activation")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def generate_semantic_axis_report(embeddings: np.ndarray,
                                    id_to_word: Dict[int, str],
                                    polarity_dims: List[int],
                                    dim_scores: List[Tuple],
                                    output_path: str,
                                    top_n: int = 10):
    """Generate text report of semantic axes with examples."""
    print(f"\n[5/5] Generating semantic axis report...")

    if not polarity_dims:
        print("  No polarity dimensions to report")
        return

    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("DISCOVERED SEMANTIC AXES REPORT\n")
        f.write("="*70 + "\n\n")

        f.write(f"Total polarity dimensions discovered: {len(polarity_dims)}\n\n")

        # Report each dimension
        for rank, (dim_idx, score, consistency, disc_power) in enumerate(dim_scores[:20], 1):
            dim_values = embeddings[:, dim_idx]

            f.write("="*70 + "\n")
            f.write(f"DIMENSION {dim_idx} (Rank #{rank})\n")
            f.write("="*70 + "\n")
            f.write(f"  Interpretability Score: {score:.4f}\n")
            f.write(f"  Consistency: {consistency:.2%}\n")
            f.write(f"  Discriminative Power: {disc_power:.4f}\n")
            f.write(f"  Variance: {np.var(dim_values):.4f}\n")
            f.write(f"  Activation Rate: {np.mean(np.abs(dim_values) > 0.01):.1%}\n\n")

            # Find extremes
            sorted_indices = np.argsort(dim_values)

            # Positive pole
            f.write("  POSITIVE POLE:\n")
            for idx in sorted_indices[-top_n:][::-1]:
                word = id_to_word[idx]
                word_display = word.split('_wn.')[0] if '_wn.' in word else word
                value = dim_values[idx]
                if abs(value) > 0.01:
                    f.write(f"    {word_display:25s} {value:+.3f}\n")

            f.write("\n")

            # Negative pole
            f.write("  NEGATIVE POLE:\n")
            for idx in sorted_indices[:top_n]:
                word = id_to_word[idx]
                word_display = word.split('_wn.')[0] if '_wn.' in word else word
                value = dim_values[idx]
                if abs(value) > 0.01:
                    f.write(f"    {word_display:25s} {value:+.3f}\n")

            f.write("\n")

        # Summary statistics
        f.write("="*70 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("="*70 + "\n")

        # Sparsity
        sparsity = np.mean(np.abs(embeddings) < 0.01)
        f.write(f"  Overall sparsity: {sparsity:.1%}\n")

        # Dimension usage
        activation_per_dim = np.mean(np.abs(embeddings) > 0.01, axis=0)
        f.write(f"  Mean dimension activation: {np.mean(activation_per_dim):.1%}\n")
        f.write(f"  Median dimension activation: {np.median(activation_per_dim):.1%}\n\n")

        # Categorize
        unused = np.sum(activation_per_dim < 0.1)
        sparse = np.sum((activation_per_dim >= 0.1) & (activation_per_dim < 0.3))
        moderate = np.sum((activation_per_dim >= 0.3) & (activation_per_dim < 0.6))
        saturated = np.sum(activation_per_dim >= 0.6)

        f.write("  Dimension categories:\n")
        f.write(f"    Unused (<10%):     {unused:3d} dimensions\n")
        f.write(f"    Sparse (10-30%):   {sparse:3d} dimensions\n")
        f.write(f"    Moderate (30-60%): {moderate:3d} dimensions\n")
        f.write(f"    Saturated (>60%):  {saturated:3d} dimensions\n")

    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize discovered dimensions')
    parser.add_argument('--embeddings', type=str, required=True,
                       help='Path to embeddings (.npy file)')
    parser.add_argument('--vocabulary', type=str, required=True,
                       help='Path to vocabulary (.json file)')
    parser.add_argument('--polarity-dims', type=str, default=None,
                       help='Path to polarity dimensions (.json file)')
    parser.add_argument('--output-dir', type=str, default='results/visualizations',
                       help='Output directory for visualizations')

    args = parser.parse_args()

    print("="*70)
    print("DISCOVERED DIMENSION VISUALIZATION")
    print("="*70)

    # Load data
    print("\nLoading data...")
    embeddings, word_to_id, id_to_word, polarity_dims, dim_scores = load_data(
        args.embeddings, args.vocabulary, args.polarity_dims
    )

    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Vocabulary: {len(word_to_id):,} words")
    if polarity_dims:
        print(f"  Polarity dimensions: {len(polarity_dims)}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    dims_to_show = polarity_dims[:20] if polarity_dims else list(range(min(20, embeddings.shape[1])))

    plot_dimension_heatmap(
        embeddings, id_to_word, dims_to_show,
        output_dir / "dimension_heatmap.png"
    )

    plot_tsne(
        embeddings, id_to_word, word_to_id,
        output_dir / "tsne_projection.png"
    )

    if polarity_dims:
        plot_polarity_analysis(
            embeddings, id_to_word, word_to_id, polarity_dims,
            output_dir / "polarity_analysis.png"
        )

    plot_sparsity_distribution(
        embeddings,
        output_dir / "sparsity_distribution.png"
    )

    if polarity_dims and dim_scores:
        generate_semantic_axis_report(
            embeddings, id_to_word, polarity_dims, dim_scores,
            output_dir / "semantic_axes_report.txt"
        )

    # Summary
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\nGenerated visualizations:")
    print(f"  - dimension_heatmap.png")
    print(f"  - tsne_projection.png")
    if polarity_dims:
        print(f"  - polarity_analysis.png")
        print(f"  - semantic_axes_report.txt")
    print(f"  - sparsity_distribution.png")
    print(f"\nOutput directory: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
