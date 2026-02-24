"""
Hierarchical clustering of antonym pairs into semantic axes.

This module implements the core clustering algorithm that groups antonym pairs
into interpretable semantic dimensions using hierarchical agglomerative clustering
with complete linkage and silhouette score optimization.

Author: NAOMI-II Development Team
Date: 2025-11-30
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Handle both direct and package imports
try:
    from .similarity_signals import SimilaritySignals
except ImportError:
    from similarity_signals import SimilaritySignals


class AntonymClusterer:
    """Clusters antonym pairs into semantic axes using hierarchical clustering."""

    def __init__(self, antonym_pairs: List[Dict]):
        """
        Initialize clusterer with antonym pairs.

        Args:
            antonym_pairs: List of dicts with keys:
                - word1, word2: String words
                - synset1, synset2: WordNet synset names
        """
        self.antonym_pairs = antonym_pairs
        self.n_pairs = len(antonym_pairs)

        # Initialize similarity signal computer
        self.signal_computer = SimilaritySignals(antonym_pairs)

        # Will be populated by methods
        self.similarity_matrix = None
        self.distance_matrix = None
        self.linkage_matrix = None
        self.clusters = None
        self.optimal_height = None
        self.silhouette_scores = None

    def build_similarity_matrix(
        self,
        progress_callback: Optional[callable] = None
    ) -> np.ndarray:
        """
        Build pairwise similarity matrix for all antonym pairs.

        Args:
            progress_callback: Optional callback(current, total) for progress

        Returns:
            (N × N) symmetric similarity matrix
        """
        print(f"Building similarity matrix for {self.n_pairs} antonym pairs...")

        self.similarity_matrix = self.signal_computer.compute_similarity_matrix(
            progress_callback=progress_callback
        )

        # Convert to distance matrix
        self.distance_matrix = 1.0 - self.similarity_matrix

        print("Similarity matrix built successfully")
        print(f"  Mean similarity: {self.similarity_matrix.mean():.4f}")
        print(f"  Max similarity: {self.similarity_matrix.max():.4f}")
        print(f"  Min similarity: {self.similarity_matrix.min():.4f}")

        return self.similarity_matrix

    def perform_hierarchical_clustering(
        self,
        method: str = 'complete'
    ) -> np.ndarray:
        """
        Perform hierarchical agglomerative clustering.

        Args:
            method: Linkage method ('complete', 'average', 'ward', 'single')
                   Default: 'complete' (maximum distance between cluster members)

        Returns:
            Linkage matrix (hierarchical clustering structure)
        """
        if self.distance_matrix is None:
            raise ValueError("Must build similarity matrix first")

        print(f"Performing hierarchical clustering (method: {method})...")

        # Convert distance matrix to condensed form for scipy
        condensed_distance = squareform(self.distance_matrix, checks=False)

        # Hierarchical clustering
        self.linkage_matrix = linkage(condensed_distance, method=method)

        print("Hierarchical clustering complete")
        return self.linkage_matrix

    def optimize_cut_height(
        self,
        min_clusters: int = 10,
        max_clusters: int = 200,
        n_steps: int = 100,
        verbose: bool = True
    ) -> Tuple[float, int]:
        """
        Find optimal cut height using silhouette score optimization.

        Args:
            min_clusters: Minimum number of clusters to consider
            max_clusters: Maximum number of clusters to consider
            n_steps: Number of height values to test
            verbose: Print progress

        Returns:
            (optimal_height, optimal_n_clusters)
        """
        if self.linkage_matrix is None:
            raise ValueError("Must perform clustering first")

        print(f"Optimizing cut height (testing {n_steps} values)...")

        # Test different cut heights
        heights = np.linspace(0.1, 0.95, n_steps)
        scores = []
        n_clusters_list = []

        best_score = -1.0
        best_height = None
        best_n_clusters = None

        for height in heights:
            # Get cluster assignments at this height
            clusters = fcluster(self.linkage_matrix, height, criterion='distance')
            n_clusters = len(set(clusters))

            # Skip if outside valid range
            if n_clusters < min_clusters or n_clusters > max_clusters:
                scores.append(None)
                n_clusters_list.append(n_clusters)
                continue

            # Compute silhouette score
            try:
                score = silhouette_score(
                    self.distance_matrix,
                    clusters,
                    metric='precomputed'
                )
                scores.append(score)
                n_clusters_list.append(n_clusters)

                # Track best
                if score > best_score:
                    best_score = score
                    best_height = height
                    best_n_clusters = n_clusters

                if verbose and len(scores) % 10 == 0:
                    print(f"  Height {height:.3f}: {n_clusters} clusters, "
                          f"silhouette = {score:.4f}")

            except Exception as e:
                scores.append(None)
                n_clusters_list.append(n_clusters)

        # Store results
        self.silhouette_scores = {
            'heights': heights,
            'scores': scores,
            'n_clusters': n_clusters_list,
            'best_score': best_score,
            'best_height': best_height,
            'best_n_clusters': best_n_clusters
        }

        self.optimal_height = best_height

        print(f"\nOptimal configuration:")
        if best_height is not None:
            print(f"  Cut height: {best_height:.4f}")
            print(f"  Number of clusters: {best_n_clusters}")
            print(f"  Silhouette score: {best_score:.4f}")
        else:
            print(f"  WARNING: No valid configuration found in range!")
            print(f"  Using fallback: height=0.5")
            best_height = 0.5
            best_n_clusters = len(set(fcluster(self.linkage_matrix, best_height, criterion='distance')))
            self.optimal_height = best_height

        return best_height, best_n_clusters

    def extract_clusters(
        self,
        height: Optional[float] = None
    ) -> np.ndarray:
        """
        Extract cluster assignments at specified height.

        Args:
            height: Cut height (uses optimal if None)

        Returns:
            Cluster assignment array (N,)
        """
        if self.linkage_matrix is None:
            raise ValueError("Must perform clustering first")

        if height is None:
            if self.optimal_height is None:
                raise ValueError("Must optimize cut height or provide height")
            height = self.optimal_height

        print(f"Extracting clusters at height {height:.4f}...")

        self.clusters = fcluster(self.linkage_matrix, height, criterion='distance')

        n_clusters = len(set(self.clusters))
        cluster_sizes = [np.sum(self.clusters == i) for i in range(1, n_clusters + 1)]

        print(f"Extracted {n_clusters} clusters")
        print(f"  Cluster sizes: min={min(cluster_sizes)}, "
              f"max={max(cluster_sizes)}, mean={np.mean(cluster_sizes):.1f}")

        return self.clusters

    def get_cluster_members(self, cluster_id: int) -> List[Dict]:
        """
        Get all antonym pairs in a cluster.

        Args:
            cluster_id: Cluster ID (1-indexed from fcluster)

        Returns:
            List of antonym pair dicts
        """
        if self.clusters is None:
            raise ValueError("Must extract clusters first")

        indices = np.where(self.clusters == cluster_id)[0]
        return [self.antonym_pairs[i] for i in indices]

    def plot_dendrogram(
        self,
        max_display: int = 100,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (20, 10)
    ):
        """
        Plot hierarchical clustering dendrogram.

        Args:
            max_display: Maximum number of leaves to display (truncates if more)
            output_path: Optional path to save figure
            figsize: Figure size (width, height)
        """
        if self.linkage_matrix is None:
            raise ValueError("Must perform clustering first")

        print("Plotting dendrogram...")

        fig, ax = plt.subplots(figsize=figsize)

        # Plot dendrogram
        if self.n_pairs > max_display:
            # Truncate for readability
            dendrogram(
                self.linkage_matrix,
                truncate_mode='lastp',
                p=max_display,
                ax=ax
            )
            title = f"Hierarchical Clustering Dendrogram (truncated to {max_display} clusters)"
        else:
            dendrogram(self.linkage_matrix, ax=ax)
            title = "Hierarchical Clustering Dendrogram"

        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Cluster Index or (Cluster Size)", fontsize=12)
        ax.set_ylabel("Distance", fontsize=12)

        # Add optimal cut line if available
        if self.optimal_height is not None:
            ax.axhline(
                y=self.optimal_height,
                color='r',
                linestyle='--',
                linewidth=2,
                label=f'Optimal cut (height={self.optimal_height:.3f})'
            )
            ax.legend()

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Dendrogram saved to {output_path}")

        return fig, ax

    def plot_silhouette_optimization(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Plot silhouette score optimization curve.

        Args:
            output_path: Optional path to save figure
            figsize: Figure size
        """
        if self.silhouette_scores is None:
            raise ValueError("Must optimize cut height first")

        print("Plotting silhouette optimization curve...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        heights = self.silhouette_scores['heights']
        scores = self.silhouette_scores['scores']
        n_clusters = self.silhouette_scores['n_clusters']

        # Filter out None scores
        valid = [(h, s, n) for h, s, n in zip(heights, scores, n_clusters) if s is not None]
        if valid:
            heights_valid, scores_valid, n_clusters_valid = zip(*valid)

            # Plot 1: Silhouette vs Height
            ax1.plot(heights_valid, scores_valid, 'b-', linewidth=2)
            ax1.axvline(
                x=self.silhouette_scores['best_height'],
                color='r',
                linestyle='--',
                label=f"Optimal (height={self.silhouette_scores['best_height']:.3f})"
            )
            ax1.set_xlabel("Cut Height", fontsize=12)
            ax1.set_ylabel("Silhouette Score", fontsize=12)
            ax1.set_title("Silhouette Score vs Cut Height", fontsize=14)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Silhouette vs Number of Clusters
            ax2.plot(n_clusters_valid, scores_valid, 'g-', linewidth=2)
            ax2.axvline(
                x=self.silhouette_scores['best_n_clusters'],
                color='r',
                linestyle='--',
                label=f"Optimal ({self.silhouette_scores['best_n_clusters']} clusters)"
            )
            ax2.set_xlabel("Number of Clusters", fontsize=12)
            ax2.set_ylabel("Silhouette Score", fontsize=12)
            ax2.set_title("Silhouette Score vs Cluster Count", fontsize=14)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Silhouette plot saved to {output_path}")

        return fig, (ax1, ax2)

    def get_cluster_statistics(self) -> Dict:
        """
        Get comprehensive clustering statistics.

        Returns:
            Dict with clustering metrics
        """
        if self.clusters is None:
            raise ValueError("Must extract clusters first")

        unique_clusters = set(self.clusters)
        n_clusters = len(unique_clusters)
        cluster_sizes = [np.sum(self.clusters == i) for i in unique_clusters]

        # Singletons
        singletons = [i for i, size in zip(unique_clusters, cluster_sizes) if size == 1]
        n_singletons = len(singletons)

        # Large clusters
        large_clusters = [(i, size) for i, size in zip(unique_clusters, cluster_sizes) if size >= 10]
        large_clusters.sort(key=lambda x: x[1], reverse=True)

        stats = {
            'total_antonym_pairs': self.n_pairs,
            'n_clusters': n_clusters,
            'n_singletons': n_singletons,
            'singleton_rate': n_singletons / n_clusters if n_clusters > 0 else 0.0,
            'min_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
            'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
            'mean_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0.0,
            'median_cluster_size': np.median(cluster_sizes) if cluster_sizes else 0.0,
            'large_clusters': large_clusters,  # (id, size) for clusters with >= 10 members
            'silhouette_score': self.silhouette_scores['best_score'] if self.silhouette_scores else None,
            'optimal_height': self.optimal_height
        }

        return stats

    def run_full_pipeline(
        self,
        min_clusters: int = 10,
        max_clusters: int = 200,
        verbose: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Run complete clustering pipeline.

        Args:
            min_clusters: Minimum number of clusters
            max_clusters: Maximum number of clusters
            verbose: Print progress

        Returns:
            (cluster_assignments, statistics_dict)
        """
        print("=" * 80)
        print("ANTONYM CLUSTERING PIPELINE")
        print("=" * 80)

        # Step 1: Build similarity matrix
        self.build_similarity_matrix()

        # Step 2: Hierarchical clustering
        self.perform_hierarchical_clustering()

        # Step 3: Optimize cut height
        self.optimize_cut_height(
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            verbose=verbose
        )

        # Step 4: Extract clusters
        self.extract_clusters()

        # Step 5: Get statistics
        stats = self.get_cluster_statistics()

        print("\n" + "=" * 80)
        print("CLUSTERING COMPLETE")
        print("=" * 80)
        print(f"Total antonym pairs: {stats['total_antonym_pairs']}")
        print(f"Clusters discovered: {stats['n_clusters']}")
        print(f"Singletons: {stats['n_singletons']} ({stats['singleton_rate']:.1%})")
        print(f"Cluster size range: {stats['min_cluster_size']} - {stats['max_cluster_size']}")
        print(f"Mean cluster size: {stats['mean_cluster_size']:.1f}")
        print(f"Silhouette score: {stats['silhouette_score']:.4f}")
        print("\nTop 10 largest clusters:")
        for i, (cluster_id, size) in enumerate(stats['large_clusters'][:10], 1):
            print(f"  {i}. Cluster {cluster_id}: {size} pairs")

        return self.clusters, stats
