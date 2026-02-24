"""
Compute regular simplex vertex coordinates for n-ary semantic axes.

This module provides functions to compute the vertex coordinates of regular
simplices in (n-1)-dimensional space, used to represent n-pole semantic axes
with equidistant pole positions.

Author: NAOMI-II Development Team
Date: 2025-12-01
"""

from typing import Dict, List
import numpy as np
import math


def compute_simplex_vertices(n_poles: int) -> np.ndarray:
    """
    Compute vertices of regular simplex in (n-1)D space.

    The simplex is centered at the origin and normalized such that all vertices
    lie on the unit sphere, ensuring all poles are equidistant.

    Args:
        n_poles: Number of poles (vertices)

    Returns:
        Array of shape (n_poles, n_poles-1) with vertex coordinates
    """
    if n_poles < 2:
        raise ValueError(f"Need at least 2 poles, got {n_poles}")

    if n_poles == 2:
        # Binary case: simple 1D opposition
        return np.array([[1.0], [-1.0]])

    # For n > 2, use standard simplex construction
    # Place vertices on unit sphere in (n-1)D space

    n_dims = n_poles - 1
    vertices = np.zeros((n_poles, n_dims))

    # Standard simplex construction:
    # First vertex at [1, 0, 0, ..., 0]
    vertices[0, 0] = 1.0

    # Remaining vertices constructed recursively
    for i in range(1, n_poles):
        # Coordinate perpendicular to previous vertices
        if i < n_dims:
            # Fill in coordinates for new dimension
            # Compute coordinate to maintain equal distances
            sum_sq = sum(vertices[i, :i]**2)
            if i < n_dims:
                vertices[i, i] = math.sqrt(1.0 - sum_sq) if sum_sq < 1.0 else 0.0

        # Set first i-1 coordinates to maintain equidistance
        for j in range(i):
            # Use recurrence relation for regular simplex
            vertices[i, j] = -1.0 / n_poles

    # Normalize to unit sphere
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

    # Center at origin (subtract centroid)
    centroid = np.mean(vertices, axis=0)
    vertices = vertices - centroid

    # Re-normalize
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

    return vertices


def compute_pole_geometry(
    pole_names: List[str],
    n_poles: int
) -> Dict[str, List[float]]:
    """
    Compute pole geometry mapping pole names to coordinates.

    Args:
        pole_names: List of pole names (e.g., ["solid", "liquid", "gas"])
        n_poles: Number of poles (should match len(pole_names))

    Returns:
        Dict mapping pole_name → coordinates (list of floats)
    """
    if len(pole_names) != n_poles:
        raise ValueError(
            f"Number of pole names ({len(pole_names)}) must match n_poles ({n_poles})"
        )

    # Compute simplex vertices
    vertices = compute_simplex_vertices(n_poles)

    # Map names to coordinates
    geometry = {}
    for i, name in enumerate(pole_names):
        geometry[name] = vertices[i].tolist()

    return geometry


def verify_simplex_properties(vertices: np.ndarray, tolerance: float = 1e-6) -> bool:
    """
    Verify that vertices form a valid regular simplex.

    Properties checked:
    1. All vertices on unit sphere
    2. All pairwise distances equal
    3. Centered at origin

    Args:
        vertices: Array of shape (n_poles, n_dims)
        tolerance: Numerical tolerance for checks

    Returns:
        True if all properties satisfied
    """
    n_poles, n_dims = vertices.shape

    # Check 1: All vertices on unit sphere
    norms = np.linalg.norm(vertices, axis=1)
    if not np.allclose(norms, 1.0, atol=tolerance):
        return False

    # Check 2: All pairwise distances equal
    distances = []
    for i in range(n_poles):
        for j in range(i + 1, n_poles):
            dist = np.linalg.norm(vertices[i] - vertices[j])
            distances.append(dist)

    if not np.allclose(distances, distances[0], atol=tolerance):
        return False

    # Check 3: Centered at origin
    centroid = np.mean(vertices, axis=0)
    if not np.allclose(centroid, 0.0, atol=tolerance):
        return False

    return True


# Precomputed small simplices for common cases
_SIMPLEX_CACHE = {
    2: np.array([[1.0], [-1.0]]),
    3: np.array([[1.0, 0.0], [-0.5, 0.866], [-0.5, -0.866]]),
    4: np.array([
        [1.0, 0.0, 0.0],
        [-0.333, 0.943, 0.0],
        [-0.333, -0.471, 0.816],
        [-0.333, -0.471, -0.816]
    ])
}


def get_simplex_vertices_cached(n_poles: int) -> np.ndarray:
    """
    Get simplex vertices with caching for common cases.

    Args:
        n_poles: Number of poles

    Returns:
        Array of shape (n_poles, n_poles-1) with vertex coordinates
    """
    if n_poles in _SIMPLEX_CACHE:
        return _SIMPLEX_CACHE[n_poles].copy()
    else:
        return compute_simplex_vertices(n_poles)
