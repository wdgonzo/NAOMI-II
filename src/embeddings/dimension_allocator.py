"""
Dimension allocation and metadata generation for discovered semantic axes.

This module allocates contiguous dimension ranges to discovered axes and generates
comprehensive metadata files:
- dimension_metadata.json: Axis definitions with geometry
- word_axis_assignments.json: Word→axis mappings
- dimension_names.json: Ordered list of dimension labels

Author: NAOMI-II Development Team
Date: 2025-12-01
"""

from typing import Dict, List, Set, Tuple
from collections import defaultdict
from datetime import datetime
import numpy as np

try:
    from .simplex_geometry import compute_pole_geometry, get_simplex_vertices_cached
except ImportError:
    from simplex_geometry import compute_pole_geometry, get_simplex_vertices_cached


class DimensionAllocator:
    """Allocates dimensions to axes and generates metadata files."""

    def __init__(
        self,
        axes: List[Dict],
        dim_offset: int = 51,
        anchor_dims: int = 51
    ):
        """
        Initialize dimension allocator.

        Args:
            axes: List of axis dicts (from antonym/adjective clustering)
            dim_offset: Starting dimension index (after anchors)
            anchor_dims: Number of anchor dimensions
        """
        self.axes = axes
        self.dim_offset = dim_offset
        self.anchor_dims = anchor_dims

    def allocate_dimensions(self) -> Tuple[List[Dict], List[str], Dict]:
        """
        Allocate dimensions to all axes.

        Returns:
            (axes_with_dims, dimension_names, dimension_index)
        """
        # Sort axes by priority (size desc, coherence desc, source)
        sorted_axes = sorted(
            self.axes,
            key=lambda a: (-a.get('size', 0), -a.get('coherence_score', 0.0), a.get('source', ''))
        )

        current_dim = self.dim_offset
        dimension_names = []
        dimension_index = {}

        for axis in sorted_axes:
            n_poles = len(axis['poles'])
            n_dims = n_poles - 1 if n_poles > 1 else 1

            # Allocate dimension range
            axis['dimensions'] = {
                'start': current_dim,
                'end': current_dim + n_dims,
                'count': n_dims,
                'labels': []
            }

            # Generate dimension labels
            axis_name = axis['name']
            if n_dims == 1:
                dim_label = axis_name
                dimension_names.append(dim_label)
                axis['dimensions']['labels'].append(dim_label)
                dimension_index[axis_name] = [current_dim]
            else:
                dim_labels = []
                dim_indices = []
                for i in range(n_dims):
                    dim_label = f"{axis_name}_{i}"
                    dimension_names.append(dim_label)
                    dim_labels.append(dim_label)
                    dim_indices.append(current_dim + i)
                axis['dimensions']['labels'] = dim_labels
                dimension_index[axis_name] = dim_indices

            current_dim += n_dims

        total_dims = current_dim

        return (sorted_axes, dimension_names, dimension_index, total_dims)

    def generate_dimension_metadata(
        self,
        axes_with_dims: List[Dict],
        dimension_index: Dict,
        total_dims: int
    ) -> Dict:
        """
        Generate dimension_metadata.json structure.

        Args:
            axes_with_dims: Axes with allocated dimensions
            dimension_index: Quick lookup dict
            total_dims: Total number of dimensions

        Returns:
            Complete metadata dict
        """
        metadata = {
            'metadata': {
                'total_axes': len(axes_with_dims),
                'total_dimensions': total_dims,
                'anchor_dimensions': self.anchor_dims,
                'learned_dimensions': total_dims - self.anchor_dims,
                'generation_date': datetime.now().strftime('%Y-%m-%d'),
                'phase': 'phase_1_bootstrap'
            },
            'axes': [],
            'dimension_index': dimension_index
        }

        for i, axis in enumerate(axes_with_dims):
            # Determine axis type
            n_poles = len(axis['poles'])
            if n_poles == 2:
                axis_type = 'binary'
            elif n_poles == 3:
                axis_type = 'ternary'
            elif n_poles == 4:
                axis_type = 'quaternary'
            else:
                axis_type = f'{n_poles}-ary'

            # Compute pole geometry
            pole_names = axis['pole_names']
            pole_geometry = compute_pole_geometry(pole_names, n_poles)

            # Build axis entry
            axis_entry = {
                'axis_id': i,
                'name': axis['name'],
                'type': axis_type,
                'source': axis.get('source', 'unknown'),
                'dimensions': axis['dimensions'],
                'poles': {
                    'count': n_poles,
                    'names': pole_names,
                    'geometry': pole_geometry
                },
                'metrics': {
                    'size': axis.get('size', 0),
                    'coherence': float(axis.get('coherence_score', 0.0)),
                    'separation': float(axis.get('separation_score', 0.0))
                },
                'representative_pairs': axis.get('representative_pairs', [])[:5]
            }

            metadata['axes'].append(axis_entry)

        return metadata

    def generate_word_axis_assignments(self, axes_with_dims: List[Dict]) -> Dict:
        """
        Generate word_axis_assignments.json structure.

        Args:
            axes_with_dims: Axes with allocated dimensions

        Returns:
            Word assignment dict
        """
        assignments = {
            'metadata': {
                'total_words': 0,
                'total_axes': len(axes_with_dims),
                'generation_date': datetime.now().strftime('%Y-%m-%d')
            },
            'assignments': [],
            'word_index': defaultdict(list)
        }

        total_words = 0

        for axis in axes_with_dims:
            axis_id = axis.get('axis_id', 0)
            axis_name = axis['name']
            pole_names = axis['pole_names']

            # Build pole assignments
            pole_assignments = {}
            for pole_name, pole_words in zip(pole_names, axis['poles']):
                pole_assignments[pole_name] = sorted(list(pole_words))
                total_words += len(pole_words)

                # Update word index
                for word in pole_words:
                    assignments['word_index'][word].append({
                        'axis_id': axis_id,
                        'axis_name': axis_name,
                        'pole': pole_name
                    })

            # Add assignment entry
            assignment_entry = {
                'axis_id': axis_id,
                'axis_name': axis_name,
                'pole_assignments': pole_assignments,
                'total_words': sum(len(words) for words in pole_assignments.values())
            }

            assignments['assignments'].append(assignment_entry)

        # Convert defaultdict to regular dict
        assignments['word_index'] = dict(assignments['word_index'])
        assignments['metadata']['total_words'] = total_words

        return assignments

    def generate_dimension_names(self, dimension_labels: List[str]) -> List[str]:
        """
        Generate dimension_names.json structure.

        Args:
            dimension_labels: List of learned dimension labels

        Returns:
            Complete dimension names list (anchors + learned)
        """
        # Start with anchor dimensions
        anchor_names = [f"anchor_{i}" for i in range(self.anchor_dims)]

        # Append learned dimension labels
        full_names = anchor_names + dimension_labels

        return full_names


def allocate_and_generate_metadata(
    axes: List[Dict],
    dim_offset: int = 51,
    anchor_dims: int = 51
) -> Tuple[Dict, Dict, List[str]]:
    """
    Convenience function to allocate dimensions and generate all metadata.

    Args:
        axes: List of axis dicts
        dim_offset: Starting dimension index
        anchor_dims: Number of anchor dimensions

    Returns:
        (dimension_metadata, word_axis_assignments, dimension_names)
    """
    allocator = DimensionAllocator(axes, dim_offset, anchor_dims)

    # Allocate dimensions
    axes_with_dims, dimension_labels, dimension_index, total_dims = allocator.allocate_dimensions()

    # Generate metadata files
    dimension_metadata = allocator.generate_dimension_metadata(
        axes_with_dims,
        dimension_index,
        total_dims
    )

    word_axis_assignments = allocator.generate_word_axis_assignments(axes_with_dims)

    dimension_names = allocator.generate_dimension_names(dimension_labels)

    return (dimension_metadata, word_axis_assignments, dimension_names)


def merge_axes(
    antonym_axes: List[Dict],
    adjective_axes: List[Dict],
    overlap_threshold: float = 0.5
) -> List[Dict]:
    """
    Merge axes from multiple sources, resolving overlaps.

    Args:
        antonym_axes: Axes from antonym pair clustering
        adjective_axes: Axes from adjective clustering
        overlap_threshold: Minimum word overlap to merge (0-1)

    Returns:
        Merged axis list
    """
    all_axes = antonym_axes + adjective_axes
    merged = []
    used = set()

    for i, axis1 in enumerate(all_axes):
        if i in used:
            continue

        # Check for overlaps with remaining axes
        best_match = None
        best_overlap = 0.0

        for j, axis2 in enumerate(all_axes[i+1:], start=i+1):
            if j in used:
                continue

            # Compute word overlap
            words1 = set()
            for pole in axis1['poles']:
                words1.update(pole)

            words2 = set()
            for pole in axis2['poles']:
                words2.update(pole)

            if len(words1) == 0 or len(words2) == 0:
                continue

            overlap = len(words1 & words2) / min(len(words1), len(words2))

            if overlap > overlap_threshold and overlap > best_overlap:
                best_match = j
                best_overlap = overlap

        if best_match is not None:
            # Merge axes
            axis2 = all_axes[best_match]

            # Keep the one with higher coherence
            if axis1.get('coherence_score', 0.0) >= axis2.get('coherence_score', 0.0):
                merged.append(axis1)
            else:
                merged.append(axis2)

            used.add(best_match)
        else:
            # No overlap, keep as-is
            merged.append(axis1)

        used.add(i)

    return merged
