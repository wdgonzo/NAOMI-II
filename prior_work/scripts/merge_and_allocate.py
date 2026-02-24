"""
Merge axes from multiple sources and allocate dimensions.

This script:
1. Loads axes from antonym clustering and adjective clustering
2. Merges overlapping axes
3. Allocates contiguous dimension ranges to each axis
4. Generates comprehensive metadata files:
   - dimension_metadata.json (axis definitions with geometry)
   - word_axis_assignments.json (word→axis mappings)
   - dimension_names.json (ordered dimension labels)

Usage:
    python scripts/merge_and_allocate.py \
        --antonym-axes data/discovered_axes/axes.json \
        --adjective-axes data/adjective_axes/adjective_axes.json \
        --output-dir data/final_axes \
        --dim-offset 51

Author: NAOMI-II Development Team
Date: 2025-12-01
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from embeddings.dimension_allocator import (
    allocate_and_generate_metadata,
    merge_axes
)


def load_axes(filepath: str) -> list:
    """Load axes from JSON file."""
    print(f"Loading axes from {filepath}...")
    with open(filepath, 'r') as f:
        axes = json.load(f)

    # Handle both list and dict formats
    if isinstance(axes, dict) and 'axes' in axes:
        axes = axes['axes']

    print(f"  Loaded {len(axes)} axes")
    return axes


def main():
    parser = argparse.ArgumentParser(
        description="Merge axes from multiple sources and allocate dimensions"
    )
    parser.add_argument(
        '--antonym-axes',
        type=str,
        required=True,
        help='Path to antonym axes JSON file'
    )
    parser.add_argument(
        '--adjective-axes',
        type=str,
        default=None,
        help='Path to adjective axes JSON file (optional)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/final_axes',
        help='Output directory for merged axes and metadata'
    )
    parser.add_argument(
        '--dim-offset',
        type=int,
        default=51,
        help='Starting dimension index (after anchor dimensions)'
    )
    parser.add_argument(
        '--anchor-dims',
        type=int,
        default=51,
        help='Number of anchor dimensions'
    )
    parser.add_argument(
        '--overlap-threshold',
        type=float,
        default=0.5,
        help='Minimum word overlap to merge axes (0-1)'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 100)
    print("MERGE AXES AND ALLOCATE DIMENSIONS")
    print("=" * 100)
    print(f"Dim offset: {args.dim_offset}")
    print(f"Anchor dims: {args.anchor_dims}")
    print(f"Overlap threshold: {args.overlap_threshold}")
    print()

    # Load antonym axes
    antonym_axes = load_axes(args.antonym_axes)

    # Load adjective axes (if provided)
    if args.adjective_axes and os.path.exists(args.adjective_axes):
        adjective_axes = load_axes(args.adjective_axes)
    else:
        adjective_axes = []
        print("No adjective axes provided, using only antonym axes")

    print()

    # Merge axes
    if adjective_axes:
        print(f"Merging {len(antonym_axes)} antonym axes with {len(adjective_axes)} adjective axes...")
        merged_axes = merge_axes(
            antonym_axes,
            adjective_axes,
            overlap_threshold=args.overlap_threshold
        )
        print(f"  Merged to {len(merged_axes)} axes")
    else:
        merged_axes = antonym_axes
        print(f"Using {len(merged_axes)} antonym axes (no merging)")

    print()

    # Allocate dimensions and generate metadata
    print("Allocating dimensions and generating metadata...")

    dimension_metadata, word_axis_assignments, dimension_names = allocate_and_generate_metadata(
        merged_axes,
        dim_offset=args.dim_offset,
        anchor_dims=args.anchor_dims
    )

    print(f"  Total dimensions: {dimension_metadata['metadata']['total_dimensions']}")
    print(f"  Learned dimensions: {dimension_metadata['metadata']['learned_dimensions']}")
    print(f"  Total axes: {dimension_metadata['metadata']['total_axes']}")

    print()

    # Save dimension metadata
    metadata_path = os.path.join(args.output_dir, 'dimension_metadata.json')
    print(f"Saving dimension metadata to {metadata_path}...")

    with open(metadata_path, 'w') as f:
        json.dump(dimension_metadata, f, indent=2)

    print("✓ Saved dimension_metadata.json")

    # Save word-axis assignments
    assignments_path = os.path.join(args.output_dir, 'word_axis_assignments.json')
    print(f"Saving word-axis assignments to {assignments_path}...")

    with open(assignments_path, 'w') as f:
        json.dump(word_axis_assignments, f, indent=2)

    print("✓ Saved word_axis_assignments.json")

    # Save dimension names
    names_path = os.path.join(args.output_dir, 'dimension_names.json')
    print(f"Saving dimension names to {names_path}...")

    with open(names_path, 'w') as f:
        json.dump(dimension_names, f, indent=2)

    print("✓ Saved dimension_names.json")

    print()

    # Print summary
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Total axes: {dimension_metadata['metadata']['total_axes']}")
    print(f"Total dimensions: {dimension_metadata['metadata']['total_dimensions']}")
    print(f"  - Anchor dimensions: {dimension_metadata['metadata']['anchor_dimensions']}")
    print(f"  - Learned dimensions: {dimension_metadata['metadata']['learned_dimensions']}")
    print()

    # Show axis type breakdown
    axis_types = {}
    for axis in dimension_metadata['axes']:
        axis_type = axis['type']
        axis_types[axis_type] = axis_types.get(axis_type, 0) + 1

    print("Axis type breakdown:")
    for axis_type, count in sorted(axis_types.items()):
        print(f"  - {axis_type}: {count}")

    print()

    # Show top 10 axes by size
    print("Top 10 axes by size:")
    for i, axis in enumerate(dimension_metadata['axes'][:10], 1):
        dims = axis['dimensions']
        print(f"  {i}. {axis['name']} ({axis['type']})")
        print(f"     Size: {axis['metrics']['size']} pairs/words")
        print(f"     Dimensions: [{dims['start']}:{dims['end']}] ({dims['count']} dims)")
        print(f"     Labels: {', '.join(dims['labels'])}")
        print(f"     Poles: {', '.join(axis['poles']['names'])}")

    print()
    print(f"✓ Merge and allocation complete!")
    print(f"✓ Output saved to {args.output_dir}")


if __name__ == '__main__':
    main()
