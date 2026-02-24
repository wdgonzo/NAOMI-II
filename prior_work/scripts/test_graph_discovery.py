"""Test graph-based axis discovery."""

import json
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / 'src' / 'embeddings'
sys.path.insert(0, str(src_path))

from graph_based_axis_discovery import discover_axes_with_expansion

# Load antonym pairs
pairs_path = Path(__file__).parent.parent / 'data' / 'full_wordnet' / 'antonym_pairs.json'
with open(pairs_path, 'r') as f:
    all_pairs = json.load(f)

print("=" * 100)
print("TEST: Graph-Based Axis Discovery")
print("=" * 100)
print()

# Test with fundamental pairs
fundamental_keywords = ['good', 'bad', 'evil', 'hot', 'cold', 'warm', 'cool', 'big', 'small', 'large', 'little', 'fast', 'slow', 'strong', 'weak']
fundamental_pairs = [
    p for p in all_pairs
    if any(kw in [p['word1'].lower(), p['word2'].lower()] for kw in fundamental_keywords)
]

print(f"Testing with {len(fundamental_pairs)} fundamental pairs")
print()

axes = discover_axes_with_expansion(
    fundamental_pairs,
    min_shared_antonyms=1,
    min_axis_pairs=1,
    expand_threshold=0.5,
    verbose=True
)

print()
print("=" * 100)
print(f"DISCOVERED {len(axes)} AXES:")
print("=" * 100)

for i, axis in enumerate(axes, 1):
    print(f"\n{i}. {' / '.join(axis['pole_names'])} ({axis['n_poles']} poles, {axis['size']} words, {axis['n_pairs']} pairs)")
    for j, pole in enumerate(axis['poles']):
        print(f"   Pole {j+1} ({axis['pole_names'][j]}): {pole}")
    print(f"   Sample pairs: {[(p['word1'], p['word2']) for p in axis['representative_pairs'][:3]]}")

print()
print("=" * 100)
print("TEST WITH FULL DATASET")
print("=" * 100)
print()

axes_full = discover_axes_with_expansion(
    all_pairs,
    min_shared_antonyms=2,
    min_axis_pairs=3,
    expand_threshold=0.6,
    verbose=True
)

print()
print(f"DISCOVERED {len(axes_full)} AXES FROM FULL DATASET")
print()
print("Top 20 axes:")
for i, axis in enumerate(axes_full[:20], 1):
    print(f"  {i}. {' / '.join(axis['pole_names'])}: {axis['size']} words, {axis['n_pairs']} pairs, {axis['n_poles']} poles")
