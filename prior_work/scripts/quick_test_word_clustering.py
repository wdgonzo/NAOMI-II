"""
Quick test of word-based clustering with just 100 pairs.
"""

import json
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / 'src' / 'embeddings'
sys.path.insert(0, str(src_path))

from word_clustering import cluster_words_and_detect_axes

# Load first 100 pairs
pairs_path = Path(__file__).parent.parent / 'data' / 'full_wordnet' / 'antonym_pairs.json'
with open(pairs_path, 'r') as f:
    all_pairs = json.load(f)

# Get fundamental pairs
fundamental_keywords = ['good', 'bad', 'evil', 'hot', 'cold', 'warm', 'big', 'small', 'large', 'fast', 'slow', 'strong', 'weak']
fundamental_pairs = [
    p for p in all_pairs
    if any(kw in [p['word1'].lower(), p['word2'].lower()] for kw in fundamental_keywords)
]

print(f"Found {len(fundamental_pairs)} fundamental pairs")
print("Sample pairs:")
for p in fundamental_pairs[:10]:
    print(f"  {p['word1']} / {p['word2']}")
print()

# Test clustering
print("Testing word-based clustering...")
word_clusters, axes = cluster_words_and_detect_axes(
    fundamental_pairs,
    n_clusters=10,
    min_cluster_size=2,
    min_antonym_edges=1,
    linkage_method='average',
    verbose=True
)

print()
print(f"Found {len(axes)} axes:")
for i, axis in enumerate(axes, 1):
    print(f"\n{i}. {' / '.join(axis['pole_names'])} ({axis['size']} words, {axis['n_pairs']} pairs)")
    for j, pole in enumerate(axis['poles']):
        print(f"   Pole {j+1}: {sorted(pole)}")
