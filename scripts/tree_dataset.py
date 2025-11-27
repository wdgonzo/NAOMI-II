"""
Tree Structure Dataset

Dataset for structure-aware training with Tree-LSTM.
Provides parse hypotheses along with distance constraints from knowledge graph.
"""

import pickle
import json
from pathlib import Path
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset

from src.parser.data_structures import Hypothesis


class TreeStructureDataset(Dataset):
    """
    Dataset that provides parse tree structures for training.

    This dataset combines:
    1. Parse hypotheses (tree structures)
    2. Distance constraints from knowledge graph

    Each sample contains:
    - sentence: Original text
    - hypothesis: Parsed tree structure
    - word_ids: List of word IDs for the sentence
    """

    def __init__(self, parsed_sentences: List[Tuple[str, Hypothesis]],
                 word_to_id: Dict[str, int]):
        """
        Initialize dataset.

        Args:
            parsed_sentences: List of (sentence, hypothesis) tuples
            word_to_id: Vocabulary mapping
        """
        self.parsed_sentences = parsed_sentences
        self.word_to_id = word_to_id

    def __len__(self):
        return len(self.parsed_sentences)

    def __getitem__(self, idx):
        sentence, hypothesis = self.parsed_sentences[idx]
        return {
            'sentence': sentence,
            'hypothesis': hypothesis,
            'idx': idx
        }


class PairwiseDistanceDataset(Dataset):
    """
    Dataset for pairwise distance constraints (backward compatibility).

    This provides word-pair distance constraints from the knowledge graph.
    Used for the distance loss component of training.
    """

    def __init__(self, edges: List[Tuple], vocab_size: int,
                 relation_to_distance: Dict):
        """
        Initialize pairwise distance dataset.

        Args:
            edges: List of (source_id, relation_id, target_id, confidence) tuples
            vocab_size: Size of vocabulary
            relation_to_distance: Map from relation type to target distance
        """
        self.edges = edges
        self.vocab_size = vocab_size
        self.samples = []

        for source_id, relation_id, target_id, confidence in edges:
            if relation_id in relation_to_distance:
                target_dist = relation_to_distance[relation_id]
                self.samples.append((source_id, target_id, target_dist, confidence))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        source_id, target_id, target_dist, confidence = self.samples[idx]
        return {
            'source_id': source_id,
            'target_id': target_id,
            'target_distance': target_dist,
            'confidence': confidence
        }


def load_training_data(data_dir: Path) -> Tuple[TreeStructureDataset,
                                                  PairwiseDistanceDataset,
                                                  Dict[str, int],
                                                  Dict[str, int]]:
    """
    Load all training data from directory.

    Args:
        data_dir: Directory containing training data files

    Returns:
        (tree_dataset, distance_dataset, word_to_id, id_to_word)
    """
    # Load vocabulary
    with open(data_dir / "vocabulary.json", 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    word_to_id = vocab_data['word_to_id']
    id_to_word = vocab_data['id_to_word']

    # Load parsed sentences
    with open(data_dir / "parsed_sentences.pkl", 'rb') as f:
        parsed_sentences = pickle.load(f)

    # Load distance constraints
    with open(data_dir / "training_edges.pkl", 'rb') as f:
        edges = pickle.load(f)

    # Create relation distance map
    relation_to_distance = {
        0: 0.1,   # synonym
        1: 0.3,   # hypernym
        2: 0.3,   # hyponym
        3: 0.9,   # antonym
        4: 0.4,   # parse relations
        5: 0.4,
        6: 0.3,
        7: 0.3,
        8: 0.4,
        9: 0.5,
        10: 0.5,
    }

    # Create datasets
    tree_dataset = TreeStructureDataset(parsed_sentences, word_to_id)
    distance_dataset = PairwiseDistanceDataset(
        edges,
        vocab_data['vocab_size'],
        relation_to_distance
    )

    return tree_dataset, distance_dataset, word_to_id, id_to_word


def collate_tree_batch(batch):
    """
    Collate function for tree dataset.

    Since tree structures have variable size, we don't stack them.
    Instead, we return a list of hypotheses.

    Args:
        batch: List of dataset items

    Returns:
        Dict with batch data
    """
    return {
        'sentences': [item['sentence'] for item in batch],
        'hypotheses': [item['hypothesis'] for item in batch],
        'indices': [item['idx'] for item in batch]
    }


def collate_distance_batch(batch):
    """
    Collate function for distance dataset.

    Args:
        batch: List of dataset items

    Returns:
        Dict with tensors
    """
    return {
        'source_id': torch.tensor([item['source_id'] for item in batch],
                                   dtype=torch.long),
        'target_id': torch.tensor([item['target_id'] for item in batch],
                                   dtype=torch.long),
        'target_distance': torch.tensor([item['target_distance'] for item in batch],
                                        dtype=torch.float32),
        'confidence': torch.tensor([item['confidence'] for item in batch],
                                   dtype=torch.float32)
    }
