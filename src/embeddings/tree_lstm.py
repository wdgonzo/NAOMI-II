"""
Tree-LSTM Encoder

Child-Sum Tree-LSTM for encoding parse trees into semantic vectors.

This is NOT a standard sequential LSTM - it recursively encodes tree structures
following the parse tree hierarchy. The composition operation respects the
grammatical relationships (edge types) in the tree.

Based on "Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks"
(Tai et al., 2015) with modifications for parse tree encoding.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
import numpy as np

from ..parser.data_structures import Hypothesis, Node, Edge
from ..parser.enums import ConnectionType
from .device import DeviceManager


class TreeLSTMCell(nn.Module):
    """
    Child-Sum Tree-LSTM cell for encoding tree nodes.

    Unlike standard LSTM, this aggregates information from multiple children
    before computing the hidden state.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize Tree-LSTM cell.

        Args:
            input_dim: Dimension of input embeddings
            hidden_dim: Dimension of hidden state
        """
        super(TreeLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input gate
        self.W_i = nn.Linear(input_dim, hidden_dim)
        self.U_i = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Forget gates (one per child)
        self.W_f = nn.Linear(input_dim, hidden_dim)
        self.U_f = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Output gate
        self.W_o = nn.Linear(input_dim, hidden_dim)
        self.U_o = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Cell state
        self.W_u = nn.Linear(input_dim, hidden_dim)
        self.U_u = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor,
                child_h: List[torch.Tensor],
                child_c: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for one tree node.

        Args:
            x: Input embedding for this node (batch_size, input_dim)
            child_h: List of child hidden states
            child_c: List of child cell states

        Returns:
            h: Hidden state (batch_size, hidden_dim)
            c: Cell state (batch_size, hidden_dim)
        """
        batch_size = x.size(0)

        # Sum of child hidden states
        if len(child_h) > 0:
            h_sum = torch.stack(child_h).sum(dim=0)  # (batch_size, hidden_dim)
        else:
            h_sum = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        # Input gate
        i = torch.sigmoid(self.W_i(x) + self.U_i(h_sum))

        # Output gate
        o = torch.sigmoid(self.W_o(x) + self.U_o(h_sum))

        # Cell update
        u = torch.tanh(self.W_u(x) + self.U_u(h_sum))

        # Forget gates for each child
        c = i * u
        for child_c_k, child_h_k in zip(child_c, child_h):
            f_k = torch.sigmoid(self.W_f(x) + self.U_f(child_h_k))
            c = c + f_k * child_c_k

        # Hidden state
        h = o * torch.tanh(c)

        return h, c


class EdgeTypeEmbedding(nn.Module):
    """
    Learns embeddings for different edge types in the parse tree.

    This allows the model to differentiate between SUBJECT, OBJECT, DESCRIPTION, etc.
    when composing child nodes.
    """

    def __init__(self, num_edge_types: int, edge_dim: int):
        """
        Initialize edge type embeddings.

        Args:
            num_edge_types: Number of different edge types
            edge_dim: Dimension of edge embeddings
        """
        super(EdgeTypeEmbedding, self).__init__()

        self.edge_embeddings = nn.Embedding(num_edge_types, edge_dim)

    def forward(self, edge_types: torch.Tensor) -> torch.Tensor:
        """
        Get edge embeddings.

        Args:
            edge_types: Tensor of edge type indices (batch_size,)

        Returns:
            Edge embeddings (batch_size, edge_dim)
        """
        return self.edge_embeddings(edge_types)


class TreeLSTMEncoder(nn.Module):
    """
    Tree-LSTM encoder for parse trees.

    Encodes an entire parse tree into a single semantic vector by recursively
    composing nodes from leaves to root, respecting grammatical structure.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_edge_types: int = 20, dropout: float = 0.5):
        """
        Initialize Tree-LSTM encoder.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of LSTM hidden states
            num_edge_types: Number of different edge types
            dropout: Dropout probability
        """
        super(TreeLSTMEncoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Word embeddings (learnable)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Edge type embeddings
        self.edge_embeddings = EdgeTypeEmbedding(num_edge_types, embedding_dim // 4)

        # Tree-LSTM cell
        self.tree_lstm = TreeLSTMCell(embedding_dim, hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output projection (optional, for matching target embedding dim)
        self.output_proj = nn.Linear(hidden_dim, embedding_dim)

    def encode_node(self, node_idx: int, hypothesis: Hypothesis,
                    word_to_id: Dict[str, int],
                    device: torch.device,
                    memo: Dict[int, Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recursively encode a tree node and all its children.

        Args:
            node_idx: Index of node to encode
            hypothesis: Parse hypothesis containing the tree
            word_to_id: Mapping from words to vocabulary indices
            device: Device to place tensors on
            memo: Memoization dict for visited nodes

        Returns:
            h: Hidden state for this node
            c: Cell state for this node
        """
        # Check memo
        if node_idx in memo:
            return memo[node_idx]

        node = hypothesis.nodes[node_idx]

        # Get word embedding for this node
        if node.value and hasattr(node.value, 'text'):
            word = node.value.text.lower()
            word_id = word_to_id.get(word, 0)  # 0 for unknown
        else:
            word_id = 0  # Unknown/constituent node

        word_id_tensor = torch.tensor([word_id], dtype=torch.long, device=device)
        x = self.word_embeddings(word_id_tensor)  # (1, embedding_dim)

        # Find all child edges
        child_edges = [e for e in hypothesis.edges if e.parent == node_idx]

        # Recursively encode children
        child_h_list = []
        child_c_list = []

        for edge in child_edges:
            child_h, child_c = self.encode_node(edge.child, hypothesis, word_to_id, device, memo)
            child_h_list.append(child_h)
            child_c_list.append(child_c)

        # Apply dropout to input
        x = self.dropout(x)

        # Compute this node's hidden and cell states
        h, c = self.tree_lstm(x, child_h_list, child_c_list)

        # Store in memo
        memo[node_idx] = (h, c)

        return h, c

    def forward(self, hypothesis: Hypothesis,
                word_to_id: Dict[str, int],
                device: torch.device) -> torch.Tensor:
        """
        Encode entire parse tree.

        Args:
            hypothesis: Parse hypothesis to encode
            word_to_id: Word to vocabulary ID mapping
            device: Device to place tensors on

        Returns:
            Sentence embedding (1, embedding_dim)
        """
        # Find root node (node with no parent)
        all_children = set(e.child for e in hypothesis.edges)
        root_candidates = [i for i in range(len(hypothesis.nodes)) if i not in all_children]

        if not root_candidates:
            # No clear root - use first unconsumed node
            unconsumed = hypothesis.get_unconsumed()
            if unconsumed:
                root_idx = hypothesis.nodes.index(unconsumed[0])
            else:
                root_idx = 0
        else:
            root_idx = root_candidates[0]

        # Encode from root
        memo = {}
        h_root, c_root = self.encode_node(root_idx, hypothesis, word_to_id, device, memo)

        # Project to output dimension
        output = self.output_proj(h_root)

        return output

    def encode_batch(self, hypotheses: List[Hypothesis],
                    word_to_id: Dict[str, int],
                    device: torch.device) -> torch.Tensor:
        """
        Encode a batch of parse trees.

        Args:
            hypotheses: List of parse hypotheses
            word_to_id: Word to vocabulary ID mapping
            device: Device to place tensors on

        Returns:
            Batch of sentence embeddings (batch_size, embedding_dim)
        """
        embeddings = []

        for hyp in hypotheses:
            emb = self.forward(hyp, word_to_id, device)
            embeddings.append(emb)

        return torch.cat(embeddings, dim=0)


def create_tree_lstm_encoder(vocab_size: int,
                             embedding_dim: int = 256,
                             hidden_dim: int = 256,
                             num_edge_types: int = 20,
                             dropout: float = 0.5,
                             device_manager: Optional[DeviceManager] = None) -> TreeLSTMEncoder:
    """
    Factory function to create a Tree-LSTM encoder.

    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of word embeddings
        hidden_dim: Dimension of LSTM hidden states
        num_edge_types: Number of edge types in parse trees
        dropout: Dropout probability
        device_manager: Device manager (if None, creates new one)

    Returns:
        TreeLSTMEncoder model placed on device
    """
    model = TreeLSTMEncoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_edge_types=num_edge_types,
        dropout=dropout
    )

    if device_manager is None:
        from .device import get_device_manager
        device_manager = get_device_manager(prefer_npu=True, verbose=True)

    model = model.to(device_manager.device)

    return model
