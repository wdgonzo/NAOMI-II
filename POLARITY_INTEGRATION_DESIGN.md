# Polarity Loss Integration Design

## Problem

Current two-stage approach doesn't work:
- Stage 1: Distance training → antonyms are far apart but signs are random
- Stage 2: Can't discover polarity dimensions because they don't exist yet!

**Root cause:** Polarity structure must be learned DURING training, not discovered after.

## Solution: Integrated Polarity Loss

Similar to how node-splitting/merging happens dynamically during RNN training, we'll integrate polarity loss directly into the training loop.

### Architecture

```python
total_loss = α * distance_loss + β * polarity_loss
```

Where:
- **distance_loss**: Existing fuzzy distance constraints (all relations)
- **polarity_loss**: NEW - enforces opposite signs for antonyms

### Polarity Loss Function

```python
def polarity_loss(emb1, emb2, polarity_weight=1.0):
    """
    Encourage antonym pairs to have opposite signs on ALL dimensions.

    The network will naturally concentrate opposite-sign structure
    on a few key dimensions (emergent polarity dimensions).
    """
    loss = 0.0

    for dim in range(embedding_dim):
        val1 = emb1[dim]
        val2 = emb2[dim]

        # Sign agreement score
        sign_product = sign(val1) * sign(val2)

        if sign_product > 0:
            # Same sign - BAD for antonyms
            # Penalty proportional to magnitudes
            loss += (|val1| + |val2|) * polarity_weight
        elif sign_product < 0:
            # Opposite signs - GOOD for antonyms
            # Reward (negative loss) proportional to magnitudes
            loss -= (|val1| + |val2|) * polarity_weight * 0.5
        else:
            # One is zero - slight penalty
            loss += 0.1 * polarity_weight

    return loss
```

### Key Insights

1. **Emergent Structure**: By applying polarity loss to ALL dimensions, the network naturally learns to concentrate opposite-sign structure on a FEW key dimensions (polarity dimensions emerge!)

2. **Magnitude Matters**: The loss is proportional to magnitude, so dimensions with stronger values contribute more to polarity structure.

3. **Natural Selection**: Dimensions that don't help with antonym distinction will have random signs. Dimensions that DO help will develop strong opposite-sign patterns.

4. **Compatibility**: Works alongside distance loss - distance keeps antonyms far apart, polarity organizes their signs.

## Implementation Plan

### Step 1: Add Polarity Loss to PyTorch Training

Modify `scripts/train_pytorch_polarity.py` to:

1. Extract antonym pairs at training start
2. Create `PolarityLoss` module
3. Compute polarity loss on every batch containing antonym pairs
4. Add to total loss with tunable weight

### Step 2: Single-Stage Training (No Discovery Needed!)

```
Before:
  Stage 1: Distance training (no polarity) → Stage 2: Discover + fine-tune

After:
  Single Stage: Distance + Polarity training together
```

### Step 3: Post-Training Analysis

After training, analyze which dimensions emerged as polarity dimensions:
```python
def identify_polarity_dimensions(model, antonym_pairs, threshold=0.7):
    """Find dimensions where antonyms consistently have opposite signs."""
    polarity_dims = []

    for dim in range(embedding_dim):
        opposite_count = 0
        for word1_id, word2_id in antonym_pairs:
            emb1 = model.embeddings[word1_id]
            emb2 = model.embeddings[word2_id]

            if sign(emb1[dim]) * sign(emb2[dim]) < 0:
                opposite_count += 1

        consistency = opposite_count / len(antonym_pairs)
        if consistency >= threshold:
            polarity_dims.append(dim)

    return polarity_dims
```

## Expected Results

After integrated training:

1. **Spatial Separation**: Antonyms are far apart (from distance loss)
2. **Sign Structure**: Antonyms have opposite signs on 5-15 key dimensions
3. **Emergent Polarity**: Polarity dimensions emerge naturally without explicit selection
4. **Compositional Semantics**: NOT(good) ≈ bad works through sign flipping on polarity dims

## Code Changes Required

### 1. Create PolarityLoss Module (`scripts/train_pytorch_polarity.py`)

```python
class PolarityLoss(nn.Module):
    """Polarity loss encourages opposite signs for antonym pairs."""

    def __init__(self, polarity_weight=1.0):
        super().__init__()
        self.polarity_weight = polarity_weight

    def forward(self, embeddings: torch.Tensor, antonym_pairs: List[Tuple[int, int]]) -> torch.Tensor:
        """
        Args:
            embeddings: (vocab_size, embedding_dim)
            antonym_pairs: List of (id1, id2) antonym pairs
        """
        if not antonym_pairs:
            return torch.tensor(0.0)

        loss = 0.0
        for id1, id2 in antonym_pairs:
            emb1 = embeddings[id1]
            emb2 = embeddings[id2]

            # Sign product for each dimension
            sign_product = torch.sign(emb1) * torch.sign(emb2)

            # Penalty for same signs, reward for opposite signs
            same_sign_mask = sign_product > 0
            opposite_sign_mask = sign_product < 0

            # Loss = sum of magnitudes where signs agree (bad)
            #      - 0.5 * sum of magnitudes where signs disagree (good)
            loss += (torch.abs(emb1[same_sign_mask]).sum() +
                    torch.abs(emb2[same_sign_mask]).sum()) * self.polarity_weight

            loss -= (torch.abs(emb1[opposite_sign_mask]).sum() +
                    torch.abs(emb2[opposite_sign_mask]).sum()) * self.polarity_weight * 0.5

        return loss / len(antonym_pairs)
```

### 2. Modify Training Loop

```python
def train_integrated(model, distance_dataloader, antonym_pairs, device, epochs, lr,
                     distance_weight=1.0, polarity_weight=0.1):
    """Single-stage training with integrated polarity loss."""

    distance_loss_fn = DistanceLoss()
    polarity_loss_fn = PolarityLoss(polarity_weight)
    optimizer = torch.optim.Adam([model.embeddings], lr=lr)

    for epoch in range(epochs):
        epoch_dist_loss = 0.0
        epoch_pol_loss = 0.0

        for batch in distance_dataloader:
            optimizer.zero_grad()

            # Distance loss
            dist_loss = distance_loss_fn(model.embeddings, batch)

            # Polarity loss (computed on antonym pairs)
            pol_loss = polarity_loss_fn(model.embeddings, antonym_pairs)

            # Combined loss
            total_loss = distance_weight * dist_loss + pol_loss

            total_loss.backward()
            optimizer.step()

            epoch_dist_loss += dist_loss.item()
            epoch_pol_loss += pol_loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: "
                  f"dist_loss={epoch_dist_loss/len(distance_dataloader):.4f}, "
                  f"pol_loss={epoch_pol_loss/len(distance_dataloader):.4f}")
```

### 3. Post-Training Polarity Analysis

```python
def analyze_polarity_dimensions(checkpoint_path, antonym_pairs_file):
    """After training, identify which dimensions became polarity dimensions."""

    model = load_model(checkpoint_path)
    antonym_pairs = load_antonym_pairs(antonym_pairs_file)

    polarity_dims = identify_polarity_dimensions(model, antonym_pairs, threshold=0.7)

    print(f"Emerged polarity dimensions: {polarity_dims}")
    print(f"Found {len(polarity_dims)} dimensions with 70%+ opposite-sign consistency")

    # Test NOT operation
    test_NOT_operation(model, polarity_dims, antonym_pairs)
```

## Benefits Over Two-Stage Approach

1. **No Discovery Needed**: Polarity structure emerges during training
2. **More Stable**: Gradient-based learning vs heuristic discovery
3. **Adaptive**: Network chooses best dimensions for polarity
4. **Simpler**: One training stage instead of two
5. **Better Results**: Polarity and distance learned together, not sequentially

## Tuning Parameters

- `distance_weight`: Weight for distance loss (default: 1.0)
- `polarity_weight`: Weight for polarity loss (default: 0.1-0.5)
  - Too low: Polarity structure doesn't emerge
  - Too high: May interfere with other semantic structure
  - Recommended: Start at 0.1, increase if needed

## Files to Modify

1. `scripts/train_pytorch_polarity.py` - Add PolarityLoss, integrate into training
2. `scripts/analyze_polarity.py` - NEW - Analyze emerged polarity dimensions
3. `scripts/test_polarity_constraints.py` - Use emerged dimensions for testing

## Success Criteria

After integrated training, we should see:

1. **Spatial**: Antonyms have low similarity (<0.3) and high distance (>12)
2. **Sign Structure**: 70%+ antonym pairs have opposite signs on 5-15 dimensions
3. **NOT Operation**: NOT(good) ≈ bad with >0.6 similarity
4. **Emergent**: Polarity dimensions can be identified post-training

## Next Steps

1. Implement PolarityLoss module
2. Modify training loop to integrate polarity loss
3. Train on full WordNet data with both losses
4. Analyze emerged polarity dimensions
5. Test NOT operation on polarity dimensions
6. Compare with two-stage approach (should be better!)
