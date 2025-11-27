# Memory Graph Vision: Non-Transformer Reasoning System

## Overview

This document outlines the vision for a reasoning and memory system that operates **beyond** the Semantic Vector Space (SVS). Once words have semantic encodings, we can build a knowledge graph that stores learned facts, supports logical reasoning, and enables self-teaching.

**Key principle**: This is **NOT** a transformer-based system. Reasoning happens through graph traversal, logical inference, and symbolic manipulation - not attention mechanisms or gradient descent on facts.

## Architecture Layers

The complete system has three layers:

### Layer 1: Semantic Vector Space (Current Phase)
- **Purpose**: Encode word meanings as vectors
- **Components**:
  - Embeddings: 128-512 dimensional vectors
  - Grammar metadata: Categorical features (tense, number, etc.)
- **Training**: Unsupervised discovery from parse trees and WordNet
- **Output**: `SVS("dog") = [0.23, -0.45, 0.0, 0.87, ...]`

### Layer 2: Memory Graph (This Document)
- **Purpose**: Store factual knowledge and relationships
- **Components**:
  - Nodes: Concepts (linked to SVS vectors)
  - Edges: Relationships (is-a, part-of, causes, equals, etc.)
  - Facts: Atomic true statements
- **Operations**: Graph traversal, pattern matching, logical inference
- **Output**: Knowledge base of learned truths

### Layer 3: Reasoning Engine (Future)
- **Purpose**: Answer questions, solve problems, learn new concepts
- **Components**:
  - Query parser: Natural language → graph query
  - Inference engine: Derive new facts from existing facts
  - Self-teaching: Store derived facts back to memory graph
- **Operations**: Question answering, problem solving, concept learning
- **Output**: Answers, solutions, new knowledge

## Memory Graph Structure

### Nodes: Concepts

Each node represents a concept and links to its SVS encoding:

```python
class ConceptNode:
    concept_id: str           # Unique identifier
    svs_vector: np.array      # Link to semantic vector space
    grammar_metadata: dict    # Grammatical features
    type: str                 # 'concrete', 'abstract', 'number', 'operation'
    instances: List[str]      # Specific instances of this concept
```

**Examples**:
```python
# Concrete concept
dog_node = ConceptNode(
    concept_id="dog_wn.01_n",
    svs_vector=SVS("dog"),
    type="concrete",
    instances=["Fido", "Rex", "my neighbor's dog"]
)

# Abstract concept
addition_node = ConceptNode(
    concept_id="addition_operation",
    svs_vector=SVS("add") + SVS("sum") / 2,  # Composite vector
    type="operation",
    instances=[]
)

# Number concept
three_node = ConceptNode(
    concept_id="number_3",
    svs_vector=SVS("three"),
    type="number",
    value=3  # Ground to actual number
)
```

### Edges: Relationships

Edges represent typed relationships between concepts:

```python
class RelationshipEdge:
    source: ConceptNode
    target: ConceptNode
    relation_type: str        # 'is-a', 'part-of', 'causes', 'equals', etc.
    confidence: float         # 0.0 to 1.0
    source: str               # 'told', 'inferred', 'observed'
    timestamp: datetime       # When learned
```

**Relationship Types**:
- **Taxonomic**: is-a, part-of, instance-of
- **Semantic**: similar-to, opposite-of, related-to
- **Mathematical**: equals, greater-than, less-than
- **Causal**: causes, prevents, enables
- **Temporal**: before, after, during
- **Spatial**: inside, outside, near, far

**Example Graph**:
```
[1] --equals--> [1]
[1] --is-a--> [number]
[1+1] --equals--> [2]
[2+1] --equals--> [3]
[addition] --is-a--> [operation]
[addition] --requires--> [two operands]
```

### Facts: Atomic Statements

Facts are stored as subgraphs that represent true statements:

```python
class Fact:
    fact_id: str
    subject: ConceptNode
    predicate: RelationshipEdge
    object: ConceptNode
    confidence: float
    derivation: Optional[List[Fact]]  # What facts this was derived from
```

**Example Facts**:
```python
# Told directly
fact_1 = Fact(
    subject=ConceptNode("1"),
    predicate="equals",
    object=ConceptNode("1"),
    confidence=1.0,
    derivation=None  # Axiom
)

# Told directly
fact_2 = Fact(
    subject=Expression("1+1"),
    predicate="equals",
    object=ConceptNode("2"),
    confidence=1.0,
    derivation=None  # Told by user
)

# Inferred from fact_1 and fact_2
fact_3 = Fact(
    subject=Expression("1+1+1"),
    predicate="equals",
    object=ConceptNode("3"),
    confidence=0.95,
    derivation=[fact_2, algebraic_rule_addition]  # Derived
)
```

## Reasoning Operations

### 1. Pattern Matching

**Goal**: Find subgraphs that match a query pattern.

**Example**:
```
Query: "What is 1+1+1?"
Pattern: [1+1+1] --equals--> [?]

Search:
  - Look for direct fact: [1+1+1] = [?]
  - Not found in graph!
  - Trigger inference...
```

### 2. Logical Inference

**Goal**: Derive new facts from existing facts using inference rules.

**Inference Rules**:
```python
# Addition is associative
Rule_1: If (a+b)=c and (c+d)=e, then (a+b+d)=e

# Transitive equality
Rule_2: If a=b and b=c, then a=c

# Substitution
Rule_3: If a=b, then f(a)=f(b) for any function f
```

**Example Inference**:
```
Known facts:
  [1+1] = [2]
  [2+1] = [3]

Query: What is [1+1+1]?

Inference:
  Step 1: Recognize [1+1+1] = [(1+1)+1]
  Step 2: Substitute [1+1] → [2] using known fact
  Step 3: Get [2+1]
  Step 4: Substitute [2+1] → [3] using known fact
  Step 5: Conclude [1+1+1] = [3]

Confidence: 0.95 (high, because derived from high-confidence facts)
```

### 3. Self-Teaching: Storing Derived Facts

**Critical feature**: When the system derives a new fact through inference, it **stores** that fact in the memory graph.

**Benefits**:
- **Faster next time**: [1+1+1]=3 is now a direct lookup, no inference needed
- **Compound learning**: New facts become building blocks for further inference
- **Knowledge accumulation**: Graph grows over time

**Example**:
```python
# First time: What is 1+1+1?
inference_result = infer_from_rules([fact_1_plus_1, fact_2_plus_1])
# Takes 3 steps, 0.2 seconds

# Store result
new_fact = Fact(
    subject=Expression("1+1+1"),
    predicate="equals",
    object=ConceptNode("3"),
    confidence=0.95,
    derivation=[fact_1_plus_1, fact_2_plus_1, Rule_addition_associative]
)
memory_graph.add_fact(new_fact)

# Second time: What is 1+1+1?
result = memory_graph.lookup([1+1+1] --equals--> [?])
# Direct lookup, 0.001 seconds
```

### 4. Algebraic Reasoning

**Example progression** (user's original idea):

**Session 1: Being Told Facts**
```
User: "1+1 is 2"
System: [Stores fact: 1+1=2]

User: "2+1 is 3"
System: [Stores fact: 2+1=3]
```

**Session 2: Basic Inference**
```
User: "What is 1+1+1?"
System: [Searches graph, doesn't find direct fact]
System: [Infers: 1+1+1 = (1+1)+1 = 2+1 = 3]
System: [Stores new fact: 1+1+1=3]
System: "1+1+1 is 3"
```

**Session 3: Learning Patterns**
```
User: "What is 1+1+1+1?"
System: [Infers: 1+1+1+1 = (1+1+1)+1 = 3+1 = 4]
System: [Wait, I don't know 3+1 yet!]
System: [Further inference: 3+1 = 2+1+1 = 2+2 = ...]
System: [Actually, let me ask] "I don't know 3+1 yet. Can you tell me?"
User: "3+1 is 4"
System: [Stores fact: 3+1=4]
System: [Re-evaluates: 1+1+1+1 = 3+1 = 4]
System: "1+1+1+1 is 4"
```

**Session 4: Generalizing**
```
System: [Notices pattern in stored facts]
System: [All my facts have form: a+1 = b]
System: [Generates hypothesis: "For any number n, n+1 = successor(n)"]
System: [Creates inference rule, not just fact]
System: "I've learned that adding 1 gives the next number. I can now answer any n+1 question."
```

This is **self-teaching**: the system learns algebraic rules from examples and stores them for future use.

## Beyond Numbers: Semantic Reasoning

The same approach works for semantic knowledge:

**Example: Learning About Dogs**
```
User: "Dogs are animals"
System: [Stores: dog --is-a--> animal]

User: "Animals need food"
System: [Stores: animal --requires--> food]

User: "Do dogs need food?"
System: [Searches: dog --requires--> food]
System: [Not found directly]
System: [Infers: dog --is-a--> animal, animal --requires--> food]
System: [Applies transitivity: dog --requires--> food]
System: "Yes, dogs need food" [confidence: 0.9, because inferred]
System: [Stores derived fact]
```

**Example: Learning Concepts on the Edge**
```
User: "A smartphone is a small computer you carry in your pocket"
System: [Searches SVS for "smartphone"]
System: [Not found - new concept!]
System: [Parses definition: small computer + portable + pocket-sized]
System: [Creates new node: smartphone]
System: [Initializes SVS vector: near "computer" + "portable" + "small"]
System: [Adds relationships:
  smartphone --is-a--> computer
  smartphone --has-property--> portable
  smartphone --has-property--> pocket-sized
]
System: [Stores fact]
System: "I've learned what a smartphone is. It's similar to a computer but portable."
```

**Example: Updating SVS Encodings**
```
Over many conversations, system sees "smartphone" used near:
- "app", "touchscreen", "internet", "camera", "social media"

System: [Notices: smartphone co-occurs with these concepts frequently]
System: [Current SVS(smartphone) is near SVS(computer)]
System: [But usage suggests it's also near SVS(camera) + SVS(communication)]
System: [Adjusts SVS(smartphone) to reflect actual usage]
System: [New vector is centroid of: computer + camera + communication device]
System: [Stores updated vector]
```

**When to update SVS**: Only after seeing enough evidence (e.g., 100+ usages in diverse contexts). This prevents one-off usages from corrupting the semantic space.

## Integration with Parsing

### Query Processing Pipeline

```
User Input: "What is 1+1+1?"
    ↓
[1] Parse using quantum parser
    ↓
Parse Tree: [ROOT [NP What] [VP is [NP [1+1+1]]]]
    ↓
[2] Extract semantic structure
    ↓
Query Structure: EQUALS(?x, [1+1+1])
    ↓
[3] Convert to graph query
    ↓
Graph Query: [1+1+1] --equals--> [?]
    ↓
[4] Search memory graph
    ↓
Not found → Trigger inference
    ↓
[5] Apply inference rules
    ↓
Inferred fact: [1+1+1] = [3]
    ↓
[6] Store new fact in graph
    ↓
[7] Generate response
    ↓
Output: "1+1+1 is 3"
```

### Learning Pipeline

```
User Input: "Dogs are mammals"
    ↓
[1] Parse using quantum parser
    ↓
Parse Tree: [S [NP Dogs] [VP are [NP mammals]]]
    ↓
[2] Extract relationships from parse tree
    ↓
Relationship: dog --is-a--> mammal
    ↓
[3] Check if nodes exist in graph
    ↓
dog: Exists (SVS vector loaded)
mammal: Exists (SVS vector loaded)
    ↓
[4] Check if relationship exists
    ↓
Not found → New fact!
    ↓
[5] Store in memory graph
    ↓
New edge: [dog] --is-a--> [mammal] (confidence: 1.0, source: told)
    ↓
[6] Check for derivable facts
    ↓
Found: [mammal] --is-a--> [animal]
Inferred: [dog] --is-a--> [animal] (by transitivity)
    ↓
[7] Store derived facts
    ↓
Output: "Okay, I've learned that dogs are mammals. I also inferred that dogs are animals."
```

## Data Structures

### Memory Graph Storage

**Option 1: Graph Database (Neo4j, ArangoDB)**
- Native graph queries
- Efficient traversal
- ACID transactions
- Supports billions of nodes/edges

**Option 2: In-Memory Graph (NetworkX)**
- Fast for small-medium graphs (<1M nodes)
- Easy to manipulate
- Python-native
- Limited scalability

**Recommendation**: Start with NetworkX, migrate to Neo4j when graph exceeds 100K facts.

### Fact Storage Format

```json
{
  "fact_id": "fact_000123",
  "subject": {
    "concept_id": "dog_wn.01_n",
    "svs_vector_ref": "embeddings.npy[42]"
  },
  "predicate": {
    "type": "is-a",
    "confidence": 1.0
  },
  "object": {
    "concept_id": "mammal_wn.01_n",
    "svs_vector_ref": "embeddings.npy[157]"
  },
  "metadata": {
    "source": "told",
    "timestamp": "2025-11-27T10:30:00Z",
    "derivation": null
  }
}
```

### Inference Rule Format

```python
class InferenceRule:
    rule_id: str
    name: str
    pattern: GraphPattern           # What to match
    conclusion: GraphPattern        # What to infer
    confidence_decay: float         # How much confidence decreases

# Example: Transitivity of is-a
transitivity_rule = InferenceRule(
    rule_id="rule_transitivity_is_a",
    name="Transitivity of is-a relation",
    pattern=GraphPattern([
        ("?a", "is-a", "?b"),
        ("?b", "is-a", "?c")
    ]),
    conclusion=GraphPattern([
        ("?a", "is-a", "?c")
    ]),
    confidence_decay=0.95  # Slightly less confident in derived fact
)
```

## Advantages Over Transformer-Based Systems

### 1. Explainability
```
Transformer: "The answer is 3" [black box]
Memory Graph: "The answer is 3 because:
  1. I know 1+1=2 (you told me)
  2. I know 2+1=3 (you told me)
  3. Therefore 1+1+1 = (1+1)+1 = 2+1 = 3"
```

### 2. Truthfulness
```
Transformer: Generates plausible-sounding but potentially false answers
Memory Graph: Only asserts what it has derived from known facts
              If derivation fails, admits "I don't know"
```

### 3. Learning Efficiency
```
Transformer: Needs thousands of examples to learn "1+x=x+1"
Memory Graph: Learns from single example + algebraic rule
              Generalizes immediately
```

### 4. Memory Persistence
```
Transformer: Forgets after training, needs fine-tuning to update
Memory Graph: Stores every learned fact permanently
              Retrieves instantly without retraining
```

### 5. Incremental Learning
```
Transformer: Catastrophic forgetting when learning new facts
Memory Graph: Seamlessly adds new facts to existing knowledge
              No forgetting of old facts
```

## Implementation Roadmap

### Phase 1: Basic Memory Graph (1-2 months)
- Implement ConceptNode and RelationshipEdge classes
- Build graph storage using NetworkX
- Store facts from parsed sentences
- Basic lookup: "Does X = Y?"

### Phase 2: Simple Inference (2-3 months)
- Implement pattern matching on graph
- Add basic inference rules (transitivity, substitution)
- Enable "1+1+1" reasoning from "1+1=2" and "2+1=3"
- Store derived facts back to graph

### Phase 3: Self-Teaching (3-4 months)
- Detect gaps in knowledge ("I don't know 3+1 yet")
- Ask user for missing facts
- Learn patterns from stored facts
- Generate new inference rules from patterns

### Phase 4: Semantic Reasoning (4-6 months)
- Extend beyond numbers to semantic concepts
- "Dogs are animals" → "Dogs need food"
- Handle taxonomic, causal, temporal reasoning
- Update SVS encodings based on usage

### Phase 5: Concept Learning (6-12 months)
- Learn new concepts from definitions
- Initialize SVS vectors for new concepts
- Adjust vectors based on co-occurrence
- Prune incorrect facts based on contradictions

### Ultimate Goal: AGI Foundation
- Memory graph with millions of facts
- Inference engine that can reason about any domain
- Self-teaching system that learns from conversation
- Integration with SVS for semantic grounding

## Summary

**Vision**: Build a reasoning system that stores factual knowledge in a memory graph and derives new knowledge through logical inference, NOT gradient descent.

**Key Properties**:
- **Explainable**: Every answer traces back to known facts and inference rules
- **Truthful**: Only asserts what it can derive, admits ignorance otherwise
- **Efficient**: Learns from single examples, generalizes immediately
- **Persistent**: Never forgets, always grows knowledge
- **Self-teaching**: Stores derived facts, learns patterns, generates rules

**Architecture**:
- Nodes: Concepts (linked to SVS)
- Edges: Relationships (is-a, equals, causes, etc.)
- Facts: Atomic true statements
- Rules: Inference patterns

**Example**: "1+1=2" + "2+1=3" → infer "1+1+1=3" → store as fact → answer "What is 1+1+1?" instantly

**This is the path from semantic space to general intelligence.**
