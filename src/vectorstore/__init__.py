"""
Vector Storage and Retrieval Module

Author: Nishit Attrey

This module provides vector storage and nearest neighbor search:
- Stage 4: Brute-force exact search (baseline) ✓
- Stage 5: Index persistence (save/load to disk) ✓
- Stage 6: HNSW data structures definition ✓
- Stage 7: HNSW insertion algorithm
- Stage 8: HNSW approximate nearest neighbor search

Main Classes:
    From brute_force:
        - BruteForceVectorStore: Exact nearest neighbor search with persistence
        - create_vector_store: Factory function for vector stores

    From hnsw (Stage 6):
        - HNSWNode: Node in the HNSW graph
        - HNSWIndex: HNSW graph index structure
        - create_hnsw_index: Factory function for HNSW indexes

    From similarity:
        - cosine_similarity: Compute cosine similarity between vectors
        - batch_cosine_similarity: Vectorized cosine similarity
        - euclidean_distance: L2 distance between vectors
        - top_k_indices: Find indices of top-k scores

    From persistence:
        - save_index: Save vector index to disk
        - load_index: Load vector index from disk
        - index_exists: Check if index exists
        - get_index_info: Get index metadata
        - list_indexes: List all available indexes

Performance Characteristics (Stage 4):
    - Insert: O(1) per vector
    - Search: O(n * d) - compares against all vectors
    - Memory: O(n * d) - stores all vectors in RAM
    - Save/Load: O(n * d) - read/write to disk

When to Use Brute-Force:
    - Small datasets (< 10k vectors)
    - When 100% recall is required
    - For baseline benchmarking

Example:
    >>> from src.vectorstore import BruteForceVectorStore
    >>> store = BruteForceVectorStore(dimension=384, similarity_metric="cosine")
    >>> store.add(embedding, metadata={"text": "Hello", "chunk_id": "c1"})
    >>> results = store.search(query_embedding, k=5)
    >>>
    >>> # Save to disk (Stage 5)
    >>> from pathlib import Path
    >>> store.save(Path("data/indexes/my_index"))
    >>>
    >>> # Load from disk later
    >>> loaded_store = BruteForceVectorStore.load(Path("data/indexes/my_index"))

Future Stages:
    Stages 6-8 will add:
    - HNSWVectorStore: Approximate nearest neighbor search
    - 10-100x faster than brute-force for large datasets
    - Configurable recall/speed trade-off
"""

# Stage 4: Import brute-force vector store
from .brute_force import (
    BruteForceVectorStore,
    create_vector_store,
)

# Stage 4: Import similarity metrics
from .similarity import (
    cosine_similarity,
    batch_cosine_similarity,
    cosine_distance,
    euclidean_distance,
    batch_euclidean_distance,  # NEW
    dot_product,
    batch_dot_product,  # NEW
    top_k_indices,
    validate_similarity_score,
)

# Stage 5: Import persistence functions
from .persistence import (
    save_index,
    load_index,
    index_exists,
    get_index_info,
    list_indexes,
    delete_index,
)

# Stage 6: Import HNSW data structures
from .hnsw import (
    HNSWNode,
    HNSWIndex,
    create_hnsw_index,
)

__all__ = [
    # Vector stores
    "BruteForceVectorStore",
    "create_vector_store",
    # HNSW (Stage 6)
    "HNSWNode",
    "HNSWIndex",
    "create_hnsw_index",
    # Similarity metrics
    "cosine_similarity",
    "batch_cosine_similarity",
    "cosine_distance",
    "euclidean_distance",
    "batch_euclidean_distance",  # NEW
    "dot_product",
    "batch_dot_product",  # NEW
    "top_k_indices",
    "validate_similarity_score",
    # Persistence (Stage 5)
    "save_index",
    "load_index",
    "index_exists",
    "get_index_info",
    "list_indexes",
    "delete_index",
]
