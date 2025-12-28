"""
Brute-Force Vector Store

Author: Nishit Attrey

This module implements a simple, exact vector store that compares the query
against ALL stored vectors. This is the baseline implementation that we'll
later optimize with HNSW (Stages 6-8).

Performance Characteristics:
    - Insert: O(1) - just append to array
    - Search: O(n * d) - compare against all n vectors of dimension d
    - Memory: O(n * d) - store all vectors
    - Save: O(n * d) - write vectors to disk
    - Load: O(n * d) - read vectors from disk

Trade-offs:
    - ✓ Exact results (100% recall)
    - ✓ Simple implementation
    - ✓ No index build time
    - ✓ Can persist to disk (Stage 5)
    - ✗ Slow for large datasets (>10k vectors)
    - ✗ Linear search time

When to use:
    - Small datasets (< 10k vectors)
    - When 100% recall is required
    - For baseline benchmarking

Author: RAG Team
Version: 0.1.0-stage5
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

from .similarity import (
    batch_cosine_similarity,
    top_k_indices,
    validate_similarity_score,
)
from .persistence import (
    save_index,
    load_index,
    index_exists,
)

logger = logging.getLogger(__name__)


class BruteForceVectorStore:
    """
    Brute-force exact nearest neighbor search.

    This is the simplest possible vector store: it stores all vectors in memory
    and performs exhaustive linear search to find nearest neighbors.

    Attributes:
        vectors: numpy array of shape (n, d) storing all vectors
        metadata: list of metadata dicts for each vector
        dimension: expected vector dimension
        similarity_metric: distance metric ("cosine", "euclidean", etc.)
        normalized: whether vectors are L2-normalized

    Example:
        >>> store = BruteForceVectorStore(dimension=384, similarity_metric="cosine")
        >>> store.add(embedding, {"text": "Hello world", "doc_id": "doc1"})
        >>> results = store.search(query_embedding, k=5)
        >>> for result in results:
        ...     print(f"Score: {result['score']}, Text: {result['metadata']['text']}")
    """

    def __init__(self, dimension: int, similarity_metric: str = "cosine", normalized: bool = True):
        """
        Initialize an empty brute-force vector store.

        Args:
            dimension: Expected dimensionality of vectors
            similarity_metric: Similarity metric to use ("cosine", "dot", "euclidean")
            normalized: If True, assumes vectors are L2-normalized (faster cosine)

        Raises:
            ValueError: If invalid metric or dimension
        """
        if dimension <= 0:
            raise ValueError(f"Dimension must be positive, got {dimension}")

        if similarity_metric not in ["cosine", "dot", "euclidean"]:
            raise ValueError(
                f"Unsupported metric: {similarity_metric}. Supported: cosine, dot, euclidean"
            )

        self.dimension = dimension
        self.similarity_metric = similarity_metric
        self.normalized = normalized

        self._capacity = 1000
        self._num_vectors = 0
        self.vectors = np.zeros((self._capacity, dimension), dtype=np.float32)
        self.metadata: List[Dict[str, Any]] = []

        logger.info(
            f"Initialized BruteForceVectorStore: "
            f"dim={dimension}, metric={similarity_metric}, normalized={normalized}"
        )

    def _ensure_capacity(self, needed_capacity: int) -> None:
        """
        Grow vectors array if needed (ArrayList-style dynamic growth).

        This prevents O(n²) complexity from repeated vstack operations.

        Args:
            needed_capacity: Minimum capacity required
        """
        if needed_capacity <= self._capacity:
            return

        new_capacity = max(int(self._capacity * 1.5), needed_capacity)

        logger.debug(f"Growing vector array: {self._capacity} → {new_capacity}")

        new_vectors = np.zeros((new_capacity, self.dimension), dtype=np.float32)
        new_vectors[:self._num_vectors] = self.vectors[:self._num_vectors]

        self.vectors = new_vectors
        self._capacity = new_capacity

    def add(self, vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Add a single vector to the store.

        Args:
            vector: Embedding vector to add (1D array of shape (d,))
            metadata: Optional metadata to store with vector (e.g., chunk_id, text)

        Returns:
            Index of added vector

        Raises:
            ValueError: If vector has wrong dimension

        Example:
            >>> idx = store.add(embedding, {"chunk_id": "c1", "text": "sample"})
            >>> print(f"Added vector at index {idx}")
        """
        if vector.ndim != 1:
            raise ValueError(f"Vector must be 1D, got shape {vector.shape}")

        if vector.shape[0] != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, got {vector.shape[0]}"
            )

        self._ensure_capacity(self._num_vectors + 1)

        self.vectors[self._num_vectors] = vector
        new_index = self._num_vectors
        self._num_vectors += 1

        self.metadata.append(metadata or {})

        logger.debug(f"Added vector at index {new_index}")

        return new_index

    def add_batch(
        self, vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[int]:
        """
        Add multiple vectors at once (more efficient than individual adds).

        Args:
            vectors: Array of vectors to add (2D array of shape (n, d))
            metadata: Optional list of metadata dicts (length must match n)

        Returns:
            List of indices for added vectors

        Raises:
            ValueError: If dimensions don't match or metadata length is wrong
        """
        if vectors.ndim != 2:
            raise ValueError(f"Vectors must be 2D array, got shape {vectors.shape}")

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, got {vectors.shape[1]}"
            )

        n = len(vectors)
        if metadata is not None and len(metadata) != n:
            raise ValueError(
                f"Metadata length ({len(metadata)}) must match number of vectors ({n})"
            )

        start_index = self._num_vectors

        self._ensure_capacity(self._num_vectors + n)

        self.vectors[self._num_vectors:self._num_vectors + n] = vectors
        self._num_vectors += n

        if metadata is None:
            metadata = [{} for _ in range(n)]

        self.metadata.extend(metadata)

        indices = list(range(start_index, start_index + n))

        logger.info(f"Added batch of {n} vectors")

        return indices

    def search(
        self,
        query: np.ndarray,
        k: int = 5,
        return_scores: bool = True,
        return_vectors: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Search for k nearest neighbors using brute-force comparison.

        This is the core search method: it computes similarity between the query
        and ALL stored vectors, then returns the top-k most similar.

        Algorithm:
            1. Compute similarity scores for all vectors: O(n * d)
            2. Find indices of top-k scores: O(n)
            3. Return results with metadata

        Args:
            query: Query vector (1D array of shape (d,))
            k: Number of results to return
            return_scores: If True, include similarity scores in results
            return_vectors: If True, include vectors in results

        Returns:
            List of result dicts, sorted by similarity (descending).
            Each dict contains:
                - index: Index in vector store
                - metadata: Associated metadata
                - score: Similarity score (if return_scores=True)
                - vector: Vector array (if return_vectors=True)

        Raises:
            ValueError: If query has wrong dimension
            RuntimeError: If store is empty

        Example:
            >>> results = store.search(query_embedding, k=3)
            >>> for r in results:
            ...     print(f"Score: {r['score']:.3f}, Text: {r['metadata']['text']}")
        """
        if query.ndim != 1:
            raise ValueError(f"Query must be 1D vector, got shape {query.shape}")

        if query.shape[0] != self.dimension:
            raise ValueError(
                f"Query dimension mismatch: expected {self.dimension}, got {query.shape[0]}"
            )

        if self._num_vectors == 0:
            logger.warning("Search called on empty vector store")
            return []

        k = min(k, self._num_vectors)

        logger.debug(f"Searching for top-{k} vectors (metric={self.similarity_metric})")

        vectors_used = self.vectors[:self._num_vectors]

        if self.similarity_metric == "cosine":
            scores = batch_cosine_similarity(query, vectors_used, normalized=self.normalized)
        elif self.similarity_metric == "dot":
            scores = np.dot(vectors_used, query)
        else:
            diffs = vectors_used - query.reshape(1, -1)
            distances = np.linalg.norm(diffs, axis=1)
            scores = -distances

        top_indices = top_k_indices(scores, k)

        results = []
        for idx in top_indices:
            idx = int(idx)
            result = {
                "index": idx,
                "metadata": self.metadata[idx],
            }

            if return_scores:
                score = float(scores[idx])
                if self.similarity_metric == "euclidean":
                    score = -score
                result["score"] = score

            if return_vectors:
                result["vector"] = self.vectors[idx]

            results.append(result)

        logger.debug(f"Found {len(results)} results")

        return results

    def __len__(self) -> int:
        """Return number of vectors in store."""
        return self._num_vectors

    def __repr__(self) -> str:
        """String representation of store."""
        return (
            f"BruteForceVectorStore("
            f"n={len(self)}, "
            f"dim={self.dimension}, "
            f"metric={self.similarity_metric})"
        )

    def get_vector(self, index: int) -> np.ndarray:
        """
        Get vector at given index.

        Args:
            index: Index of vector to retrieve

        Returns:
            Vector array

        Raises:
            IndexError: If index out of range
        """
        if not 0 <= index < len(self):
            raise IndexError(f"Index {index} out of range [0, {len(self)})")

        return self.vectors[index]

    def get_metadata(self, index: int) -> Dict[str, Any]:
        """
        Get metadata at given index.

        Args:
            index: Index of metadata to retrieve

        Returns:
            Metadata dict

        Raises:
            IndexError: If index out of range
        """
        if not 0 <= index < len(self):
            raise IndexError(f"Index {index} out of range [0, {len(self)})")

        return self.metadata[index]

    def statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dict with:
                - num_vectors: Number of vectors stored
                - dimension: Vector dimension
                - similarity_metric: Distance metric used
                - normalized: Whether vectors are normalized
                - memory_bytes: Approximate memory usage
        """
        memory_bytes = self.vectors.nbytes + sum(len(str(m)) for m in self.metadata)

        return {
            "num_vectors": len(self),
            "dimension": self.dimension,
            "similarity_metric": self.similarity_metric,
            "normalized": self.normalized,
            "memory_bytes": memory_bytes,
            "memory_mb": memory_bytes / (1024 * 1024),
        }

    def save(self, index_dir: Path, index_name: str = "default") -> None:
        """
        Save this vector store to disk (Stage 5).

        Creates a directory with:
            - vectors.npy: Binary vector data
            - metadata.json: Metadata for each vector
            - index_info.json: Index configuration and checksum

        Args:
            index_dir: Directory to save index in
            index_name: Name for this index (for logging)

        Raises:
            ValueError: If store is empty
            IOError: If unable to write files

        Example:
            >>> store.save(Path("data/indexes/my_index"))
            >>> # Later...
            >>> loaded_store = BruteForceVectorStore.load(Path("data/indexes/my_index"))
        """
        if len(self) == 0:
            raise ValueError("Cannot save empty vector store")

        logger.info(f"Saving vector store to {index_dir}")

        vectors_to_save = self.vectors[:self._num_vectors]

        save_index(
            index_dir=index_dir,
            vectors=vectors_to_save,
            metadata=self.metadata,
            dimension=self.dimension,
            similarity_metric=self.similarity_metric,
            normalized=self.normalized,
            index_name=index_name,
        )

        logger.info(f"Vector store saved: {len(self)} vectors")

    @classmethod
    def load(cls, index_dir: Path, verify_checksum: bool = True) -> "BruteForceVectorStore":
        """
        Load a vector store from disk (Stage 5).

        Args:
            index_dir: Directory containing saved index
            verify_checksum: If True, verify data integrity

        Returns:
            Loaded BruteForceVectorStore instance

        Raises:
            FileNotFoundError: If index doesn't exist
            ValueError: If index is corrupted or incompatible

        Example:
            >>> store = BruteForceVectorStore.load(Path("data/indexes/my_index"))
            >>> print(f"Loaded {len(store)} vectors")
        """
        logger.info(f"Loading vector store from {index_dir}")

        vectors, metadata, index_info = load_index(
            index_dir=index_dir, verify_checksum=verify_checksum
        )

        store = cls(
            dimension=index_info["dimension"],
            similarity_metric=index_info["similarity_metric"],
            normalized=index_info["normalized"],
        )

        n_vectors = len(vectors)
        store._ensure_capacity(n_vectors)
        store.vectors[:n_vectors] = vectors
        store._num_vectors = n_vectors
        store.metadata = metadata

        logger.info(f"Vector store loaded: {len(store)} vectors, {store.dimension}D")

        return store


def create_vector_store(
    dimension: int,
    algorithm: str = "brute_force",
    similarity_metric: str = "cosine",
    normalized: bool = True,
    **kwargs,
) -> BruteForceVectorStore:
    """
    Factory function to create a vector store.

    This provides a unified interface for creating different types of vector stores.
    Currently only supports brute_force, but will support HNSW in Stages 6-8.

    Args:
        dimension: Vector dimension
        algorithm: Algorithm to use ("brute_force", "hnsw" in future)
        similarity_metric: Distance metric ("cosine", "dot", "euclidean")
        normalized: Whether vectors are L2-normalized
        **kwargs: Additional algorithm-specific parameters

    Returns:
        Vector store instance

    Raises:
        ValueError: If unsupported algorithm

    Example:
        >>> store = create_vector_store(
        ...     dimension=384,
        ...     algorithm="brute_force",
        ...     similarity_metric="cosine"
        ... )
    """
    if algorithm == "brute_force":
        return BruteForceVectorStore(
            dimension=dimension, similarity_metric=similarity_metric, normalized=normalized
        )
    else:
        raise ValueError(
            f"Unsupported algorithm: {algorithm}. "
            f"Supported: brute_force (hnsw coming in Stages 6-8)"
        )
