"""
Benchmark Data Generation

Author: Nishit Attrey

This module provides utilities for generating synthetic datasets for benchmarking
vector search algorithms.

Functions:
    generate_random_vectors: Generate random normalized vectors
    generate_queries: Generate query vectors (random or from dataset)
    create_benchmark_dataset: Create a complete benchmark dataset with metadata

The generated data is suitable for testing recall, latency, and scalability.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def generate_random_vectors(
    n: int,
    dimension: int,
    normalize: bool = True,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate random vectors for benchmarking.

    Creates random vectors sampled from a normal distribution, optionally
    normalized to unit length (required for cosine similarity).

    Args:
        n: Number of vectors to generate
        dimension: Dimensionality of each vector
        normalize: If True, L2-normalize vectors to unit length
        seed: Random seed for reproducibility

    Returns:
        Array of shape (n, dimension) containing random vectors

    Example:
        >>> vectors = generate_random_vectors(n=1000, dimension=384, normalize=True)
        >>> print(vectors.shape)  # (1000, 384)
        >>> print(np.linalg.norm(vectors[0]))  # ~1.0 (normalized)
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if dimension <= 0:
        raise ValueError(f"dimension must be positive, got {dimension}")

    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    logger.debug(f"Generating {n} random {dimension}-dim vectors (normalize={normalize})")

    # Generate random vectors from normal distribution
    vectors = np.random.randn(n, dimension).astype(np.float32)

    # Normalize if requested
    if normalize:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.maximum(norms, 1e-8)
        vectors = vectors / norms

    logger.info(f"Generated {n} vectors of dimension {dimension}")

    return vectors


def generate_queries(
    n_queries: int,
    dimension: int,
    normalize: bool = True,
    seed: Optional[int] = None,
    from_dataset: Optional[np.ndarray] = None,
    noise_level: float = 0.0
) -> np.ndarray:
    """
    Generate query vectors for benchmarking.

    Can generate completely random queries, or queries based on existing dataset
    with optional noise (to simulate real-world scenarios where queries are
    similar but not identical to indexed vectors).

    Args:
        n_queries: Number of query vectors to generate
        dimension: Dimensionality of each vector
        normalize: If True, L2-normalize vectors
        seed: Random seed for reproducibility
        from_dataset: Optional dataset to sample queries from
                     If provided, queries will be based on this data
        noise_level: Amount of random noise to add (0.0 = no noise, 1.0 = high noise)
                    Only used if from_dataset is provided

    Returns:
        Array of shape (n_queries, dimension) containing query vectors

    Example:
        >>> # Random queries
        >>> queries = generate_queries(n_queries=100, dimension=384)
        >>>
        >>> # Queries based on dataset with noise
        >>> queries = generate_queries(
        ...     n_queries=100,
        ...     dimension=384,
        ...     from_dataset=vectors,
        ...     noise_level=0.1
        ... )
    """
    if n_queries <= 0:
        raise ValueError(f"n_queries must be positive, got {n_queries}")

    # Set random seed
    if seed is not None:
        np.random.seed(seed)

    if from_dataset is None:
        # Generate completely random queries
        logger.debug(f"Generating {n_queries} random query vectors")
        queries = generate_random_vectors(
            n=n_queries,
            dimension=dimension,
            normalize=normalize,
            seed=None  # Already set seed above
        )
    else:
        # Sample from dataset and add noise
        if len(from_dataset) == 0:
            raise ValueError("from_dataset cannot be empty")

        logger.debug(
            f"Generating {n_queries} queries from dataset "
            f"(noise_level={noise_level})"
        )

        # Randomly sample indices from dataset
        n_dataset = len(from_dataset)
        indices = np.random.choice(n_dataset, size=n_queries, replace=True)
        queries = from_dataset[indices].copy()

        # Add noise if requested
        if noise_level > 0:
            noise = np.random.randn(n_queries, dimension).astype(np.float32)
            noise = noise * noise_level
            queries = queries + noise

            # Re-normalize after adding noise
            if normalize:
                norms = np.linalg.norm(queries, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-8)
                queries = queries / norms

    logger.info(f"Generated {n_queries} query vectors")

    return queries


def create_benchmark_dataset(
    n_vectors: int,
    dimension: int,
    normalize: bool = True,
    seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Create a complete benchmark dataset with vectors and metadata.

    Generates synthetic vectors with metadata suitable for benchmarking
    the full RAG pipeline (not just vector search).

    Args:
        n_vectors: Number of vectors to generate
        dimension: Vector dimensionality
        normalize: Whether to normalize vectors
        seed: Random seed for reproducibility

    Returns:
        List of dictionaries containing:
            - vector: The embedding vector
            - metadata: Dict with chunk_id, doc_id, text (synthetic)

    Example:
        >>> dataset = create_benchmark_dataset(n_vectors=1000, dimension=384)
        >>> print(len(dataset))  # 1000
        >>> print(dataset[0].keys())  # ['vector', 'metadata']
    """
    logger.info(f"Creating benchmark dataset: n={n_vectors}, dim={dimension}")

    # Generate vectors
    vectors = generate_random_vectors(
        n=n_vectors,
        dimension=dimension,
        normalize=normalize,
        seed=seed
    )

    # Create synthetic metadata
    dataset = []
    for i, vector in enumerate(vectors):
        # Synthetic document/chunk structure
        doc_id = f"bench_doc_{i // 10}"  # ~10 chunks per document
        chunk_id = f"{doc_id}_chunk_{i % 10}"
        text = f"Synthetic benchmark text for chunk {i}"

        dataset.append({
            "vector": vector,
            "metadata": {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "text": text,
                "index": i,
            }
        })

    logger.info(f"Created benchmark dataset with {len(dataset)} items")

    return dataset
