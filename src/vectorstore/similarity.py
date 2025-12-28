"""
Vector Similarity Metrics

Author: Nishit Attrey

This module provides explicit, from-scratch implementations of vector similarity
metrics. These are the core distance functions used for nearest neighbor search.

Similarity Metrics:
    - Cosine Similarity: Measures angle between vectors (range: -1 to 1)
    - Euclidean Distance: L2 distance between vectors
    - Dot Product: Inner product (for normalized vectors, same as cosine)

Mathematical Background:
    Cosine Similarity:
        sim(A, B) = (A · B) / (||A|| * ||B||)
        Where:
        - A · B = dot product = Σ(a_i * b_i)
        - ||A|| = L2 norm = √(Σa_i²)

    For L2-normalized vectors (||A|| = ||B|| = 1.0):
        sim(A, B) = A · B (dot product only)

    Relationship to Distance:
        distance = 1 - similarity (for cosine)

Author: RAG Team
Version: 0.1.0-stage4
"""

import numpy as np
from typing import Union, List, Tuple


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute dot product of two vectors.

    This is the most basic vector operation:
        dot(a, b) = a₁*b₁ + a₂*b₂ + ... + aₙ*bₙ

    Args:
        a: First vector (1D array)
        b: Second vector (1D array)

    Returns:
        Dot product (scalar)

    Raises:
        ValueError: If vectors have different dimensions

    Example:
        >>> a = np.array([1.0, 2.0, 3.0])
        >>> b = np.array([4.0, 5.0, 6.0])
        >>> dot_product(a, b)
        32.0  # (1*4 + 2*5 + 3*6)
    """
    if a.shape != b.shape:
        raise ValueError(f"Vector dimension mismatch: {a.shape} vs {b.shape}")

    result = np.dot(a, b)

    return float(result)


def cosine_similarity(a: np.ndarray, b: np.ndarray, normalized: bool = False) -> float:
    """
    Compute cosine similarity between two vectors.

    Cosine similarity measures the angle between vectors, ranging from -1 to 1:
        - 1.0: Vectors point in same direction (0° angle)
        - 0.0: Vectors are orthogonal (90° angle)
        - -1.0: Vectors point in opposite directions (180° angle)

    Formula:
        cos_sim(a, b) = (a · b) / (||a|| * ||b||)

    For normalized vectors (||a|| = ||b|| = 1.0):
        cos_sim(a, b) = a · b

    Args:
        a: First vector (1D array)
        b: Second vector (1D array)
        normalized: If True, assumes vectors are already L2-normalized (faster)

    Returns:
        Cosine similarity score (float in [-1, 1])

    Raises:
        ValueError: If vectors have different dimensions or are zero vectors

    Example:
        >>> a = np.array([1.0, 0.0, 0.0])
        >>> b = np.array([1.0, 0.0, 0.0])
        >>> cosine_similarity(a, b)
        1.0  # Same direction

        >>> a = np.array([1.0, 0.0])
        >>> b = np.array([0.0, 1.0])
        >>> cosine_similarity(a, b)
        0.0  # Orthogonal
    """
    if a.shape != b.shape:
        raise ValueError(f"Vector dimension mismatch: {a.shape} vs {b.shape}")

    dot_prod = dot_product(a, b)

    if normalized:
        return float(dot_prod)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("Cannot compute similarity with zero vector")

    similarity = dot_prod / (norm_a * norm_b)

    return float(similarity)


def batch_cosine_similarity(
    query: np.ndarray, vectors: np.ndarray, normalized: bool = False
) -> np.ndarray:
    """
    Compute cosine similarity between a query and multiple vectors.

    This is the core operation for vector search: compare one query against
    many stored vectors to find the most similar ones.

    Args:
        query: Query vector (1D array of shape (d,))
        vectors: Matrix of vectors to compare against (2D array of shape (n, d))
        normalized: If True, assumes all vectors are L2-normalized

    Returns:
        Array of similarity scores (1D array of shape (n,))

    Raises:
        ValueError: If dimensions don't match

    Example:
        >>> query = np.array([1.0, 0.0, 0.0])
        >>> vectors = np.array([
        ...     [1.0, 0.0, 0.0],
        ...     [0.0, 1.0, 0.0],
        ...     [0.5, 0.5, 0.0]
        ... ])
        >>> batch_cosine_similarity(query, vectors)
        array([1.0, 0.0, 0.707...])
    """
    if query.ndim != 1:
        raise ValueError(f"Query must be 1D vector, got shape {query.shape}")

    if vectors.ndim != 2:
        raise ValueError(f"Vectors must be 2D matrix, got shape {vectors.shape}")

    if query.shape[0] != vectors.shape[1]:
        raise ValueError(
            f"Dimension mismatch: query dim={query.shape[0]}, vectors dim={vectors.shape[1]}"
        )

    if len(vectors) == 0:
        return np.array([])

    dot_products = np.dot(vectors, query)

    if normalized:
        return dot_products

    query_norm = np.linalg.norm(query)
    vectors_norms = np.linalg.norm(vectors, axis=1)

    if query_norm == 0.0:
        raise ValueError("Query vector has zero norm")
    if np.any(vectors_norms == 0.0):
        raise ValueError("One or more vectors have zero norm")

    similarities = dot_products / (query_norm * vectors_norms)

    return similarities


def cosine_distance(a: np.ndarray, b: np.ndarray, normalized: bool = False) -> float:
    """
    Compute cosine distance between two vectors.

    Cosine distance is defined as:
        distance = 1 - similarity

    This converts similarity (higher = more similar) to distance (lower = more similar).

    Args:
        a: First vector
        b: Second vector
        normalized: If True, assumes vectors are L2-normalized

    Returns:
        Cosine distance (float in [0, 2])

    Example:
        >>> a = np.array([1.0, 0.0])
        >>> b = np.array([1.0, 0.0])
        >>> cosine_distance(a, b)
        0.0  # Same direction, zero distance
    """
    similarity = cosine_similarity(a, b, normalized=normalized)
    return 1.0 - similarity


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Euclidean (L2) distance between two vectors.

    This is the straight-line distance:
        dist(a, b) = √(Σ(a_i - b_i)²)

    Args:
        a: First vector
        b: Second vector

    Returns:
        Euclidean distance (float >= 0)

    Example:
        >>> a = np.array([0.0, 0.0])
        >>> b = np.array([3.0, 4.0])
        >>> euclidean_distance(a, b)
        5.0  # 3-4-5 triangle
    """
    if a.shape != b.shape:
        raise ValueError(f"Vector dimension mismatch: {a.shape} vs {b.shape}")

    diff = a - b
    squared_sum = np.sum(diff**2)
    distance = np.sqrt(squared_sum)

    return float(distance)


def batch_euclidean_distance(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distance from query to multiple vectors efficiently.

    Uses optimized formula: ||A - B||^2 = ||A||^2 + ||B||^2 - 2*A·B
    This is much faster than computing each distance individually.

    Args:
        query: Query vector (1D array of shape (d,))
        vectors: Matrix of vectors to compare against (2D array of shape (n, d))

    Returns:
        Array of Euclidean distances (1D array of shape (n,))

    Raises:
        ValueError: If dimensions don't match

    Example:
        >>> query = np.array([0.0, 0.0])
        >>> vectors = np.array([[3.0, 4.0], [1.0, 0.0]])
        >>> batch_euclidean_distance(query, vectors)
        array([5.0, 1.0])
    """
    if query.ndim != 1:
        raise ValueError(f"Query must be 1D vector, got shape {query.shape}")

    if vectors.ndim != 2:
        raise ValueError(f"Vectors must be 2D matrix, got shape {vectors.shape}")

    if query.shape[0] != vectors.shape[1]:
        raise ValueError(
            f"Dimension mismatch: query dim={query.shape[0]}, vectors dim={vectors.shape[1]}"
        )

    if len(vectors) == 0:
        return np.array([])

    query_norm_sq = np.sum(query**2)
    vectors_norm_sq = np.sum(vectors**2, axis=1)
    dot_products = np.dot(vectors, query)

    distances_sq = query_norm_sq + vectors_norm_sq - 2 * dot_products
    distances_sq = np.maximum(distances_sq, 0.0)

    return np.sqrt(distances_sq)


def batch_dot_product(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    Compute dot product between query and multiple vectors efficiently.

    Args:
        query: Query vector (1D array of shape (d,))
        vectors: Matrix of vectors (2D array of shape (n, d))

    Returns:
        Array of dot products (1D array of shape (n,))

    Raises:
        ValueError: If dimensions don't match

    Example:
        >>> query = np.array([1.0, 2.0, 3.0])
        >>> vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> batch_dot_product(query, vectors)
        array([1.0, 2.0])
    """
    if query.ndim != 1:
        raise ValueError(f"Query must be 1D vector, got shape {query.shape}")

    if vectors.ndim != 2:
        raise ValueError(f"Vectors must be 2D matrix, got shape {vectors.shape}")

    if query.shape[0] != vectors.shape[1]:
        raise ValueError(
            f"Dimension mismatch: query dim={query.shape[0]}, vectors dim={vectors.shape[1]}"
        )

    if len(vectors) == 0:
        return np.array([])

    return np.dot(vectors, query)


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """
    Get indices of top-k highest scores.

    This is used to find the k most similar vectors after computing
    similarity scores for all candidates.

    Args:
        scores: Array of scores (higher = better)
        k: Number of top scores to return

    Returns:
        Indices of top-k scores, sorted by score (descending)

    Example:
        >>> scores = np.array([0.1, 0.9, 0.3, 0.7, 0.5])
        >>> top_k_indices(scores, k=3)
        array([1, 3, 4])  # Indices of 0.9, 0.7, 0.5
    """
    if len(scores) == 0:
        return np.array([], dtype=int)

    k = min(k, len(scores))

    if k == len(scores):
        indices = np.argsort(scores)[::-1]
    else:
        partitioned = np.argpartition(scores, -k)[-k:]
        indices = partitioned[np.argsort(scores[partitioned])[::-1]]

    return indices


def validate_similarity_score(score: float, metric: str = "cosine") -> None:
    """
    Validate that a similarity score is in the expected range.

    Args:
        score: Similarity score to validate
        metric: Metric name ("cosine", "euclidean", etc.)

    Raises:
        ValueError: If score is out of valid range
    """
    if metric == "cosine":
        if not (-1.0 <= score <= 1.0):
            raise ValueError(f"Cosine similarity must be in [-1, 1], got {score}")
    elif metric == "dot":
        # Dot product can be any value
        pass
    else:
        # For distances, must be non-negative
        if score < 0.0:
            raise ValueError(f"{metric} distance cannot be negative, got {score}")
