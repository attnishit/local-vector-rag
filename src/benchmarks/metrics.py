"""
Benchmark Metrics

Author: Nishit Attrey

This module provides evaluation metrics for comparing vector search algorithms,
particularly for measuring the quality of approximate nearest neighbor (ANN)
search against exact search.

Key Metrics:
    - Recall@k: Fraction of true top-k results found by ANN
    - Precision@k: Fraction of ANN results that are in true top-k
    - Intersection@k: Number of common elements in two result sets

These metrics are essential for evaluating ANN algorithms like HNSW, which
trade perfect accuracy for speed.
"""

import numpy as np
from typing import List, Set, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_intersection(
    results_a: List[Any],
    results_b: List[Any],
    key_func: callable = None
) -> int:
    """
    Calculate the intersection (number of common elements) between two result lists.

    Args:
        results_a: First list of results
        results_b: Second list of results
        key_func: Optional function to extract comparison key from each result
                 Default: identity function (compare elements directly)

    Returns:
        Number of elements in common

    Example:
        >>> a = [1, 2, 3, 4, 5]
        >>> b = [3, 4, 5, 6, 7]
        >>> calculate_intersection(a, b)  # 3 (elements: 3, 4, 5)
        >>>
        >>> # With dictionaries
        >>> a = [{"id": 1}, {"id": 2}, {"id": 3}]
        >>> b = [{"id": 2}, {"id": 3}, {"id": 4}]
        >>> calculate_intersection(a, b, key_func=lambda x: x["id"])  # 2
    """
    if key_func is None:
        key_func = lambda x: x

    set_a = set(key_func(item) for item in results_a)
    set_b = set(key_func(item) for item in results_b)

    intersection = len(set_a & set_b)

    return intersection


def calculate_recall_at_k(
    ground_truth: List[Any],
    predictions: List[Any],
    k: int,
    key_func: callable = None
) -> float:
    """
    Calculate Recall@k metric.

    Recall@k measures what fraction of the true top-k nearest neighbors
    were found by the approximate search.

    Formula:
        Recall@k = |true_top_k ∩ predicted_top_k| / k

    Args:
        ground_truth: True nearest neighbors (from exact search)
        predictions: Predicted nearest neighbors (from approximate search)
        k: Number of top results to consider
        key_func: Optional function to extract comparison key
                 (e.g., lambda x: x['node_id'])

    Returns:
        Recall value between 0.0 and 1.0
        - 1.0 = perfect recall (all true results found)
        - 0.0 = no true results found

    Example:
        >>> # Perfect recall
        >>> true_results = [1, 2, 3, 4, 5]
        >>> pred_results = [1, 2, 3, 4, 5]
        >>> calculate_recall_at_k(true_results, pred_results, k=5)  # 1.0
        >>>
        >>> # Partial recall
        >>> true_results = [1, 2, 3, 4, 5]
        >>> pred_results = [1, 2, 6, 7, 8]  # Only 1 and 2 match
        >>> calculate_recall_at_k(true_results, pred_results, k=5)  # 0.4
        >>>
        >>> # With result dictionaries
        >>> true_results = [{"id": 1}, {"id": 2}, {"id": 3}]
        >>> pred_results = [{"id": 1}, {"id": 2}, {"id": 4}]
        >>> calculate_recall_at_k(
        ...     true_results, pred_results, k=3,
        ...     key_func=lambda x: x["id"]
        ... )  # 0.667
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    # Take top-k from each list
    true_top_k = ground_truth[:k]
    pred_top_k = predictions[:k]

    # Calculate intersection
    intersection = calculate_intersection(true_top_k, pred_top_k, key_func=key_func)

    # Recall = intersection / k
    recall = intersection / k

    logger.debug(
        f"Recall@{k}: {intersection}/{k} = {recall:.4f}"
    )

    return recall


def calculate_precision_at_k(
    ground_truth: List[Any],
    predictions: List[Any],
    k: int,
    key_func: callable = None
) -> float:
    """
    Calculate Precision@k metric.

    Precision@k measures what fraction of the predicted top-k results
    are actually in the true top-k.

    Formula:
        Precision@k = |true_top_k ∩ predicted_top_k| / |predicted_top_k|

    Note: For top-k retrieval with k fixed, Precision@k == Recall@k

    Args:
        ground_truth: True nearest neighbors (from exact search)
        predictions: Predicted nearest neighbors (from approximate search)
        k: Number of top results to consider
        key_func: Optional function to extract comparison key

    Returns:
        Precision value between 0.0 and 1.0

    Example:
        >>> true_results = [1, 2, 3, 4, 5]
        >>> pred_results = [1, 2, 6, 7, 8]
        >>> calculate_precision_at_k(true_results, pred_results, k=5)  # 0.4
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    # Take top-k from each list
    true_top_k = ground_truth[:k]
    pred_top_k = predictions[:k]

    if len(pred_top_k) == 0:
        return 0.0

    # Calculate intersection
    intersection = calculate_intersection(true_top_k, pred_top_k, key_func=key_func)

    # Precision = intersection / |predictions|
    precision = intersection / len(pred_top_k)

    logger.debug(
        f"Precision@{k}: {intersection}/{len(pred_top_k)} = {precision:.4f}"
    )

    return precision


def calculate_mean_recall_at_k(
    all_ground_truth: List[List[Any]],
    all_predictions: List[List[Any]],
    k: int,
    key_func: callable = None
) -> float:
    """
    Calculate mean Recall@k across multiple queries.

    Args:
        all_ground_truth: List of ground truth results (one per query)
        all_predictions: List of predictions (one per query)
        k: Number of top results to consider
        key_func: Optional function to extract comparison key

    Returns:
        Mean recall across all queries

    Example:
        >>> ground_truth = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        >>> predictions = [[1, 2, 4], [4, 5, 7], [7, 8, 10]]
        >>> calculate_mean_recall_at_k(ground_truth, predictions, k=3)
        # Average of (2/3, 2/3, 2/3) = 0.667
    """
    if len(all_ground_truth) != len(all_predictions):
        raise ValueError(
            f"Number of ground truth ({len(all_ground_truth)}) must match "
            f"number of predictions ({len(all_predictions)})"
        )

    if len(all_ground_truth) == 0:
        return 0.0

    recalls = []
    for gt, pred in zip(all_ground_truth, all_predictions):
        recall = calculate_recall_at_k(gt, pred, k=k, key_func=key_func)
        recalls.append(recall)

    mean_recall = np.mean(recalls)

    logger.info(
        f"Mean Recall@{k} across {len(recalls)} queries: {mean_recall:.4f}"
    )

    return float(mean_recall)


def calculate_recall_statistics(
    all_ground_truth: List[List[Any]],
    all_predictions: List[List[Any]],
    k: int,
    key_func: callable = None
) -> Dict[str, float]:
    """
    Calculate detailed recall statistics across multiple queries.

    Args:
        all_ground_truth: List of ground truth results (one per query)
        all_predictions: List of predictions (one per query)
        k: Number of top results to consider
        key_func: Optional function to extract comparison key

    Returns:
        Dictionary with statistics:
            - mean: Mean recall
            - std: Standard deviation of recall
            - min: Minimum recall
            - max: Maximum recall
            - median: Median recall

    Example:
        >>> stats = calculate_recall_statistics(ground_truth, predictions, k=5)
        >>> print(f"Mean recall: {stats['mean']:.3f} ± {stats['std']:.3f}")
    """
    if len(all_ground_truth) != len(all_predictions):
        raise ValueError("Ground truth and predictions must have same length")

    if len(all_ground_truth) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
        }

    recalls = []
    for gt, pred in zip(all_ground_truth, all_predictions):
        recall = calculate_recall_at_k(gt, pred, k=k, key_func=key_func)
        recalls.append(recall)

    recalls_array = np.array(recalls)

    stats = {
        "mean": float(np.mean(recalls_array)),
        "std": float(np.std(recalls_array)),
        "min": float(np.min(recalls_array)),
        "max": float(np.max(recalls_array)),
        "median": float(np.median(recalls_array)),
    }

    logger.info(
        f"Recall@{k} statistics: "
        f"mean={stats['mean']:.4f}, "
        f"std={stats['std']:.4f}, "
        f"min={stats['min']:.4f}, "
        f"max={stats['max']:.4f}"
    )

    return stats
