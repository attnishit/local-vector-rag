"""
Confidence Scoring

Author: Nishit Attrey

This module calculates confidence scores for generated answers based on
retrieval quality. Higher scores indicate that the retrieved context is
likely to support a good answer.

Confidence Factors:
- Average retrieval score
- Score variance/consistency
- Number of high-quality results
- Coverage of query terms

Example:
    >>> from src.generation.confidence import calculate_confidence
    >>> results = collection.search("What is HNSW?", k=5)
    >>> confidence = calculate_confidence(results, "What is HNSW?")
    >>> print(f"Confidence: {confidence:.2f}")
"""

import logging
import numpy as np
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def calculate_confidence(
    results: List[Dict[str, Any]],
    query: str,
    min_score_threshold: float = 0.5
) -> float:
    """
    Calculate confidence score based on retrieval quality.

    Args:
        results: Search results from Collection.search()
        query: Original query text
        min_score_threshold: Minimum score to consider high-quality

    Returns:
        Confidence score (0.0-1.0)

    Algorithm:
        Confidence combines:
        1. Average retrieval score (40%)
        2. Score consistency - low variance is better (30%)
        3. Fraction of high-quality results (20%)
        4. Number of results available (10%)

    Example:
        >>> results = [
        ...     {'score': 0.95, 'metadata': {...}},
        ...     {'score': 0.88, 'metadata': {...}},
        ...     {'score': 0.82, 'metadata': {...}}
        ... ]
        >>> confidence = calculate_confidence(results, "sample query")
        >>> # High scores, low variance → high confidence
    """
    if not results:
        logger.warning("No results provided for confidence calculation")
        return 0.0

    # Extract scores
    scores = [r['score'] for r in results]

    # 1. Average score (40%)
    avg_score = np.mean(scores)
    avg_score_component = avg_score * 0.4

    # 2. Score consistency - penalize high variance (30%)
    if len(scores) > 1:
        score_std = np.std(scores)
        # Normalize std to 0-1 range (assume max std of 0.5)
        normalized_std = min(score_std / 0.5, 1.0)
        # Convert to consistency score (high variance = low consistency)
        consistency = 1.0 - normalized_std
    else:
        # Single result = perfect consistency
        consistency = 1.0
    consistency_component = consistency * 0.3

    # 3. Fraction of high-quality results (20%)
    high_quality_count = sum(1 for s in scores if s >= min_score_threshold)
    high_quality_fraction = high_quality_count / len(scores)
    quality_component = high_quality_fraction * 0.2

    # 4. Number of results (10%)
    # More results (up to 10) = higher confidence
    num_results_score = min(len(scores) / 10.0, 1.0)
    num_results_component = num_results_score * 0.1

    # Combine components
    confidence = (
        avg_score_component
        + consistency_component
        + quality_component
        + num_results_component
    )

    # Ensure in [0, 1] range
    confidence = max(0.0, min(1.0, confidence))

    logger.debug(
        f"Confidence: {confidence:.3f} "
        f"(avg={avg_score:.3f}, consistency={consistency:.3f}, "
        f"quality={high_quality_fraction:.2f}, n={len(scores)})"
    )

    return confidence


def get_confidence_level(score: float) -> str:
    """
    Convert confidence score to human-readable level.

    Args:
        score: Confidence score (0.0-1.0)

    Returns:
        Confidence level: "Low", "Medium", or "High"

    Example:
        >>> get_confidence_level(0.85)
        'High'
        >>> get_confidence_level(0.45)
        'Low'
    """
    if score < 0.5:
        return "Low"
    elif score < 0.75:
        return "Medium"
    else:
        return "High"


def explain_confidence(
    score: float,
    results: List[Dict[str, Any]]
) -> str:
    """
    Generate human-readable explanation of confidence score.

    Args:
        score: Confidence score
        results: Search results used to calculate confidence

    Returns:
        Explanation string

    Example:
        >>> explanation = explain_confidence(0.85, results)
        >>> print(explanation)
        High confidence (0.85): Found 5 high-quality results with
        consistent relevance scores (avg: 0.88).
    """
    if not results:
        return "Low confidence (0.00): No results found."

    level = get_confidence_level(score)
    scores = [r['score'] for r in results]
    avg_score = np.mean(scores)
    num_results = len(scores)

    if level == "High":
        explanation = (
            f"High confidence ({score:.2f}): "
            f"Found {num_results} high-quality results with "
            f"consistent relevance scores (avg: {avg_score:.2f}). "
            f"The answer should be well-supported by the context."
        )
    elif level == "Medium":
        explanation = (
            f"Medium confidence ({score:.2f}): "
            f"Found {num_results} results with moderate relevance "
            f"(avg: {avg_score:.2f}). "
            f"The answer may have some uncertainty."
        )
    else:
        explanation = (
            f"Low confidence ({score:.2f}): "
            f"Only found {num_results} results with limited relevance "
            f"(avg: {avg_score:.2f}). "
            f"The answer may not be reliable or complete."
        )

    return explanation
