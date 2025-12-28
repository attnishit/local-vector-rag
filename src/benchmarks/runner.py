"""
Benchmark Runner

Author: Nishit Attrey

This module provides the main benchmark execution framework for comparing
brute-force and HNSW vector search algorithms.

Key Components:
    - BenchmarkConfig: Configuration for benchmark runs
    - BenchmarkResults: Results container
    - run_benchmark: Execute a complete benchmark
    - compare_algorithms: Compare brute-force vs HNSW

Metrics measured:
    - Recall@k (accuracy)
    - Search latency (speed)
    - Index build time
    - Memory usage
"""

import time
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import logging

from ..vectorstore import BruteForceVectorStore, HNSWIndex, create_hnsw_index
from .data import generate_random_vectors, generate_queries
from .metrics import calculate_mean_recall_at_k, calculate_recall_statistics

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """
    Configuration for benchmark execution.

    Attributes:
        dimension: Vector dimensionality
        dataset_size: Number of vectors to index
        n_queries: Number of queries to run
        k_values: List of k values to test (e.g., [1, 5, 10, 20])
        similarity_metric: Distance metric to use
        normalize: Whether to normalize vectors
        seed: Random seed for reproducibility

        # Query generation
        use_realistic_queries: If True, generate queries from dataset with noise
                              If False, generate completely random queries
        query_noise_level: Noise level for realistic queries (0.0 = no noise, 1.0 = high noise)
                          Only used when use_realistic_queries=True

        # HNSW-specific parameters
        hnsw_m: HNSW m parameter
        hnsw_ef_construction: HNSW ef_construction parameter
        hnsw_ef_search_values: List of ef_search values to test
    """
    dimension: int = 384
    dataset_size: int = 1000
    n_queries: int = 100
    k_values: List[int] = field(default_factory=lambda: [1, 5, 10])
    similarity_metric: str = "cosine"
    normalize: bool = True
    seed: Optional[int] = 42

    # Query generation (realistic vs random)
    use_realistic_queries: bool = True  # Default: use realistic queries
    query_noise_level: float = 0.1  # 10% noise simulates real-world query variations

    # HNSW parameters
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search_values: List[int] = field(default_factory=lambda: [10, 50, 100])


@dataclass
class BenchmarkResults:
    """
    Results from a benchmark run.

    Attributes:
        config: The configuration used
        brute_force_build_time: Time to build brute-force index (seconds)
        hnsw_build_time: Time to build HNSW index (seconds)
        brute_force_memory_mb: Memory used by brute-force index (MB)
        hnsw_memory_mb: Memory used by HNSW index (MB)
        recall_results: Dict of recall results by (k, ef_search)
        latency_results: Dict of search latency by algorithm
    """
    config: BenchmarkConfig
    brute_force_build_time: float = 0.0
    hnsw_build_time: float = 0.0
    brute_force_memory_mb: float = 0.0
    hnsw_memory_mb: float = 0.0
    recall_results: Dict[str, Any] = field(default_factory=dict)
    latency_results: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for serialization."""
        return {
            "config": asdict(self.config),
            "build_time": {
                "brute_force": self.brute_force_build_time,
                "hnsw": self.hnsw_build_time,
                "speedup": (
                    self.brute_force_build_time / self.hnsw_build_time
                    if self.hnsw_build_time > 0 else 0.0
                ),
            },
            "memory_mb": {
                "brute_force": self.brute_force_memory_mb,
                "hnsw": self.hnsw_memory_mb,
            },
            "recall": self.recall_results,
            "latency": self.latency_results,
        }


def run_benchmark(config: Optional[BenchmarkConfig] = None) -> BenchmarkResults:
    """
    Run a complete benchmark comparing brute-force and HNSW search.

    This executes a comprehensive benchmark that:
    1. Generates synthetic dataset
    2. Builds both brute-force and HNSW indexes
    3. Measures build time and memory usage
    4. Runs queries and measures search latency
    5. Calculates recall@k for HNSW vs brute-force

    Args:
        config: Benchmark configuration (uses defaults if None)

    Returns:
        BenchmarkResults object with all metrics

    Example:
        >>> config = BenchmarkConfig(dataset_size=10000, n_queries=100)
        >>> results = run_benchmark(config)
        >>> print(f"HNSW is {results.speedup:.1f}x faster")
    """
    if config is None:
        config = BenchmarkConfig()

    logger.info("=" * 80)
    logger.info("Starting benchmark")
    logger.info("=" * 80)
    logger.info(f"Dataset size: {config.dataset_size}")
    logger.info(f"Queries: {config.n_queries}")
    logger.info(f"Dimension: {config.dimension}")
    logger.info(f"k values: {config.k_values}")
    logger.info(f"HNSW ef_search values: {config.hnsw_ef_search_values}")

    results = BenchmarkResults(config=config)

    # Step 1: Generate dataset
    logger.info("\nStep 1: Generating dataset...")
    vectors = generate_random_vectors(
        n=config.dataset_size,
        dimension=config.dimension,
        normalize=config.normalize,
        seed=config.seed
    )
    logger.info(f"Generated {len(vectors)} vectors")

    # Step 2: Generate queries
    logger.info("\nStep 2: Generating queries...")

    if config.use_realistic_queries:
        # Generate realistic queries from the dataset with noise
        # This simulates real-world scenarios where queries are semantically similar
        # to indexed documents but not identical (e.g., user questions vs doc chunks)
        logger.info(f"  Using realistic queries (from dataset with {config.query_noise_level:.1%} noise)")
        queries = generate_queries(
            n_queries=config.n_queries,
            dimension=config.dimension,
            normalize=config.normalize,
            seed=config.seed + 1 if config.seed else None,
            from_dataset=vectors,  # Generate from same distribution
            noise_level=config.query_noise_level
        )
    else:
        # Generate completely random queries (stress test / adversarial)
        logger.info("  Using random queries (stress test mode)")
        queries = generate_queries(
            n_queries=config.n_queries,
            dimension=config.dimension,
            normalize=config.normalize,
            seed=config.seed + 1 if config.seed else None,
            from_dataset=None  # Completely random
        )

    logger.info(f"Generated {len(queries)} queries")

    # Step 3: Build brute-force index
    logger.info("\nStep 3: Building brute-force index...")
    start_time = time.time()

    bf_store = BruteForceVectorStore(
        dimension=config.dimension,
        similarity_metric=config.similarity_metric,
        normalized=config.normalize
    )

    for i, vector in enumerate(vectors):
        bf_store.add(vector, metadata={"index": i})

    results.brute_force_build_time = time.time() - start_time
    results.brute_force_memory_mb = bf_store.statistics()["memory_mb"]

    logger.info(
        f"Brute-force index built in {results.brute_force_build_time:.3f}s, "
        f"memory: {results.brute_force_memory_mb:.2f} MB"
    )

    # Step 4: Build HNSW index
    logger.info("\nStep 4: Building HNSW index...")
    start_time = time.time()

    hnsw_store = create_hnsw_index(
        dimension=config.dimension,
        m=config.hnsw_m,
        ef_construction=config.hnsw_ef_construction,
        similarity_metric=config.similarity_metric,
        normalized=config.normalize,
        seed=config.seed
    )

    for i, vector in enumerate(vectors):
        hnsw_store.insert(vector, metadata={"index": i})

    results.hnsw_build_time = time.time() - start_time
    results.hnsw_memory_mb = hnsw_store.statistics()["memory_mb"]

    logger.info(
        f"HNSW index built in {results.hnsw_build_time:.3f}s, "
        f"memory: {results.hnsw_memory_mb:.2f} MB"
    )

    # Step 5: Run searches and measure recall + latency
    logger.info("\nStep 5: Running searches and measuring performance...")

    for k in config.k_values:
        logger.info(f"\n  Testing k={k}...")

        # Get brute-force ground truth results
        logger.info(f"    Running brute-force search (ground truth)...")
        bf_start = time.time()
        bf_all_results = []

        for query in queries:
            bf_results = bf_store.search(query, k=k)
            # Extract indices for recall calculation
            bf_indices = [r["metadata"]["index"] for r in bf_results]
            bf_all_results.append(bf_indices)

        bf_latency = (time.time() - bf_start) / len(queries)  # Average per query

        # Store brute-force latency
        if k not in results.latency_results:
            results.latency_results[k] = {}
        results.latency_results[k]["brute_force"] = {
            "mean_ms": bf_latency * 1000,  # Convert to milliseconds
            "total_s": bf_latency * len(queries),
            "qps": 1.0 / bf_latency if bf_latency > 0 else 0.0,
        }

        # Test different ef_search values for HNSW
        for ef_search in config.hnsw_ef_search_values:
            logger.info(f"    Running HNSW search (ef_search={ef_search})...")
            hnsw_start = time.time()
            hnsw_all_results = []

            for query in queries:
                hnsw_results = hnsw_store.search(query, k=k, ef_search=ef_search)
                # Extract indices for recall calculation
                hnsw_indices = [r["metadata"]["index"] for r in hnsw_results]
                hnsw_all_results.append(hnsw_indices)

            hnsw_latency = (time.time() - hnsw_start) / len(queries)

            # Calculate recall
            mean_recall = calculate_mean_recall_at_k(
                all_ground_truth=bf_all_results,
                all_predictions=hnsw_all_results,
                k=k
            )

            recall_stats = calculate_recall_statistics(
                all_ground_truth=bf_all_results,
                all_predictions=hnsw_all_results,
                k=k
            )

            # Store results
            key = f"k={k},ef={ef_search}"
            results.recall_results[key] = recall_stats

            if "hnsw" not in results.latency_results[k]:
                results.latency_results[k]["hnsw"] = {}

            results.latency_results[k]["hnsw"][ef_search] = {
                "mean_ms": hnsw_latency * 1000,
                "total_s": hnsw_latency * len(queries),
                "qps": 1.0 / hnsw_latency if hnsw_latency > 0 else 0.0,
                "speedup": bf_latency / hnsw_latency if hnsw_latency > 0 else 0.0,
                "recall": recall_stats["mean"],
            }

            logger.info(
                f"      Recall@{k}: {recall_stats['mean']:.4f}, "
                f"Latency: {hnsw_latency*1000:.3f}ms, "
                f"Speedup: {bf_latency/hnsw_latency:.1f}x"
            )

    logger.info("\n" + "=" * 80)
    logger.info("Benchmark complete!")
    logger.info("=" * 80)

    return results


def compare_algorithms(
    dimension: int = 384,
    dataset_sizes: List[int] = [100, 1000, 10000],
    n_queries: int = 100,
    k: int = 5,
    **kwargs
) -> Dict[int, BenchmarkResults]:
    """
    Compare brute-force and HNSW across different dataset sizes.

    Args:
        dimension: Vector dimensionality
        dataset_sizes: List of dataset sizes to test
        n_queries: Number of queries per test
        k: Number of results to retrieve
        **kwargs: Additional arguments passed to BenchmarkConfig

    Returns:
        Dictionary mapping dataset_size -> BenchmarkResults

    Example:
        >>> results = compare_algorithms(
        ...     dimension=384,
        ...     dataset_sizes=[1000, 5000, 10000],
        ...     n_queries=100
        ... )
        >>> for size, result in results.items():
        ...     print(f"Size {size}: recall={result.recall:.3f}")
    """
    logger.info(f"Comparing algorithms across dataset sizes: {dataset_sizes}")

    all_results = {}

    for size in dataset_sizes:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing dataset size: {size}")
        logger.info(f"{'='*80}")

        config = BenchmarkConfig(
            dimension=dimension,
            dataset_size=size,
            n_queries=n_queries,
            k_values=[k],
            **kwargs
        )

        results = run_benchmark(config)
        all_results[size] = results

    return all_results
