"""
Benchmarking Module

Author: Nishit Attrey


This module provides comprehensive benchmarking capabilities:
- Recall@k evaluation (ANN vs exact search)
- Search latency measurements
- Index build time comparison
- Memory usage analysis
- Performance comparison reports

Key Components:
    - data: Synthetic dataset generation for benchmarking
    - metrics: Recall, precision, and other evaluation metrics
    - runner: Main benchmark execution framework
    - report: Results formatting and visualization

Example:
    >>> from src.benchmarks import run_benchmark, print_report
    >>> results = run_benchmark(dimension=384, dataset_size=1000)
    >>> print_report(results)
"""

from .data import (
    generate_random_vectors,
    generate_queries,
)

from .metrics import (
    calculate_recall_at_k,
    calculate_precision_at_k,
    calculate_intersection,
)

from .runner import (
    run_benchmark,
    compare_algorithms,
    BenchmarkConfig,
    BenchmarkResults,
)

from .report import (
    print_report,
    format_results_table,
    export_to_json,
    print_comparison_summary,
)

__all__ = [
    # Data generation
    "generate_random_vectors",
    "generate_queries",
    # Metrics
    "calculate_recall_at_k",
    "calculate_precision_at_k",
    "calculate_intersection",
    # Runner
    "run_benchmark",
    "compare_algorithms",
    "BenchmarkConfig",
    "BenchmarkResults",
    # Reporting
    "print_report",
    "format_results_table",
    "export_to_json",
    "print_comparison_summary",
]
