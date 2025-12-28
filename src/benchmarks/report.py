"""
Benchmark Reporting

Author: Nishit Attrey

This module provides utilities for formatting and displaying benchmark results.

Functions:
    print_report: Display results to console
    format_results_table: Format results as a table
    export_to_json: Save results to JSON file
"""

import json
from typing import Dict, Any, List
from pathlib import Path
import logging

from .runner import BenchmarkResults

logger = logging.getLogger(__name__)


def print_report(results: BenchmarkResults, verbose: bool = True) -> None:
    """
    Print a formatted benchmark report to console.

    Args:
        results: Benchmark results to display
        verbose: If True, show detailed statistics

    Example:
        >>> results = run_benchmark()
        >>> print_report(results)
    """
    config = results.config

    print("\n" + "=" * 80)
    print(" " * 25 + "BENCHMARK RESULTS")
    print("=" * 80)

    # Configuration summary
    print("\n[ Configuration ]")
    print(f"  Dataset size:    {config.dataset_size:,} vectors")
    print(f"  Queries:         {config.n_queries:,}")
    print(f"  Query mode:      {'Realistic (from dataset)' if config.use_realistic_queries else 'Random (stress test)'}")
    if config.use_realistic_queries:
        print(f"  Noise level:     {config.query_noise_level:.1%}")
    print(f"  Dimension:       {config.dimension}")
    print(f"  Metric:          {config.similarity_metric}")
    print(f"  k values:        {config.k_values}")
    print(f"  HNSW m:          {config.hnsw_m}")
    print(f"  HNSW ef_constr:  {config.hnsw_ef_construction}")
    print(f"  HNSW ef_search:  {config.hnsw_ef_search_values}")

    # Build time comparison
    print("\n[ Index Build Time ]")
    print(f"  Brute-force:     {results.brute_force_build_time:.3f}s")
    print(f"  HNSW:            {results.hnsw_build_time:.3f}s")
    if results.hnsw_build_time > 0:
        ratio = results.brute_force_build_time / results.hnsw_build_time
        print(f"  Ratio (BF/HNSW): {ratio:.2f}x")

    # Memory usage
    print("\n[ Memory Usage ]")
    print(f"  Brute-force:     {results.brute_force_memory_mb:.2f} MB")
    print(f"  HNSW:            {results.hnsw_memory_mb:.2f} MB")
    if results.brute_force_memory_mb > 0:
        ratio = results.hnsw_memory_mb / results.brute_force_memory_mb
        print(f"  Ratio (HNSW/BF): {ratio:.2f}x")

    # Search performance
    print("\n[ Search Performance ]")
    print(f"{'':4}{'k':>5}  {'ef_search':>10}  {'Recall@k':>10}  "
          f"{'Latency (ms)':>13}  {'QPS':>10}  {'Speedup':>8}")
    print("  " + "-" * 76)

    # Brute-force baseline
    for k in sorted(results.latency_results.keys()):
        bf_stats = results.latency_results[k].get("brute_force", {})
        if bf_stats:
            print(
                f"  BF{k:>5}  {'N/A':>10}  {'1.0000':>10}  "
                f"{bf_stats['mean_ms']:>13.3f}  "
                f"{bf_stats['qps']:>10.1f}  {'1.0x':>8}"
            )

            # HNSW results for this k
            hnsw_stats = results.latency_results[k].get("hnsw", {})
            for ef_search in sorted(hnsw_stats.keys()):
                stats = hnsw_stats[ef_search]
                print(
                    f"  HN{k:>5}  {ef_search:>10}  "
                    f"{stats['recall']:>10.4f}  "
                    f"{stats['mean_ms']:>13.3f}  "
                    f"{stats['qps']:>10.1f}  "
                    f"{stats['speedup']:>7.1f}x"
                )

    # Recall statistics (if verbose)
    if verbose and results.recall_results:
        print("\n[ Detailed Recall Statistics ]")
        print(f"{'':4}{'k':>5}  {'ef_search':>10}  {'Mean':>8}  "
              f"{'Std':>8}  {'Min':>8}  {'Max':>8}  {'Median':>8}")
        print("  " + "-" * 76)

        for key, stats in sorted(results.recall_results.items()):
            # Parse key "k=5,ef=50"
            parts = key.split(",")
            k_val = parts[0].split("=")[1]
            ef_val = parts[1].split("=")[1]

            print(
                f"    {k_val:>5}  {ef_val:>10}  "
                f"{stats['mean']:>8.4f}  "
                f"{stats['std']:>8.4f}  "
                f"{stats['min']:>8.4f}  "
                f"{stats['max']:>8.4f}  "
                f"{stats['median']:>8.4f}"
            )

    print("\n" + "=" * 80)
    print()


def format_results_table(
    results: BenchmarkResults,
    format: str = "markdown"
) -> str:
    """
    Format results as a table string.

    Args:
        results: Benchmark results
        format: Output format ("markdown" or "csv")

    Returns:
        Formatted table as a string

    Example:
        >>> table = format_results_table(results, format="markdown")
        >>> print(table)
    """
    if format == "markdown":
        return _format_markdown_table(results)
    elif format == "csv":
        return _format_csv_table(results)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _format_markdown_table(results: BenchmarkResults) -> str:
    """Format results as a Markdown table."""
    lines = []
    lines.append("## Benchmark Results\n")
    lines.append(f"Dataset: {results.config.dataset_size:,} vectors, "
                 f"Queries: {results.config.n_queries:,}\n")

    lines.append("\n### Search Performance\n")
    lines.append("| Algorithm | k | ef_search | Recall@k | Latency (ms) | QPS | Speedup |")
    lines.append("|-----------|---|-----------|----------|--------------|-----|---------|")

    for k in sorted(results.latency_results.keys()):
        # Brute-force
        bf_stats = results.latency_results[k].get("brute_force", {})
        if bf_stats:
            lines.append(
                f"| Brute-force | {k} | N/A | 1.0000 | "
                f"{bf_stats['mean_ms']:.3f} | {bf_stats['qps']:.1f} | 1.0x |"
            )

        # HNSW
        hnsw_stats = results.latency_results[k].get("hnsw", {})
        for ef_search in sorted(hnsw_stats.keys()):
            stats = hnsw_stats[ef_search]
            lines.append(
                f"| HNSW | {k} | {ef_search} | {stats['recall']:.4f} | "
                f"{stats['mean_ms']:.3f} | {stats['qps']:.1f} | "
                f"{stats['speedup']:.1f}x |"
            )

    return "\n".join(lines)


def _format_csv_table(results: BenchmarkResults) -> str:
    """Format results as CSV."""
    lines = []
    lines.append("algorithm,k,ef_search,recall,latency_ms,qps,speedup")

    for k in sorted(results.latency_results.keys()):
        # Brute-force
        bf_stats = results.latency_results[k].get("brute_force", {})
        if bf_stats:
            lines.append(
                f"brute_force,{k},N/A,1.0000,"
                f"{bf_stats['mean_ms']:.3f},{bf_stats['qps']:.1f},1.0"
            )

        # HNSW
        hnsw_stats = results.latency_results[k].get("hnsw", {})
        for ef_search in sorted(hnsw_stats.keys()):
            stats = hnsw_stats[ef_search]
            lines.append(
                f"hnsw,{k},{ef_search},{stats['recall']:.4f},"
                f"{stats['mean_ms']:.3f},{stats['qps']:.1f},"
                f"{stats['speedup']:.1f}"
            )

    return "\n".join(lines)


def export_to_json(
    results: BenchmarkResults,
    filepath: Path,
    indent: int = 2
) -> None:
    """
    Export benchmark results to JSON file.

    Args:
        results: Benchmark results to export
        filepath: Output file path
        indent: JSON indentation (default: 2)

    Example:
        >>> export_to_json(results, Path("benchmark_results.json"))
    """
    filepath = Path(filepath)

    # Create parent directory if needed
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict and write
    data = results.to_dict()

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)

    logger.info(f"Exported benchmark results to: {filepath}")
    print(f"\n✓ Results exported to: {filepath}")


def print_comparison_summary(
    all_results: Dict[int, BenchmarkResults],
    k: int = 5
) -> None:
    """
    Print a summary comparing results across different dataset sizes.

    Args:
        all_results: Dict mapping dataset_size -> BenchmarkResults
        k: The k value to focus on for comparison

    Example:
        >>> results = compare_algorithms(dataset_sizes=[1000, 5000, 10000])
        >>> print_comparison_summary(results, k=5)
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "SCALABILITY COMPARISON")
    print("=" * 80)

    print(f"\n{'Dataset Size':>12}  {'BF Build (s)':>12}  {'HNSW Build (s)':>14}  "
          f"{'BF Search (ms)':>15}  {'HNSW Search (ms)':>16}  {'Speedup':>8}")
    print("-" * 80)

    for size in sorted(all_results.keys()):
        result = all_results[size]

        bf_build = result.brute_force_build_time
        hnsw_build = result.hnsw_build_time

        # Get search times for k
        bf_latency = result.latency_results.get(k, {}).get("brute_force", {}).get("mean_ms", 0)

        # Use first ef_search value for HNSW
        hnsw_dict = result.latency_results.get(k, {}).get("hnsw", {})
        if hnsw_dict:
            first_ef = sorted(hnsw_dict.keys())[0]
            hnsw_latency = hnsw_dict[first_ef]["mean_ms"]
            speedup = hnsw_dict[first_ef]["speedup"]
        else:
            hnsw_latency = 0
            speedup = 0

        print(
            f"{size:>12,}  {bf_build:>12.3f}  {hnsw_build:>14.3f}  "
            f"{bf_latency:>15.3f}  {hnsw_latency:>16.3f}  {speedup:>7.1f}x"
        )

    print("=" * 80)
    print()
