#!/usr/bin/env python3
"""
Local Vector RAG Database - Legacy Entry Point

Author: Nishit Attrey

⚠️  NOTICE: This is a legacy entry point. For the best experience, use the CLI:

    # Activate virtual environment first
    source venv/bin/activate

    # Then use the 'rag' command from anywhere
    rag --help
    rag index ~/docs --name my_collection
    rag search "query" --collection my_collection
    rag generate "question" --collection my_collection
    rag chat --collection my_collection

This file is kept for backward compatibility and simple validation.

Usage:
    # Validate setup
    python main.py

    # Run benchmarks
    python main.py benchmark
"""

import argparse
import sys
from pathlib import Path

# Add src directory to Python path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.config import get_project_info, load_config
from src.logger import setup_logging
from src.benchmarks import (
    run_benchmark,
    compare_algorithms,
    BenchmarkConfig,
    print_report,
    print_comparison_summary,
    export_to_json,
)


def print_banner(project_name: str, version: str) -> None:
    """Print a nice banner to console."""
    banner_width = 60
    print("=" * banner_width)
    print(f"{project_name} v{version}".center(banner_width))
    print("=" * banner_width)


def validate_setup(config, logger):
    """
    Validate that the system is properly set up.

    Checks configuration, logging, and directory structure.
    """
    logger.info("=" * 60)
    logger.info("Validating system setup...")
    logger.info("=" * 60)
    logger.info("✓ Configuration loaded successfully")
    logger.info(f"✓ Data directory: {config['paths']['data_dir']}")
    logger.info(f"✓ Logs directory: {config['paths']['logs_dir']}")
    logger.info(f"✓ Log level: {config['logging']['level']}")

    print(f"\n{'=' * 60}")
    print("✓ System validation successful!")
    print(f"{'=' * 60}")
    print("\n⚠️  IMPORTANT: Use the 'rag' CLI for all operations:")
    print("\n  # First, activate your virtual environment")
    print("  source venv/bin/activate\n")
    print("  # Available commands:")
    print("  rag --help                          # Show all commands")
    print("  rag index ~/docs --name my_docs     # Index documents")
    print("  rag search 'query' --collection my_docs  # Search")
    print("  rag generate 'question' --collection my_docs  # Generate answers")
    print("  rag chat --collection my_docs       # Interactive chat")
    print("  rag list                            # List collections")
    print("  rag benchmark                       # Run benchmarks")
    print(f"\n{'=' * 60}")
    print("\nFor detailed documentation, see:")
    print("  - README.md")
    print("  - src/cli/USER_GUIDE.md")
    print(f"{'=' * 60}\n")


def cmd_benchmark(args, config, logger):
    """
    Run performance benchmarks comparing brute-force and HNSW search.

    This command executes comprehensive benchmarks that measure:
    - Recall@k (accuracy of HNSW vs exact search)
    - Search latency (speed comparison)
    - Index build time
    - Memory usage
    - Scalability across dataset sizes

    Args:
        args: Parsed command-line arguments
        config: Configuration dictionary
        logger: Logger instance
    """
    logger.info("Starting benchmark execution")

    print(f"\n{'=' * 80}")
    print("Benchmark Execution")
    print(f"{'=' * 80}\n")

    try:
        if args.compare_sizes:
            # Compare across multiple dataset sizes
            logger.info("Running scalability comparison across dataset sizes")

            print("Running scalability comparison...")
            print(f"Dataset sizes: {args.dataset_sizes}")
            print(f"Queries per size: {args.n_queries}")
            print(f"This may take several minutes...\n")

            results = compare_algorithms(
                dimension=config["embeddings"]["dimension"],
                dataset_sizes=args.dataset_sizes,
                n_queries=args.n_queries,
                k=args.k,
                similarity_metric=config["vectorstore"]["similarity_metric"],
                normalize=config["embeddings"]["normalize"],
                hnsw_m=config["vectorstore"]["hnsw"]["m"],
                hnsw_ef_construction=config["vectorstore"]["hnsw"]["ef_construction"],
                hnsw_ef_search_values=args.ef_search_values,
                seed=args.seed,
            )

            # Print comparison summary
            print_comparison_summary(results, k=args.k)

            # Export if requested
            if args.output:
                output_path = Path(args.output)
                for size, result in results.items():
                    output_file = (
                        output_path.parent / f"{output_path.stem}_{size}{output_path.suffix}"
                    )
                    export_to_json(result, output_file)

        else:
            # Single benchmark run
            logger.info("Running single benchmark")

            bench_config = BenchmarkConfig(
                dimension=config["embeddings"]["dimension"],
                dataset_size=args.dataset_size,
                n_queries=args.n_queries,
                k_values=args.k_values,
                similarity_metric=config["vectorstore"]["similarity_metric"],
                normalize=config["embeddings"]["normalize"],
                hnsw_m=config["vectorstore"]["hnsw"]["m"],
                hnsw_ef_construction=config["vectorstore"]["hnsw"]["ef_construction"],
                hnsw_ef_search_values=args.ef_search_values,
                seed=args.seed,
            )

            print(f"Running benchmark with:")
            print(f"  Dataset size: {args.dataset_size:,} vectors")
            print(f"  Queries: {args.n_queries}")
            print(f"  k values: {args.k_values}")
            print(f"  ef_search values: {args.ef_search_values}")
            print(f"\nThis may take a few minutes...\n")

            results = run_benchmark(bench_config)

            # Print report
            print_report(results, verbose=args.verbose)

            # Export if requested
            if args.output:
                export_to_json(results, Path(args.output))

        print("✓ Benchmark complete!\n")
        logger.info("Benchmark execution complete")

        return 0

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        print(f"\n✗ Error: {e}\n", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def main() -> int:
    """
    Main application entry point with CLI argument parsing.

    Returns:
        Exit code: 0 for success, 1 for failure
    """
    try:
        # Create argument parser
        parser = argparse.ArgumentParser(
            description="Local Vector RAG Database - Legacy Entry Point (Use 'rag' CLI instead)",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Validate setup
  python main.py

  # Run benchmarks
  python main.py benchmark --dataset-size 1000

⚠️  For all other operations, use the 'rag' CLI:
  rag --help
  rag index ~/docs --name my_collection
  rag search "query" --collection my_collection
            """,
        )

        # Add subcommands
        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Benchmark command
        benchmark_parser = subparsers.add_parser(
            "benchmark", help="Run performance benchmarks"
        )
        benchmark_parser.add_argument(
            "--dataset-size",
            type=int,
            default=1000,
            help="Number of vectors to index (default: 1000)",
        )
        benchmark_parser.add_argument(
            "--n-queries",
            type=int,
            default=100,
            help="Number of queries to run (default: 100)",
        )
        benchmark_parser.add_argument(
            "--k-values",
            type=int,
            nargs="+",
            default=[1, 5, 10],
            help="List of k values to test (default: 1 5 10)",
        )
        benchmark_parser.add_argument(
            "--ef-search-values",
            type=int,
            nargs="+",
            default=[10, 50, 100],
            help="List of ef_search values for HNSW (default: 10 50 100)",
        )
        benchmark_parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed for reproducibility (default: 42)",
        )
        benchmark_parser.add_argument(
            "--verbose",
            action="store_true",
            help="Show detailed statistics in report",
        )
        benchmark_parser.add_argument(
            "--output",
            type=str,
            help="Export results to JSON file",
        )
        benchmark_parser.add_argument(
            "--compare-sizes",
            action="store_true",
            help="Compare performance across multiple dataset sizes",
        )
        benchmark_parser.add_argument(
            "--dataset-sizes",
            type=int,
            nargs="+",
            default=[100, 1000, 5000],
            help="Dataset sizes for comparison mode (default: 100 1000 5000)",
        )
        benchmark_parser.add_argument(
            "--k",
            type=int,
            default=5,
            help="k value for comparison mode (default: 5)",
        )

        # Parse arguments
        args = parser.parse_args()

        # Load configuration
        config = load_config()

        # Extract project info
        project_name, version = get_project_info(config)

        # Print banner
        print_banner(project_name, version)

        # Setup logging
        logger = setup_logging(config)

        # Route to appropriate command
        if args.command == "benchmark":
            return cmd_benchmark(args, config, logger)
        else:
            # No command specified - validate setup
            validate_setup(config, logger)
            return 0

    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("\nPlease ensure config.yaml exists in the project root.", file=sys.stderr)
        return 1

    except ValueError as e:
        print(f"ERROR: Configuration validation failed", file=sys.stderr)
        print(f"{e}", file=sys.stderr)
        return 1

    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        return 130

    except Exception as e:
        print(f"FATAL ERROR: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
