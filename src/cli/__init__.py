"""
CLI entry point for the RAG system.

This module provides the main entry point for the `rag` command.
"""

import argparse
import sys
from pathlib import Path

from src.config import load_config, get_project_info
from src.logger import setup_logging
from src.cli.commands import (
    cmd_preview,
    cmd_embed_demo,
    cmd_index,
    cmd_search,
    cmd_list_collections,
    cmd_benchmark,
    cmd_delete,
    cmd_info,
)


def print_banner(project_name: str, version: str) -> None:
    """Print a nice banner to console."""
    banner_width = 60
    print("=" * banner_width)
    print(f"{project_name} v{version}".center(banner_width))
    print("=" * banner_width)


def validate_setup(config, logger):
    """Validate that the system is properly set up."""
    from src.logger import log_config_info

    print("\nValidating setup...\n")
    log_config_info(config, logger)

    # Check if data directories exist
    data_dir = Path(config["paths"]["data_dir"])
    print(f"\nData directory: {data_dir.absolute()}")
    print(f"  Exists: {data_dir.exists()}")

    if data_dir.exists():
        raw_dir = Path(config["paths"]["raw_dir"])
        processed_dir = Path(config["paths"]["processed_dir"])
        embeddings_dir = Path(config["paths"]["embeddings_dir"])

        print(f"  Raw documents: {raw_dir.exists()}")
        print(f"  Processed chunks: {processed_dir.exists()}")
        print(f"  Embeddings: {embeddings_dir.exists()}")

    print("\n✓ Setup validated!")
    print("\nNext steps:")
    print("  1. Create a collection: rag index <directory> --name <collection_name>")
    print('  2. Search: rag search "your query" --collection <collection_name>')
    print("  3. List collections: rag list")
    print()


def main() -> int:
    """
    Main CLI entry point.

    Returns:
        Exit code: 0 for success, 1 for failure
    """
    try:
        # Create argument parser
        parser = argparse.ArgumentParser(
            prog="rag",
            description="Local Vector RAG Database - A from-scratch vector search implementation",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Validate setup
  rag

  # Create a collection from documents
  rag index data/documents --name my_docs

  # Search a collection
  rag search "machine learning" --collection my_docs

  # List all collections
  rag list

  # Show collection info
  rag info my_docs

  # Delete a collection
  rag delete my_docs

  # Run benchmarks
  rag benchmark --dataset-size 1000

For more information, visit: https://github.com/yourusername/local-vector-rag
            """,
        )

        # Add version
        parser.add_argument(
            "--version", action="version", version="%(prog)s 0.1.0"
        )

        # Add subcommands
        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Preview command
        preview_parser = subparsers.add_parser(
            "preview",
            help="Preview chunks from a document file (supports: txt, pdf, docx, md)",
        )
        preview_parser.add_argument(
            "file", type=str, help="Path to document file (.txt, .pdf, .docx, .doc, .md)"
        )
        preview_parser.add_argument(
            "--num-chunks",
            type=int,
            default=3,
            help="Number of chunks to show (default: 3)",
        )
        preview_parser.add_argument(
            "--max-preview-length",
            type=int,
            default=500,
            help="Maximum characters to show per chunk (default: 500)",
        )

        # Embed demo command
        embed_parser = subparsers.add_parser(
            "embed-demo", help="Demonstrate embedding generation"
        )
        embed_parser.add_argument("text", type=str, help="Text to generate embedding for")

        # Index command - Create collection from directory
        index_parser = subparsers.add_parser(
            "index", help="Create a collection from documents in a directory"
        )
        index_parser.add_argument(
            "directory",
            type=str,
            nargs="?",
            default="data/raw/samples",
            help="Path to directory containing documents (default: data/raw/samples)",
        )
        index_parser.add_argument(
            "--name",
            type=str,
            default="my_collection",
            help="Name for the collection (default: my_collection)",
        )
        index_parser.add_argument(
            "--algorithm",
            type=str,
            choices=["brute_force", "hnsw"],
            default="hnsw",
            help="Vector search algorithm to use (default: hnsw)",
        )
        index_parser.add_argument(
            "--test-query",
            type=str,
            help="Optional test query to run after indexing",
        )

        # Search command
        search_parser = subparsers.add_parser("search", help="Search an existing collection")
        search_parser.add_argument("query", type=str, help="Query text to search for")
        search_parser.add_argument(
            "--collection",
            type=str,
            default="my_collection",
            help="Name of the collection to search (default: my_collection)",
        )
        search_parser.add_argument(
            "--top-k",
            type=int,
            default=5,
            help="Number of results to return (default: 5)",
        )
        search_parser.add_argument(
            "--min-score",
            type=float,
            default=0.0,
            help="Minimum similarity score threshold (default: 0.0)",
        )
        search_parser.add_argument(
            "--ef-search",
            type=int,
            help="HNSW ef_search parameter (higher = more accurate, slower)",
        )
        search_parser.add_argument(
            "--output",
            type=str,
            help="Export results to JSON file",
        )

        # List command
        list_parser = subparsers.add_parser("list", help="List all collections")

        # Info command
        info_parser = subparsers.add_parser("info", help="Show information about a collection")
        info_parser.add_argument("collection", type=str, help="Name of the collection")

        # Delete command
        delete_parser = subparsers.add_parser("delete", help="Delete a collection")
        delete_parser.add_argument("collection", type=str, help="Name of the collection to delete")
        delete_parser.add_argument(
            "--force", "-f", action="store_true", help="Skip confirmation prompt"
        )

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
        if args.command == "preview":
            return cmd_preview(args, config, logger)
        elif args.command == "embed-demo":
            return cmd_embed_demo(args, config, logger)
        elif args.command == "index":
            return cmd_index(args, config, logger)
        elif args.command == "search":
            return cmd_search(args, config, logger)
        elif args.command == "list":
            return cmd_list_collections(args, config, logger)
        elif args.command == "info":
            return cmd_info(args, config, logger)
        elif args.command == "delete":
            return cmd_delete(args, config, logger)
        elif args.command == "benchmark":
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
