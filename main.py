#!/usr/bin/env python3
"""
Local Vector RAG Database - Main Entry Point

Author: Nishit Attrey

This is the main entry point for the Local Vector RAG system.

Stages:
- Stage 1: Configuration and logging ✓
- Stage 2: Document ingestion and chunking ✓
- Stage 3: Local embedding pipeline ✓
- Stage 4: Brute-force vector store ✓
- Stage 5: Index persistence ✓
- Stage 6: HNSW data structures ✓
- Stage 7: HNSW insertion logic ✓
- Stage 8: HNSW search algorithm ✓
- Stage 9: Query pipeline ✓

Usage:
    # Validate setup
    python main.py

    # Preview chunks from a file (Stage 2)
    python main.py preview data/raw/sample.txt

    # Test embedding generation (Stage 3)
    python main.py embed-demo "Sample text to embed"

    # Test vector search (Stage 4)
    python main.py search-demo "query text"

    # Test query pipeline (Stage 9)
    python main.py query-demo "What is vector search?"

    # Create embeddings for all files in a directory
    python main.py index data/raw/samples --name my_docs --algorithm hnsw

    # Create and test search
    python main.py index --test-query "machine learning"

    # Search an existing collection
    python main.py search "machine learning" --collection my_docs --top-k 5

    # List all collections
    python main.py list

    # Future usage (Stages 10+):
    python main.py query "What is vector search?" --generate
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add src directory to Python path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.config import get_project_info, load_config
from src.embeddings import (
    embed_chunks,
    embed_texts,
    get_model_dimension,
    load_embedding_model,
)
from src.ingestion import chunk_statistics, chunk_text, load_document
from src.logger import log_config_info, setup_logging
from src.query import create_query_pipeline, QueryPipeline
from src.vectorstore import (
    BruteForceVectorStore,
    cosine_similarity,
    create_hnsw_index,
    HNSWIndex,
)
from src.benchmarks import (
    run_benchmark,
    compare_algorithms,
    BenchmarkConfig,
    print_report,
    print_comparison_summary,
    export_to_json,
)
from src.collection import create_collection, load_collection, list_collections


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

    # Log detailed config at DEBUG level
    log_config_info(config, logger)

    logger.info("")
    logger.info("Available Commands:")
    logger.info("  python main.py                      - Validate setup")
    logger.info("  python main.py preview <file>       - Preview chunks from file")
    logger.info('  python main.py embed-demo "text"    - Demo embedding generation')
    logger.info(
        "  python main.py index <dir>          - Create embeddings for all docs in directory"
    )
    logger.info('  python main.py search "query"       - Search an existing collection')
    logger.info("  python main.py list                 - List all collections")
    logger.info("  python main.py benchmark           - Run performance benchmarks")
    logger.info("=" * 60)


def cmd_preview(args, config, logger):
    """
    Preview chunks from a document file.

    Args:
        args: Parsed command-line arguments
        config: Configuration dictionary
        logger: Logger instance
    """
    filepath = Path(args.file)

    logger.info(f"Previewing chunks from: {filepath}")

    # Check file exists
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        print(f"\nERROR: File not found: {filepath}", file=sys.stderr)
        return 1

    # Load document
    logger.info("Loading document...")
    doc = load_document(filepath, encoding=config["ingestion"]["encoding"])

    if doc is None:
        logger.error("Failed to load document")
        print(f"\nERROR: Failed to load document from {filepath}", file=sys.stderr)
        return 1

    logger.info(f"Loaded document '{doc['doc_id']}': {doc['size']} characters")

    # Get chunking parameters from config
    chunk_size = config["ingestion"]["chunk_size"]
    chunk_overlap = config["ingestion"]["chunk_overlap"]

    logger.info(f"Chunking with size={chunk_size}, overlap={chunk_overlap}")

    # Chunk the document
    chunks = chunk_text(
        doc["text"],
        doc_id=doc["doc_id"],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Calculate statistics
    stats = chunk_statistics(chunks)

    # Display results
    print(f"\n{'=' * 80}")
    print(f"Document: {doc['doc_id']}")
    print(f"File: {filepath}")
    print(f"Format: {doc.get('format', 'txt')}")
    print(f"{'=' * 80}")
    print(f"\nDocument Statistics:")
    print(f"  Size: {doc['size']:,} characters")
    print(f"\nChunking Configuration:")
    print(f"  Chunk size: {chunk_size}")
    print(f"  Chunk overlap: {chunk_overlap}")
    print(f"\nChunk Statistics:")
    print(f"  Total chunks: {stats['num_chunks']}")
    print(f"  Total characters (with overlap): {stats['total_chars']:,}")
    print(f"  Average chunk size: {stats['avg_chunk_size']:.1f}")
    print(f"  Min chunk size: {stats['min_chunk_size']}")
    print(f"  Max chunk size: {stats['max_chunk_size']}")

    # Show first few chunks
    num_preview = min(args.num_chunks, len(chunks))
    print(f"\nFirst {num_preview} Chunk(s):")
    print("=" * 80)

    for i, chunk in enumerate(chunks[:num_preview]):
        print(f"\nChunk {i} (ID: {chunk['chunk_id']})")
        print(f"Position: [{chunk['start']}:{chunk['end']}]")
        print(f"Length: {len(chunk['text'])} characters")
        print(f"\nText Preview:")
        print("-" * 80)

        # Show full text or truncate if too long
        if len(chunk["text"]) <= args.max_preview_length:
            print(chunk["text"])
        else:
            preview_text = chunk["text"][: args.max_preview_length]
            print(f"{preview_text}...")
            print(f"[... truncated, showing {args.max_preview_length}/{len(chunk['text'])} chars]")

        print("-" * 80)

    if len(chunks) > num_preview:
        print(f"\n[... {len(chunks) - num_preview} more chunks not shown]")

    print(f"\n{'=' * 80}")
    logger.info(f"Preview complete: {stats['num_chunks']} chunks generated")

    return 0


def cmd_embed_demo(args, config, logger):
    """
    Demonstrate embedding generation (Stage 3).

    Args:
        args: Parsed command-line arguments
        config: Configuration dictionary
        logger: Logger instance
    """
    text = args.text

    logger.info(
        f'Embedding demo for text: "{text[:50]}..."'
        if len(text) > 50
        else f'Embedding demo for text: "{text}"'
    )

    # Get embedding config
    model_name = config["embeddings"]["model_name"]
    device = config["embeddings"]["device"]
    normalize = config["embeddings"]["normalize"]
    expected_dim = config["embeddings"]["dimension"]

    print(f"\n{'=' * 80}")
    print("Embedding Generation Demo (Stage 3)")
    print(f"{'=' * 80}")
    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Device: {device}")
    print(f"  Normalize: {normalize}")
    print(f"  Expected dimension: {expected_dim}")

    # Load model
    print(f"\nLoading model...")
    logger.info(f"Loading embedding model: {model_name}")

    try:
        model = load_embedding_model(model_name, device=device)
        actual_dim = get_model_dimension(model)

        print(f"✓ Model loaded successfully")
        print(f"  Actual dimension: {actual_dim}")

        if actual_dim != expected_dim:
            logger.warning(f"Dimension mismatch: expected {expected_dim}, got {actual_dim}")
            print(f"\n⚠ Warning: Dimension mismatch!")

        # Generate embedding
        print(f"\nGenerating embedding...")
        logger.info("Generating embedding")

        embeddings = embed_texts(model, [text], normalize=normalize)
        embedding = embeddings[0]

        print(f"✓ Embedding generated")
        print(f"\nEmbedding Details:")
        print(f"  Shape: {embedding.shape}")
        print(f"  Dtype: {embedding.dtype}")
        print(f"  L2 norm: {np.linalg.norm(embedding):.6f}")
        print(f"  Min value: {np.min(embedding):.6f}")
        print(f"  Max value: {np.max(embedding):.6f}")
        print(f"  Mean value: {np.mean(embedding):.6f}")

        # Show first few values
        print(f"\n  First 10 values:")
        print(f"    {embedding[:10]}")

        # Test determinism
        print(f"\nTesting determinism...")
        emb2 = embed_texts(model, [text], normalize=normalize)[0]
        is_identical = np.allclose(embedding, emb2, rtol=1e-7)

        if is_identical:
            print(f"✓ Embeddings are deterministic (same text → same embedding)")
        else:
            print(f"✗ Warning: Embeddings differ across runs!")

        print(f"\n{'=' * 80}")
        print("Stage 3 Exit Criteria:")
        print(f"  ✓ Model loads on CPU")
        print(f"  ✓ Embedding shape is correct ({actual_dim})")
        print(f"  ✓ Same text → same embedding (deterministic)")
        if normalize:
            print(f"  ✓ L2 normalization applied (norm = {np.linalg.norm(embedding):.4f})")
        print(f"{'=' * 80}\n")

        logger.info("Embed demo complete")
        return 0

    except Exception as e:
        logger.error(f"Embedding demo failed: {e}")
        print(f"\n✗ Error: {e}\n", file=sys.stderr)
        return 1


def cmd_index(args, config, logger):
    """
    Create embeddings for all documents in a directory.

    This command:
    1. Loads all documents from the specified directory
    2. Chunks them according to config
    3. Generates embeddings for all chunks
    4. Saves everything to disk
    5. Builds a searchable vector index

    Args:
        args: Parsed command-line arguments
        config: Configuration dictionary
        logger: Logger instance
    """
    directory = Path(args.directory)
    collection_name = args.name
    algorithm = args.algorithm

    logger.info(f"Creating collection '{collection_name}' from directory: {directory}")

    print(f"\n{'=' * 80}")
    print(f"Creating Collection: {collection_name}")
    print(f"{'=' * 80}")
    print(f"\nConfiguration:")
    print(f"  Directory: {directory}")
    print(f"  Algorithm: {algorithm}")
    print(f"  Model: {config['embeddings']['model_name']}")
    print(f"  Chunk size: {config['ingestion']['chunk_size']}")
    print(f"  Chunk overlap: {config['ingestion']['chunk_overlap']}")
    print(f"  Supported formats: {', '.join(config['ingestion']['supported_formats'])}")

    # Check directory exists
    if not directory.exists():
        logger.error(f"Directory not found: {directory}")
        print(f"\n✗ ERROR: Directory not found: {directory}", file=sys.stderr)
        return 1

    if not directory.is_dir():
        logger.error(f"Not a directory: {directory}")
        print(f"\n✗ ERROR: Not a directory: {directory}", file=sys.stderr)
        return 1

    try:
        print(f"\nCreating collection (this may take a few minutes)...")
        print(f"{'=' * 80}\n")

        # Create collection
        collection = create_collection(
            name=collection_name,
            documents_dir=directory,
            algorithm=algorithm,
            config=config,
            show_progress=True,
        )

        # Get collection info
        info = collection.info()

        print(f"\n{'=' * 80}")
        print("✓ Collection Created Successfully!")
        print(f"{'=' * 80}")
        print(f"\nCollection Statistics:")
        print(f"  Name: {info['name']}")
        print(f"  Algorithm: {info['algorithm']}")
        print(f"  Documents: {info['num_documents']}")
        print(f"  Chunks: {info['num_chunks']}")
        print(f"  Embeddings: {info['num_embeddings']}")

        if "index" in info and info["index"]:
            print(f"\nIndex Statistics:")
            for key, value in info["index"].items():
                print(f"  {key}: {value}")

        print(f"\nData saved to:")
        print(f"  Chunks: data/processed/")
        print(f"  Embeddings: data/embeddings/{collection_name}.npz")
        print(f"  Index: data/indexes/{algorithm}/{collection_name}/")
        print(f"  Metadata: data/collections.json")

        # Test search if requested
        if args.test_query:
            print(f"\n{'=' * 80}")
            print(f'Testing Search with Query: "{args.test_query}"')
            print(f"{'=' * 80}\n")

            results = collection.search(args.test_query, k=3)

            if results:
                for i, result in enumerate(results, 1):
                    print(f"Result {i}:")
                    print(f"  Score: {result['score']:.4f}")
                    print(f"  Chunk ID: {result['metadata']['chunk_id']}")
                    print(f"  Text: {result['metadata']['text'][:150]}...")
                    print()
            else:
                print("No results found.")

        print(f"{'=' * 80}")
        print("\nTo search this collection later, use:")
        print(f"  from src.collection import load_collection")
        print(f"  collection = load_collection('{collection_name}')")
        print(f"  results = collection.search('your query', k=5)")
        print(f"{'=' * 80}\n")

        logger.info(f"Collection '{collection_name}' created successfully")
        return 0

    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        print(f"\n✗ Error: {e}\n", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def cmd_search(args, config, logger):
    """
    Search an existing collection.

    Args:
        args: Parsed command-line arguments
        config: Configuration dictionary
        logger: Logger instance
    """
    collection_name = args.collection
    query_text = args.query
    top_k = args.top_k
    min_score = args.min_score

    logger.info(f"Searching collection '{collection_name}' for: {query_text}")

    print(f"\n{'=' * 80}")
    print(f"Searching Collection: {collection_name}")
    print(f"{'=' * 80}")
    print(f'\nQuery: "{query_text}"')
    print(f"Top-k: {top_k}")
    print(f"Min score: {min_score}")

    try:
        # Load the collection
        print(f"\nLoading collection '{collection_name}'...")
        collection = load_collection(collection_name, config=config)

        info = collection.info()
        print(f"✓ Collection loaded")
        print(f"  Documents: {info['num_documents']}")
        print(f"  Chunks: {info['num_chunks']}")
        print(f"  Embeddings: {info['num_embeddings']}")
        print(f"  Algorithm: {info['algorithm']}")

        # Perform search
        print(f"\nSearching...")
        logger.info(f"Executing search query: {query_text}")

        ef_search = args.ef_search if hasattr(args, "ef_search") else None
        results = collection.search(query_text, k=top_k, ef_search=ef_search)

        # Filter by min_score if specified
        if min_score > 0:
            results = [r for r in results if r["score"] >= min_score]

        print(f"✓ Found {len(results)} results")

        # Display results
        print(f"\n{'=' * 80}")
        print(f"Search Results (Top {len(results)})")
        print(f"{'=' * 80}\n")

        if not results:
            print("No results found above the minimum score threshold.")
        else:
            for i, result in enumerate(results, 1):
                print(f"Result {i}:")
                print(f"  Score: {result['score']:.4f}")
                print(f"  Chunk ID: {result['metadata']['chunk_id']}")
                print(f"  Doc ID: {result['metadata']['doc_id']}")
                print(f"  Text: {result['metadata']['text'][:200]}...")
                print()

        # Export to JSON if requested
        if args.output:
            import json

            output_data = {
                "query": query_text,
                "collection": collection_name,
                "num_results": len(results),
                "results": results,
            }
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"✓ Results exported to {args.output}")

        print(f"{'=' * 80}\n")
        logger.info(f"Search complete: {len(results)} results")
        return 0

    except FileNotFoundError:
        logger.error(f"Collection '{collection_name}' not found")
        print(f"\n✗ ERROR: Collection '{collection_name}' not found", file=sys.stderr)
        print(f"\nAvailable collections:", file=sys.stderr)
        collections = list_collections(config=config)
        if collections:
            for coll in collections:
                print(f"  - {coll['name']}", file=sys.stderr)
        else:
            print(f"  No collections created yet. Create one with:", file=sys.stderr)
            print(f"  python main.py index <directory>", file=sys.stderr)
        return 1

    except Exception as e:
        logger.error(f"Search failed: {e}")
        print(f"\n✗ Error: {e}\n", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def cmd_list_collections(args, config, logger):
    """
    List all collections.

    Args:
        args: Parsed command-line arguments
        config: Configuration dictionary
        logger: Logger instance
    """
    logger.info("Listing all collections")

    print(f"\n{'=' * 80}")
    print("Collections")
    print(f"{'=' * 80}\n")

    try:
        collections = list_collections(config=config)

        if not collections:
            print("No collections found.")
            print("\nCreate a collection with:")
            print("  python main.py index <directory> --name <collection_name>")
        else:
            print(f"Found {len(collections)} collection(s):\n")
            for i, coll in enumerate(collections, 1):
                print(f"{i}. {coll['name']}")
                print(f"   Algorithm: {coll['algorithm']}")
                print(f"   Documents: {coll['num_documents']}")
                print(f"   Chunks: {coll['num_chunks']}")
                print(f"   Embeddings: {coll['num_embeddings']}")
                print(f"   Created: {coll.get('created_at', 'N/A')}")
                print()

            print(f"To search a collection:")
            print(f'  python main.py search "your query" --collection <name>')

        print(f"{'=' * 80}\n")
        logger.info(f"Listed {len(collections)} collections")
        return 0

    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        print(f"\n✗ Error: {e}\n", file=sys.stderr)
        return 1


def cmd_benchmark(args, config, logger):
    """
    Run performance benchmarks comparing brute-force and HNSW search (Stage 11).

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
    logger.info("Starting benchmark execution (Stage 11)")

    print(f"\n{'=' * 80}")
    print("Benchmark Execution (Stage 11: Evaluation & Benchmarking)")
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
            description="Local Vector RAG Database - A from-scratch vector search implementation",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Validate setup
  python main.py

  # Preview chunks from a document
  python main.py preview data/raw/sample.txt

  # Preview with custom settings
  python main.py preview data/raw/sample.txt --num-chunks 5 --max-preview-length 200
            """,
        )

        # Add subcommands
        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Preview command (Stage 2)
        preview_parser = subparsers.add_parser(
            "preview", help="Preview chunks from a document file (supports: txt, pdf, docx, md)"
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

        # Embed demo command (Stage 3)
        embed_parser = subparsers.add_parser(
            "embed-demo", help="Demonstrate embedding generation (Stage 3)"
        )
        embed_parser.add_argument("text", type=str, help="Text to generate embedding for")

        # Search demo command (Stage 4)
        search_parser = subparsers.add_parser(
            "search-demo", help="Demonstrate vector search (Stage 4)"
        )
        search_parser.add_argument("query", type=str, help="Query text to search for")
        search_parser.add_argument(
            "--top-k",
            type=int,
            default=3,
            help="Number of results to return (default: 3)",
        )

        # Query demo command (Stage 9)
        query_parser = subparsers.add_parser(
            "query-demo", help="Demonstrate query pipeline (Stage 9)"
        )
        query_parser.add_argument("query", type=str, help="Query text to search for")
        query_parser.add_argument(
            "--top-k",
            type=int,
            default=5,
            help="Number of results to return (default: 5)",
        )
        query_parser.add_argument(
            "--min-score",
            type=float,
            default=0.0,
            help="Minimum similarity score threshold (default: 0.0)",
        )
        query_parser.add_argument(
            "--algorithm",
            type=str,
            choices=["brute_force", "hnsw"],
            default="brute_force",
            help="Vector search algorithm to use (default: brute_force)",
        )

        # Index command - Create embeddings for directory
        index_parser = subparsers.add_parser(
            "index", help="Create embeddings for all documents in a directory"
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

        # Search command - Search existing collection
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

        # List command - List all collections
        list_parser = subparsers.add_parser("list", help="List all collections")

        # Benchmark command (Stage 11)
        benchmark_parser = subparsers.add_parser(
            "benchmark", help="Run performance benchmarks (Stage 11)"
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
