"""
CLI command implementations for the RAG system.

This module contains all the command functions that implement the CLI functionality.
"""

import sys
from pathlib import Path
from typing import Any, Dict

from src.config import get_project_info
from src.ingestion import chunk_statistics, chunk_text, load_document
from src.embeddings import embed_texts, get_model_dimension, load_embedding_model
from src.vectorstore import BruteForceVectorStore, cosine_similarity, create_hnsw_index
from src.benchmarks import (
    run_benchmark,
    compare_algorithms,
    BenchmarkConfig,
    print_report,
    print_comparison_summary,
    export_to_json,
)
from src.collection import create_collection, load_collection, list_collections, delete_collection


def cmd_preview(args, config: Dict[str, Any], logger) -> int:
    """Preview chunks from a document file."""
    logger.info(f"Preview command for: {args.file}")

    print(f"\n{'=' * 80}")
    print("Document Preview (Stage 2: Document Loading & Chunking)")
    print(f"{'=' * 80}\n")

    try:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"✗ ERROR: File not found: {file_path}", file=sys.stderr)
            return 1

        # Load document
        print(f"Loading document: {file_path}")
        text = load_document(file_path)

        if not text:
            print("✗ ERROR: Document is empty or could not be loaded", file=sys.stderr)
            return 1

        print(f"✓ Loaded {len(text):,} characters\n")

        # Chunk the text
        chunk_size = config["ingestion"]["chunk_size"]
        chunk_overlap = config["ingestion"]["chunk_overlap"]
        print(f"Chunking with size={chunk_size}, overlap={chunk_overlap}...")

        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        print(f"✓ Created {len(chunks)} chunks\n")

        # Statistics
        stats = chunk_statistics(chunks)
        print("Statistics:")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  Avg length: {stats['avg_length']:.1f} characters")
        print(f"  Min length: {stats['min_length']} characters")
        print(f"  Max length: {stats['max_length']} characters")
        print(f"  Std dev: {stats['std_length']:.1f} characters\n")

        # Preview chunks
        num_to_show = min(args.num_chunks, len(chunks))
        print(f"Preview of first {num_to_show} chunk(s):\n")

        for i, chunk in enumerate(chunks[:num_to_show], 1):
            preview = chunk[: args.max_preview_length]
            if len(chunk) > args.max_preview_length:
                preview += "..."
            print(f"Chunk {i} ({len(chunk)} chars):")
            print(f"  {preview}\n")

        print(f"{'=' * 80}\n")
        logger.info(f"Preview complete: {len(chunks)} chunks from {file_path}")
        return 0

    except Exception as e:
        logger.error(f"Preview failed: {e}")
        print(f"\n✗ Error: {e}\n", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def cmd_embed_demo(args, config: Dict[str, Any], logger) -> int:
    """Demonstrate embedding generation."""
    logger.info("Embed demo command")

    print(f"\n{'=' * 80}")
    print("Embedding Demo (Stage 3: Local Embedding Pipeline)")
    print(f"{'=' * 80}\n")

    try:
        # Load model
        model_name = config["embeddings"]["model"]
        print(f"Loading embedding model: {model_name}...")
        model = load_embedding_model(model_name)
        print("✓ Model loaded\n")

        # Get model info
        dimension = get_model_dimension(model)
        print(f"Model dimension: {dimension}\n")

        # Generate embedding
        text = args.text
        print(f"Generating embedding for:")
        print(f'  "{text}"\n')

        embeddings = embed_texts([text], model, normalize=config["embeddings"]["normalize"])
        embedding = embeddings[0]

        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding preview (first 10 values):")
        print(f"  {embedding[:10]}\n")

        print(f"Embedding statistics:")
        print(f"  Min: {embedding.min():.4f}")
        print(f"  Max: {embedding.max():.4f}")
        print(f"  Mean: {embedding.mean():.4f}")
        print(f"  Std: {embedding.std():.4f}\n")

        print(f"{'=' * 80}\n")
        logger.info("Embed demo complete")
        return 0

    except Exception as e:
        logger.error(f"Embed demo failed: {e}")
        print(f"\n✗ Error: {e}\n", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def cmd_index(args, config: Dict[str, Any], logger) -> int:
    """Create embeddings for all documents in a directory."""
    logger.info(f"Index command for: {args.directory}")

    print(f"\n{'=' * 80}")
    print("Collection Creation")
    print(f"{'=' * 80}\n")

    try:
        directory_path = Path(args.directory)
        if not directory_path.exists():
            print(f"✗ ERROR: Directory not found: {directory_path}", file=sys.stderr)
            return 1

        # Create collection
        print(f"Creating collection '{args.name}' from: {directory_path}")
        print(f"Algorithm: {args.algorithm}\n")

        collection = create_collection(
            name=args.name,
            documents_dir=directory_path,
            algorithm=args.algorithm,
            config=config,
            show_progress=True,
        )

        print(f"\n✓ Collection '{args.name}' created successfully!")

        # Get collection info
        info = collection.info()
        print(f"  Documents: {info['num_documents']}")
        print(f"  Chunks: {info['num_chunks']}")
        print(f"  Embeddings: {info['num_embeddings']}")
        print(f"  Algorithm: {info['algorithm']}\n")

        # Test query if requested
        if args.test_query:
            print(f"Running test query: '{args.test_query}'...\n")

            results = collection.search(query=args.test_query, k=3)

            print(f"Top 3 results:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Score: {result['score']:.4f}")
                print(f"   Chunk ID: {result['metadata']['chunk_id']}")
                preview = result["metadata"]["text"][:200]
                if len(result["metadata"]["text"]) > 200:
                    preview += "..."
                print(f"   Text: {preview}")

            print()

        print(f"{'=' * 80}\n")
        logger.info(f"Index complete: collection '{args.name}' with {info['num_embeddings']} embeddings")
        return 0

    except Exception as e:
        logger.error(f"Index failed: {e}")
        print(f"\n✗ Error: {e}\n", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def cmd_search(args, config: Dict[str, Any], logger) -> int:
    """Search an existing collection."""
    collection_name = args.collection
    query_text = args.query

    logger.info(f"Search in collection '{collection_name}': '{query_text}'")

    print(f"\n{'=' * 80}")
    print(f"Search: '{query_text}'")
    print(f"Collection: {collection_name}")
    print(f"{'=' * 80}\n")

    try:
        # Load the collection
        print(f"Loading collection '{collection_name}'...")
        collection = load_collection(collection_name, config=config)

        info = collection.info()
        print(f"✓ Collection loaded")
        print(f"  Documents: {info['num_documents']}")
        print(f"  Chunks: {info['num_chunks']}")
        print(f"  Embeddings: {info['num_embeddings']}")
        print(f"  Algorithm: {info['algorithm']}\n")

        # Perform search
        print(f"Searching...")
        logger.info(f"Executing search query: {query_text}")

        ef_search = args.ef_search if hasattr(args, "ef_search") and args.ef_search else None
        results = collection.search(query_text, k=args.top_k, ef_search=ef_search)

        # Filter by min_score if specified
        if args.min_score > 0:
            results = [r for r in results if r["score"] >= args.min_score]

        print(f"✓ Found {len(results)} results\n")

        if not results:
            print("No results found.\n")
        else:
            print(f"Search Results (Top {len(results)}):\n")

            for i, result in enumerate(results, 1):
                print(f"{i}. Score: {result['score']:.4f}")
                print(f"   Chunk ID: {result['metadata']['chunk_id']}")

                # Preview text
                text = result["metadata"]["text"]
                preview = text[:300]
                if len(text) > 300:
                    preview += "..."
                print(f"   Text: {preview}")
                print()

        # Export to JSON if requested
        if hasattr(args, "output") and args.output:
            import json

            output_data = {
                "query": query_text,
                "collection": collection_name,
                "num_results": len(results),
                "results": [
                    {
                        "score": r["score"],
                        "chunk_id": r["metadata"]["chunk_id"],
                        "text": r["metadata"]["text"],
                    }
                    for r in results
                ],
            }
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"✓ Results exported to {args.output}\n")

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
            print(f"  rag index <directory> --name <collection_name>", file=sys.stderr)
        return 1

    except Exception as e:
        logger.error(f"Search failed: {e}")
        print(f"\n✗ Error: {e}\n", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def cmd_list_collections(args, config: Dict[str, Any], logger) -> int:
    """List all collections."""
    logger.info("Listing all collections")

    print(f"\n{'=' * 80}")
    print("Collections")
    print(f"{'=' * 80}\n")

    try:
        collections = list_collections(config=config)

        if not collections:
            print("No collections found.")
            print("\nCreate a collection with:")
            print("  rag index <directory> --name <collection_name>")
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
            print(f'  rag search "your query" --collection <name>')

        print(f"{'=' * 80}\n")
        logger.info(f"Listed {len(collections)} collections")
        return 0

    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        print(f"\n✗ Error: {e}\n", file=sys.stderr)
        return 1


def cmd_benchmark(args, config: Dict[str, Any], logger) -> int:
    """Run performance benchmarks."""
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


def cmd_delete(args, config: Dict[str, Any], logger) -> int:
    """Delete a collection."""
    collection_name = args.collection
    logger.info(f"Delete collection: {collection_name}")

    print(f"\n{'=' * 80}")
    print(f"Delete Collection: {collection_name}")
    print(f"{'=' * 80}\n")

    try:
        # Check if collection exists
        collections = list_collections(config=config)
        collection_names = [c["name"] for c in collections]

        if collection_name not in collection_names:
            print(f"✗ ERROR: Collection '{collection_name}' not found", file=sys.stderr)
            print(f"\nAvailable collections:", file=sys.stderr)
            for name in collection_names:
                print(f"  - {name}", file=sys.stderr)
            return 1

        # Confirm deletion
        if not args.force:
            response = input(f"Are you sure you want to delete '{collection_name}'? (yes/no): ")
            if response.lower() not in ["yes", "y"]:
                print("Deletion cancelled.")
                return 0

        # Delete collection using built-in function
        delete_collection(collection_name, config=config)

        print(f"✓ Collection '{collection_name}' deleted successfully!\n")
        logger.info(f"Deleted collection: {collection_name}")
        return 0

    except Exception as e:
        logger.error(f"Delete failed: {e}")
        print(f"\n✗ Error: {e}\n", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def cmd_info(args, config: Dict[str, Any], logger) -> int:
    """Show information about a collection."""
    collection_name = args.collection
    logger.info(f"Info for collection: {collection_name}")

    print(f"\n{'=' * 80}")
    print(f"Collection Info: {collection_name}")
    print(f"{'=' * 80}\n")

    try:
        # Load collection
        collection = load_collection(collection_name, config=config)
        info = collection.info()

        print(f"Name: {info['name']}")
        print(f"Algorithm: {info['algorithm']}")
        print(f"Documents: {info['num_documents']}")
        print(f"Chunks: {info['num_chunks']}")
        print(f"Embeddings: {info['num_embeddings']}")
        print()

        # Show index statistics if available
        if info.get('index'):
            print("Index statistics:")
            for key, value in info['index'].items():
                print(f"  {key}: {value}")
            print()

        # Show document sources if available
        print(f"Data locations:")
        print(f"  Chunks: {collection.chunks_dir}")
        print(f"  Embeddings: {collection.embeddings_dir}")
        print(f"  Index: {collection.indexes_dir}")
        print()

        print(f"{'=' * 80}\n")
        logger.info(f"Info displayed for collection: {collection_name}")
        return 0

    except FileNotFoundError:
        logger.error(f"Collection '{collection_name}' not found")
        print(f"\n✗ ERROR: Collection '{collection_name}' not found", file=sys.stderr)
        print(f"\nAvailable collections:", file=sys.stderr)
        collections = list_collections(config=config)
        for coll in collections:
            print(f"  - {coll['name']}", file=sys.stderr)
        return 1

    except Exception as e:
        logger.error(f"Info command failed: {e}")
        print(f"\n✗ Error: {e}\n", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1
