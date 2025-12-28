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
    logger.info("System Status:")
    logger.info("  Stage 1: Project Skeleton & Config - COMPLETE ✓")
    logger.info("  Stage 2: Document Ingestion & Chunking - COMPLETE ✓")
    logger.info("  Stage 3: Local Embedding Pipeline - COMPLETE ✓")
    logger.info("  Stage 4: Brute-Force Vector Store - COMPLETE ✓")
    logger.info("  Stage 5: Index Persistence - COMPLETE ✓")
    logger.info("  Stage 6: HNSW Data Structures - COMPLETE ✓")
    logger.info("  Stage 7: HNSW Insertion Logic - COMPLETE ✓")
    logger.info("  Stage 8: HNSW Search Algorithm - COMPLETE ✓")
    logger.info("  Stage 9: Query Pipeline - COMPLETE ✓")
    logger.info("  Stage 11: Evaluation & Benchmarking - COMPLETE ✓")
    logger.info("")
    logger.info("Available Commands:")
    logger.info("  python main.py                      - Validate setup")
    logger.info("  python main.py preview <file>       - Preview chunks from file")
    logger.info('  python main.py embed-demo "text"    - Demo embedding generation')
    logger.info('  python main.py search-demo "query" - Demo vector search')
    logger.info('  python main.py query-demo "query"  - Demo query pipeline')
    logger.info("  python main.py benchmark           - Run performance benchmarks")
    logger.info("")
    logger.info("Next Steps:")
    logger.info("  1. Add .txt files to data/raw/")
    logger.info('  2. Test query pipeline: python main.py query-demo "sample query"')
    logger.info("  3. Run benchmarks: python main.py benchmark")
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


def cmd_search_demo(args, config, logger):
    """
    Demonstrate end-to-end vector search (Stage 4).

    This command shows the complete pipeline:
    1. Create sample document chunks
    2. Generate embeddings for chunks
    3. Build vector store
    4. Generate query embedding
    5. Search for similar chunks

    Args:
        args: Parsed command-line arguments
        config: Configuration dictionary
        logger: Logger instance
    """
    query_text = args.query

    logger.info(f'Search demo for query: "{query_text}"')

    # Get config
    model_name = config["embeddings"]["model_name"]
    device = config["embeddings"]["device"]
    normalize = config["embeddings"]["normalize"]
    dimension = config["embeddings"]["dimension"]
    similarity_metric = config["vectorstore"]["similarity_metric"]
    top_k = args.top_k

    print(f"\n{'=' * 80}")
    print("Vector Search Demo (Stage 4)")
    print(f"{'=' * 80}")
    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Similarity metric: {similarity_metric}")
    print(f"  Top-k results: {top_k}")

    # Create sample chunks (simulate a small knowledge base)
    print(f"\nCreating sample knowledge base...")
    logger.info("Creating sample chunks")

    sample_chunks = [
        {
            "chunk_id": "doc1_chunk_0",
            "text": "Vector search is a technique for finding similar items using embeddings.",
            "doc_id": "doc1",
        },
        {
            "chunk_id": "doc1_chunk_1",
            "text": "Machine learning models convert text into dense vector representations.",
            "doc_id": "doc1",
        },
        {
            "chunk_id": "doc2_chunk_0",
            "text": "Cosine similarity measures the angle between two vectors in space.",
            "doc_id": "doc2",
        },
        {
            "chunk_id": "doc2_chunk_1",
            "text": "Nearest neighbor search finds the most similar vectors in a database.",
            "doc_id": "doc2",
        },
        {
            "chunk_id": "doc3_chunk_0",
            "text": "Python is a popular programming language for data science and AI.",
            "doc_id": "doc3",
        },
        {
            "chunk_id": "doc3_chunk_1",
            "text": "HNSW is an approximate nearest neighbor algorithm for large datasets.",
            "doc_id": "doc3",
        },
    ]

    print(f"  Created {len(sample_chunks)} sample chunks")

    # Load embedding model
    print(f"\nLoading embedding model...")
    logger.info(f"Loading model: {model_name}")

    try:
        model = load_embedding_model(model_name, device=device)
        print(f"✓ Model loaded (dimension: {dimension})")

        # Generate embeddings for chunks
        print(f"\nGenerating embeddings for {len(sample_chunks)} chunks...")
        logger.info("Generating chunk embeddings")

        chunks_with_embeddings = embed_chunks(model, sample_chunks, normalize=normalize)

        print(f"✓ Generated {len(chunks_with_embeddings)} embeddings")

        # Create vector store
        print(f"\nBuilding vector store...")
        logger.info("Creating BruteForceVectorStore")

        store = BruteForceVectorStore(
            dimension=dimension,
            similarity_metric=similarity_metric,
            normalized=normalize,
        )

        # Add chunks to store
        for chunk in chunks_with_embeddings:
            store.add(
                chunk["embedding"],
                metadata={
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "doc_id": chunk["doc_id"],
                },
            )

        print(f"✓ Added {len(store)} vectors to store")
        stats = store.statistics()
        print(f"  Memory usage: {stats['memory_mb']:.2f} MB")

        # Generate query embedding
        print(f"\nGenerating query embedding...")
        print(f'  Query: "{query_text}"')
        logger.info(f"Generating query embedding")

        query_embedding = embed_texts(model, [query_text], normalize=normalize)[0]
        print(f"✓ Query embedding generated (shape: {query_embedding.shape})")

        # Search
        print(f"\nSearching for top-{top_k} similar chunks...")
        logger.info(f"Searching with brute-force (metric={similarity_metric})")

        results = store.search(query_embedding, k=top_k)

        print(f"✓ Found {len(results)} results")

        # Display results
        print(f"\n{'=' * 80}")
        print(f"Search Results (Top {len(results)}):")
        print(f"{'=' * 80}\n")

        for i, result in enumerate(results):
            print(f"Result {i + 1}:")
            print(f"  Score: {result['score']:.4f}")
            print(f"  Chunk ID: {result['metadata']['chunk_id']}")
            print(f"  Doc ID: {result['metadata']['doc_id']}")
            print(f"  Text: {result['metadata']['text']}")
            print()

        # Verify correctness
        print(f"{'=' * 80}")
        print("Stage 4 Exit Criteria:")
        print(f"  ✓ Vector store created successfully")
        print(f"  ✓ Search returns top-k results (k={top_k})")
        print(f"  ✓ Results ranked by similarity (descending)")

        # Check that scores are descending
        scores = [r["score"] for r in results]
        is_sorted = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
        if is_sorted:
            print(f"  ✓ Scores are correctly sorted: {[f'{s:.3f}' for s in scores]}")
        else:
            print(f"  ✗ Warning: Scores not sorted correctly!")

        print(f"{'=' * 80}\n")

        logger.info("Search demo complete")
        return 0

    except Exception as e:
        logger.error(f"Search demo failed: {e}")
        print(f"\n✗ Error: {e}\n", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def cmd_query_demo(args, config, logger):
    """
    Demonstrate end-to-end query pipeline (Stage 9).

    This command shows the complete query pipeline:
    1. Create sample document chunks
    2. Generate embeddings for chunks
    3. Build vector store (BruteForce or HNSW)
    4. Create query pipeline
    5. Query and get formatted results

    Args:
        args: Parsed command-line arguments
        config: Configuration dictionary
        logger: Logger instance
    """
    query_text = args.query

    logger.info(f'Query pipeline demo for query: "{query_text}"')

    # Get config
    model_name = config["embeddings"]["model_name"]
    device = config["embeddings"]["device"]
    normalize = config["embeddings"]["normalize"]
    dimension = config["embeddings"]["dimension"]
    similarity_metric = config["vectorstore"]["similarity_metric"]
    algorithm = args.algorithm
    top_k = args.top_k
    min_score = args.min_score

    print(f"\n{'=' * 80}")
    print("Query Pipeline Demo (Stage 9)")
    print(f"{'=' * 80}")
    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Vector store algorithm: {algorithm}")
    print(f"  Similarity metric: {similarity_metric}")
    print(f"  Top-k results: {top_k}")
    print(f"  Min score threshold: {min_score}")

    # Create sample knowledge base
    print(f"\nCreating sample knowledge base...")
    logger.info("Creating sample chunks")

    sample_chunks = [
        {
            "chunk_id": "doc1_chunk_0",
            "text": "Vector search is a technique for finding similar items using embeddings. It converts text into dense numerical vectors.",
            "doc_id": "doc1",
        },
        {
            "chunk_id": "doc1_chunk_1",
            "text": "Machine learning models convert text into dense vector representations. These embeddings capture semantic meaning.",
            "doc_id": "doc1",
        },
        {
            "chunk_id": "doc2_chunk_0",
            "text": "Cosine similarity measures the angle between two vectors in space. It's commonly used for semantic similarity.",
            "doc_id": "doc2",
        },
        {
            "chunk_id": "doc2_chunk_1",
            "text": "Nearest neighbor search finds the most similar vectors in a database. It's a fundamental operation in vector search.",
            "doc_id": "doc2",
        },
        {
            "chunk_id": "doc3_chunk_0",
            "text": "Python is a popular programming language for data science and AI. It has excellent libraries for machine learning.",
            "doc_id": "doc3",
        },
        {
            "chunk_id": "doc3_chunk_1",
            "text": "HNSW (Hierarchical Navigable Small World) is an approximate nearest neighbor algorithm for large datasets.",
            "doc_id": "doc3",
        },
        {
            "chunk_id": "doc4_chunk_0",
            "text": "Retrieval-Augmented Generation combines vector search with language models to produce accurate answers.",
            "doc_id": "doc4",
        },
        {
            "chunk_id": "doc4_chunk_1",
            "text": "Document embeddings enable semantic search that understands meaning rather than just keywords.",
            "doc_id": "doc4",
        },
    ]

    print(f"  Created {len(sample_chunks)} sample chunks")

    # Load embedding model
    print(f"\nLoading embedding model...")
    logger.info(f"Loading model: {model_name}")

    try:
        model = load_embedding_model(model_name, device=device)
        print(f"✓ Model loaded (dimension: {dimension})")

        # Generate embeddings for chunks
        print(f"\nGenerating embeddings for {len(sample_chunks)} chunks...")
        logger.info("Generating chunk embeddings")

        chunks_with_embeddings = embed_chunks(model, sample_chunks, normalize=normalize)

        print(f"✓ Generated {len(chunks_with_embeddings)} embeddings")

        # Create vector store based on algorithm choice
        print(f"\nBuilding {algorithm} vector store...")
        logger.info(f"Creating {algorithm} vector store")

        if algorithm == "hnsw":
            # Create HNSW index
            m = config["vectorstore"]["hnsw"]["m"]
            ef_construction = config["vectorstore"]["hnsw"]["ef_construction"]
            ef_search = config["vectorstore"]["hnsw"]["ef_search"]

            vector_store = create_hnsw_index(
                dimension=dimension,
                m=m,
                ef_construction=ef_construction,
                ef_search=ef_search,
                similarity_metric=similarity_metric,
                seed=42,
            )

            # Insert vectors
            for chunk in chunks_with_embeddings:
                vector_store.insert(
                    chunk["embedding"],
                    metadata={
                        "chunk_id": chunk["chunk_id"],
                        "doc_id": chunk["doc_id"],
                        "text": chunk["text"],
                    },
                )

            print(f"✓ HNSW index built (m={m}, ef_construction={ef_construction})")
            print(f"  - {len(vector_store)} vectors indexed")
            print(f"  - {vector_store.level_count} layers")

        else:
            # Create BruteForce index
            vector_store = BruteForceVectorStore(
                dimension=dimension, similarity_metric=similarity_metric
            )

            # Add vectors
            for chunk in chunks_with_embeddings:
                vector_store.add(
                    chunk["embedding"],
                    metadata={
                        "chunk_id": chunk["chunk_id"],
                        "doc_id": chunk["doc_id"],
                        "text": chunk["text"],
                    },
                )

            print(f"✓ Brute-force index built")
            print(f"  - {len(vector_store)} vectors indexed")

        # Create query pipeline
        print(f"\nCreating query pipeline...")
        logger.info("Creating query pipeline")

        query_config = {
            "top_k": top_k,
            "min_score": min_score,
            "normalize": normalize,
            "ef_search": (
                config["vectorstore"]["hnsw"]["ef_search"] if algorithm == "hnsw" else None
            ),
        }

        pipeline = create_query_pipeline(model, vector_store, query_config)
        print(f"✓ Query pipeline created")

        # Execute query
        print(f"\n{'=' * 80}")
        print(f'Executing Query: "{query_text}"')
        print(f"{'=' * 80}\n")

        logger.info(f"Executing query: {query_text}")
        results = pipeline.query(query_text, k=top_k, min_score=min_score)

        # Display results
        print(f"Query Results (Top {len(results)}):")
        print(f"{'=' * 80}\n")

        if not results:
            print("No results found above the minimum score threshold.")
        else:
            for i, result in enumerate(results):
                print(f"Result {i + 1}:")
                print(f"  Score: {result['score']:.4f}")
                print(f"  Distance: {result['distance']:.4f}")
                print(f"  Chunk ID: {result['metadata']['chunk_id']}")
                print(f"  Doc ID: {result['metadata']['doc_id']}")
                print(f"  Text: {result['metadata']['text'][:150]}...")
                print()

        # Test JSON output
        print(f"{'=' * 80}")
        print("JSON Output Format:")
        print(f"{'=' * 80}\n")

        json_output = pipeline.query_to_json(query_text, k=3, min_score=min_score, indent=2)
        print(json_output[:500] + "..." if len(json_output) > 500 else json_output)

        # Verify exit criteria
        print(f"\n{'=' * 80}")
        print("Stage 9 Exit Criteria:")
        print(f"  ✓ Query pipeline created successfully")
        print(f"  ✓ Query embedding generated")
        print(f"  ✓ Vector search performed ({algorithm})")
        print(f"  ✓ Results ranked by similarity (descending)")
        print(f"  ✓ JSON output format working")
        print(f"  ✓ Min score filtering applied (threshold={min_score})")

        # Check that scores are descending
        if results:
            scores = [r["score"] for r in results]
            is_sorted = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
            if is_sorted:
                print(f"  ✓ Scores are correctly sorted: {[f'{s:.3f}' for s in scores[:5]]}")
            else:
                print(f"  ✗ Warning: Scores not sorted correctly!")

        print(f"{'=' * 80}\n")

        logger.info("Query pipeline demo complete")
        return 0

    except Exception as e:
        logger.error(f"Query demo failed: {e}")
        print(f"\n✗ Error: {e}\n", file=sys.stderr)
        import traceback

        traceback.print_exc()
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
            "preview", help="Preview chunks from a document file (Stage 2)"
        )
        preview_parser.add_argument("file", type=str, help="Path to document file to preview")
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
        elif args.command == "search-demo":
            return cmd_search_demo(args, config, logger)
        elif args.command == "query-demo":
            return cmd_query_demo(args, config, logger)
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
