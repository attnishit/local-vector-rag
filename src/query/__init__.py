"""
Query Processing and Response Generation Module

Author: Nishit Attrey

This module provides the end-to-end query pipeline:
- Query embedding generation
- Vector search (BruteForce or HNSW)
- Result ranking and formatting
- JSON output

Components:
- pipeline.py: QueryPipeline class and factory function

Configuration (config.yaml):
    query:
        top_k: 5                 # Number of results to retrieve
        min_score: 0.0           # Minimum similarity threshold
        normalize: true          # Normalize query embeddings
        ef_search: null          # HNSW search parameter (null = use index default)

Usage:
    >>> from src.query import create_query_pipeline
    >>> from src.embeddings import load_embedding_model
    >>> from src.vectorstore import BruteForceVectorStore
    >>>
    >>> model = load_embedding_model()
    >>> vector_store = BruteForceVectorStore(dimension=384)
    >>>
    >>> # Add documents...
    >>>
    >>> pipeline = create_query_pipeline(model, vector_store)
    >>> results = pipeline.query("What is vector search?", k=5)
    >>> for r in results:
    ...     print(f"{r['score']:.3f}: {r['metadata']['text']}")

Exit Criteria (Stage 9):
✓ Query embedding works
✓ Vector search integration (both BruteForce and HNSW)
✓ Result ranking and formatting
✓ JSON output format
✓ CLI query tool
"""

# Stage 9: Query Pipeline
from .pipeline import (
    QueryPipeline,
    create_query_pipeline,
)

__all__ = [
    "QueryPipeline",
    "create_query_pipeline",
]
