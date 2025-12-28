"""
Local Vector RAG Database

Author: Nishit Attrey

A from-scratch implementation of vector embeddings and retrieval using
explicit algorithms and local-first architecture.

This package contains:
- config: Configuration loading and validation
- logger: Centralized logging setup
- ingestion: Document loading and chunking (Stage 2)
- embeddings: Local embedding generation (Stage 3)
- vectorstore: Vector storage and retrieval (Stages 4-8)
- query: Query processing and response generation (Stage 9)

Project follows a 12-stage development workflow:
    Stage 1: Project skeleton and configuration ✓
    Stage 2: Document ingestion and chunking ✓
    Stage 3: Local embedding pipeline ✓
    Stage 4: Brute-force vector store (baseline) ✓
    Stage 5: Index persistence ✓
    Stage 6: HNSW data structures ✓
    Stage 7: HNSW insertion logic ✓
    Stage 8: HNSW search algorithm ✓
    Stage 9: Query pipeline ✓
    Stage 11: Evaluation and benchmarking
    Stage 12: Documentation and final review

Global Constraints:
- No hosted APIs
- No LangChain/LlamaIndex/FAISS (except for benchmarking)
- Everything runs locally
- Explicit algorithms and data structures
- Heavy inline comments
- Correctness > performance
"""

__version__ = "0.1.0-stage9"
__author__ = "Nishit Attrey"

# Version tuple for programmatic access
VERSION = tuple(map(int, __version__.split('-')[0].split('.')))
