"""
Document Ingestion and Chunking Module

Author: Nishit Attrey

This module provides:
- Document loading from .txt files
- Text normalization and whitespace handling
- Chunking with configurable size and overlap
- Stable chunk ID generation
- Document and chunk statistics

Main Functions:
    From loader:
        - load_document: Load a single document
        - load_documents_from_directory: Load all documents from a directory
        - document_statistics: Calculate statistics about documents

    From chunker:
        - chunk_text: Split text into overlapping chunks
        - normalize_text: Normalize whitespace
        - chunk_statistics: Calculate statistics about chunks

Example:
    >>> from src.ingestion import load_documents_from_directory, chunk_text
    >>> docs = load_documents_from_directory("data/raw")
    >>> all_chunks = []
    >>> for doc in docs:
    ...     chunks = chunk_text(doc['text'], doc['doc_id'], chunk_size=512, chunk_overlap=50)
    ...     all_chunks.extend(chunks)
"""

from .loader import (
    load_document,
    load_documents_from_directory,
    get_document_id,
    document_statistics,
)

from .chunker import (
    chunk_text,
    normalize_text,
    generate_chunk_id,
    chunk_statistics,
)

__all__ = [
    # Loader functions
    "load_document",
    "load_documents_from_directory",
    "get_document_id",
    "document_statistics",
    # Chunker functions
    "chunk_text",
    "normalize_text",
    "generate_chunk_id",
    "chunk_statistics",
]
