"""
Document Ingestion and Chunking Module

Author: Nishit Attrey

This module provides:
- Document loading from multiple formats (.txt, .pdf, .docx, .md)
- Text normalization and whitespace handling
- Chunking with configurable size and overlap
- Stable chunk ID generation
- Document and chunk statistics
- Chunk persistence (save/load from disk)

Main Functions:
    From loader:
        - load_document: Load a single document
        - load_documents_from_directory: Load all documents from a directory
        - document_statistics: Calculate statistics about documents

    From chunker:
        - chunk_text: Split text into overlapping chunks
        - normalize_text: Normalize whitespace
        - chunk_statistics: Calculate statistics about chunks

    From persistence:
        - save_chunks: Save chunks to disk with metadata
        - load_chunks: Load chunks from disk
        - chunks_exist: Check if chunks exist for a document
        - should_rechunk: Determine if document needs re-chunking
        - get_all_chunked_documents: List all chunked documents
        - delete_chunks: Delete chunks for a document

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

from .persistence import (
    save_chunks,
    load_chunks,
    chunks_exist,
    should_rechunk,
    get_all_chunked_documents,
    delete_chunks,
    compute_file_hash,
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
    # Persistence functions
    "save_chunks",
    "load_chunks",
    "chunks_exist",
    "should_rechunk",
    "get_all_chunked_documents",
    "delete_chunks",
    "compute_file_hash",
]
