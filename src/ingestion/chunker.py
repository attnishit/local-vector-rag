"""
Text Chunking Module

Author: Nishit Attrey

This module provides functionality to split text into overlapping chunks
with configurable size and overlap. Chunks are deterministic and include
stable IDs for tracking.

Key Features:
- Fixed-size chunking with character-based windows
- Configurable overlap between chunks
- Deterministic chunking (same input → same output)
- Stable chunk IDs based on content and position
- Whitespace normalization

Functions:
    chunk_text: Split text into overlapping chunks
    normalize_text: Normalize whitespace in text
    generate_chunk_id: Generate stable ID for a chunk

Example:
    >>> from src.ingestion.chunker import chunk_text
    >>> text = "This is a sample document..."
    >>> chunks = chunk_text(text, doc_id="doc1", chunk_size=100, overlap=20)
    >>> for chunk in chunks:
    ...     print(f"{chunk['chunk_id']}: {chunk['text'][:50]}...")
"""

import hashlib
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """
    Normalize whitespace in text.

    Performs the following normalization:
    1. Replace all whitespace characters (tabs, newlines, etc.) with single spaces
    2. Collapse multiple consecutive spaces into a single space
    3. Strip leading and trailing whitespace

    Args:
        text: Raw text string to normalize

    Returns:
        Normalized text with consistent whitespace

    Example:
        >>> normalize_text("Hello\\n\\n  world\\t!")
        'Hello world !'
        >>> normalize_text("  Multiple    spaces  ")
        'Multiple spaces'

    Note:
        This normalization makes chunking deterministic by ensuring
        consistent whitespace handling across different input formats.
    """
    if not text:
        return ""

    normalized = ' '.join(text.split())

    return normalized


def generate_chunk_id(doc_id: str, chunk_index: int, text: str) -> str:
    """
    Generate a stable, deterministic chunk ID.

    The chunk ID is generated using:
    - Document ID (for traceability to source)
    - Chunk index (for ordering)
    - Content hash (for uniqueness)

    Format: {doc_id}_chunk_{index}_{hash_prefix}

    Args:
        doc_id: Identifier of the source document
        chunk_index: Zero-based index of this chunk within the document
        text: The chunk text content

    Returns:
        Stable chunk ID string

    Example:
        >>> generate_chunk_id("doc1", 0, "Hello world")
        'doc1_chunk_0_b10a8db'

    Note:
        The hash ensures that if chunk content changes, the ID changes.
        This is important for detecting when chunks need to be re-embedded.
    """
    content_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    hash_prefix = content_hash[:7]
    chunk_id = f"{doc_id}_chunk_{chunk_index}_{hash_prefix}"

    return chunk_id


def chunk_text(
    text: str,
    doc_id: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50
) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks of fixed size.

    This function implements a sliding window algorithm to chunk text:
    1. Normalize the text (consistent whitespace)
    2. Create fixed-size windows with overlap
    3. Generate stable IDs for each chunk
    4. Return list of chunk dictionaries

    Args:
        text: Source text to chunk
        doc_id: Identifier for the source document
        chunk_size: Size of each chunk in characters (default: 512)
        chunk_overlap: Number of characters to overlap between chunks (default: 50)

    Returns:
        List of chunk dictionaries, each containing:
        - chunk_id: Stable unique identifier
        - doc_id: Source document ID
        - text: The chunk text content
        - start: Starting character position in original text
        - end: Ending character position in original text
        - index: Zero-based chunk index

    Raises:
        ValueError: If chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size

    Example:
        >>> text = "This is a long document that needs to be split into chunks."
        >>> chunks = chunk_text(text, doc_id="doc1", chunk_size=30, overlap=10)
        >>> len(chunks)
        3
        >>> chunks[0]['text']
        'This is a long document that'
        >>> chunks[1]['text'][:10]  # Shows overlap
        'ment that '

    Algorithm:
        For a text of length N with chunk_size C and overlap O:
        - Chunk 0: [0, C)
        - Chunk 1: [C-O, 2C-O)
        - Chunk 2: [2C-2O, 3C-2O)
        - ...
        - Last chunk: May be smaller than C if remaining text < C

    Note:
        - Chunks are deterministic: same input always produces same chunks
        - Chunk IDs are stable: same content produces same ID
        - Overlap ensures context continuity between chunks
        - Empty or whitespace-only text returns an empty list
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got: {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be non-negative, got: {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
        )

    normalized_text = normalize_text(text)

    if not normalized_text:
        logger.warning(f"Empty text for doc_id: {doc_id}, returning empty chunk list")
        return []

    step_size = chunk_size - chunk_overlap

    chunks = []
    chunk_index = 0
    start_pos = 0

    while start_pos < len(normalized_text):
        end_pos = start_pos + chunk_size
        chunk_text_content = normalized_text[start_pos:end_pos]

        if not chunk_text_content.strip():
            break

        chunk_id = generate_chunk_id(doc_id, chunk_index, chunk_text_content)

        chunk = {
            'chunk_id': chunk_id,
            'doc_id': doc_id,
            'text': chunk_text_content,
            'start': start_pos,
            'end': min(end_pos, len(normalized_text)),
            'index': chunk_index,
        }

        chunks.append(chunk)

        logger.debug(
            f"Created chunk {chunk_index} for {doc_id}: "
            f"[{start_pos}:{end_pos}] ({len(chunk_text_content)} chars)"
        )

        start_pos += step_size
        chunk_index += 1

    logger.info(
        f"Chunked document {doc_id}: {len(normalized_text)} chars → "
        f"{len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})"
    )

    return chunks


def chunk_statistics(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate statistics about a list of chunks.

    Useful for debugging and validation.

    Args:
        chunks: List of chunk dictionaries from chunk_text()

    Returns:
        Dictionary with statistics:
        - num_chunks: Total number of chunks
        - total_chars: Total characters across all chunks
        - avg_chunk_size: Average chunk size
        - min_chunk_size: Size of smallest chunk
        - max_chunk_size: Size of largest chunk

    Example:
        >>> chunks = chunk_text("Sample text...", "doc1", chunk_size=100, overlap=20)
        >>> stats = chunk_statistics(chunks)
        >>> print(f"Created {stats['num_chunks']} chunks")
    """
    if not chunks:
        return {
            'num_chunks': 0,
            'total_chars': 0,
            'avg_chunk_size': 0,
            'min_chunk_size': 0,
            'max_chunk_size': 0,
        }

    chunk_sizes = [len(chunk['text']) for chunk in chunks]

    return {
        'num_chunks': len(chunks),
        'total_chars': sum(chunk_sizes),
        'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
        'min_chunk_size': min(chunk_sizes),
        'max_chunk_size': max(chunk_sizes),
    }
