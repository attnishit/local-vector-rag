"""
Chunk Persistence Module

Author: Nishit Attrey

This module provides functionality to persist document chunks to disk,
enabling:
- Avoiding re-chunking documents on every run
- Detecting when source documents change
- Loading previously chunked documents
- Incremental document processing

File Format:
    Chunks are saved as JSON files in data/processed/:
    - Filename: {doc_id}_chunks.json
    - Contains: source file hash, chunk config, all chunks

Functions:
    save_chunks: Save chunks to disk with metadata
    load_chunks: Load chunks from disk
    chunks_exist: Check if chunks exist for a document
    compute_file_hash: Hash file content for change detection
    should_rechunk: Determine if document needs re-chunking
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def compute_file_hash(filepath: Path) -> str:
    """
    Compute SHA256 hash of file content for change detection.

    Args:
        filepath: Path to file

    Returns:
        Hex string of SHA256 hash

    Example:
        >>> hash1 = compute_file_hash(Path("data/raw/sample.txt"))
        >>> # Edit the file
        >>> hash2 = compute_file_hash(Path("data/raw/sample.txt"))
        >>> hash1 != hash2
        True
    """
    sha256_hash = hashlib.sha256()

    with open(filepath, "rb") as f:
        # Read in chunks to handle large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()


def save_chunks(
    chunks: List[Dict[str, Any]],
    output_dir: Path,
    source_file: Optional[Path] = None,
    chunk_config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save chunks to disk with metadata.

    Creates a JSON file in output_dir named {doc_id}_chunks.json
    containing all chunks and metadata.

    Args:
        chunks: List of chunk dictionaries from chunk_text()
        output_dir: Directory to save chunks in (e.g., data/processed/)
        source_file: Optional path to source document (for hash computation)
        chunk_config: Optional chunk configuration (size, overlap, etc.)

    Raises:
        ValueError: If chunks list is empty
        IOError: If unable to write file

    Example:
        >>> chunks = chunk_text("Sample text", "doc1")
        >>> save_chunks(
        ...     chunks,
        ...     Path("data/processed"),
        ...     source_file=Path("data/raw/sample.txt"),
        ...     chunk_config={"chunk_size": 512, "chunk_overlap": 50}
        ... )
    """
    if not chunks:
        raise ValueError("Cannot save empty chunks list")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract doc_id from first chunk
    doc_id = chunks[0]["doc_id"]

    # Compute source file hash if provided
    source_hash = None
    source_path_str = None
    if source_file:
        source_file = Path(source_file)
        if source_file.exists():
            source_hash = compute_file_hash(source_file)
            source_path_str = str(source_file)

    # Build metadata
    metadata = {
        "doc_id": doc_id,
        "source_file": source_path_str,
        "source_hash": source_hash,
        "chunked_at": datetime.now().isoformat(),
        "num_chunks": len(chunks),
        "chunk_config": chunk_config or {},
        "chunks": chunks,
    }

    # Save to file
    output_path = output_dir / f"{doc_id}_chunks.json"
    temp_path = output_dir / f"{doc_id}_chunks.json.tmp"

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Atomic rename
    temp_path.replace(output_path)

    logger.info(
        f"Saved {len(chunks)} chunks for document '{doc_id}' to {output_path}"
    )


def load_chunks(doc_id: str, chunks_dir: Path) -> Dict[str, Any]:
    """
    Load chunks from disk.

    Args:
        doc_id: Document identifier
        chunks_dir: Directory containing chunks files (e.g., data/processed/)

    Returns:
        Dictionary with metadata and chunks:
        {
            "doc_id": str,
            "source_file": str,
            "source_hash": str,
            "chunked_at": str,
            "num_chunks": int,
            "chunk_config": dict,
            "chunks": List[dict]
        }

    Raises:
        FileNotFoundError: If chunks file doesn't exist

    Example:
        >>> data = load_chunks("doc1", Path("data/processed"))
        >>> chunks = data["chunks"]
        >>> print(f"Loaded {len(chunks)} chunks")
    """
    chunks_dir = Path(chunks_dir)
    chunks_path = chunks_dir / f"{doc_id}_chunks.json"

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks not found for document '{doc_id}' at {chunks_path}"
        )

    with open(chunks_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    logger.info(
        f"Loaded {metadata['num_chunks']} chunks for document '{doc_id}' from {chunks_path}"
    )

    return metadata


def chunks_exist(doc_id: str, chunks_dir: Path) -> bool:
    """
    Check if chunks exist for a document.

    Args:
        doc_id: Document identifier
        chunks_dir: Directory containing chunks files

    Returns:
        True if chunks file exists

    Example:
        >>> if not chunks_exist("doc1", Path("data/processed")):
        ...     # Need to chunk the document
        ...     pass
    """
    chunks_dir = Path(chunks_dir)
    chunks_path = chunks_dir / f"{doc_id}_chunks.json"
    return chunks_path.exists()


def should_rechunk(
    doc_id: str,
    source_file: Path,
    chunks_dir: Path,
    chunk_config: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Determine if a document should be re-chunked.

    Returns True if:
    - Chunks don't exist
    - Source file hash has changed
    - Chunk configuration has changed

    Args:
        doc_id: Document identifier
        source_file: Path to source document
        chunks_dir: Directory containing chunks files
        chunk_config: Current chunk configuration to compare against

    Returns:
        True if document should be re-chunked

    Example:
        >>> if should_rechunk("doc1", Path("data/raw/sample.txt"), Path("data/processed")):
        ...     # Re-chunk the document
        ...     chunks = chunk_text(...)
        ...     save_chunks(chunks, ...)
    """
    # If chunks don't exist, need to chunk
    if not chunks_exist(doc_id, chunks_dir):
        logger.debug(f"Chunks don't exist for '{doc_id}', should rechunk")
        return True

    # Load existing chunks metadata
    try:
        metadata = load_chunks(doc_id, chunks_dir)
    except Exception as e:
        logger.warning(f"Failed to load chunks for '{doc_id}': {e}, will rechunk")
        return True

    # Check if source file hash has changed
    source_file = Path(source_file)
    if source_file.exists():
        current_hash = compute_file_hash(source_file)
        saved_hash = metadata.get("source_hash")

        if current_hash != saved_hash:
            logger.info(
                f"Source file hash changed for '{doc_id}' "
                f"(saved: {saved_hash[:16]}..., current: {current_hash[:16]}...), should rechunk"
            )
            return True

    # Check if chunk configuration has changed
    if chunk_config:
        saved_config = metadata.get("chunk_config", {})
        if chunk_config != saved_config:
            logger.info(
                f"Chunk config changed for '{doc_id}' "
                f"(saved: {saved_config}, current: {chunk_config}), should rechunk"
            )
            return True

    logger.debug(f"Chunks up-to-date for '{doc_id}', no need to rechunk")
    return False


def get_all_chunked_documents(chunks_dir: Path) -> List[str]:
    """
    Get list of all document IDs that have been chunked.

    Args:
        chunks_dir: Directory containing chunks files

    Returns:
        List of document IDs

    Example:
        >>> doc_ids = get_all_chunked_documents(Path("data/processed"))
        >>> print(f"Found {len(doc_ids)} chunked documents")
    """
    chunks_dir = Path(chunks_dir)

    if not chunks_dir.exists():
        return []

    doc_ids = []
    for chunks_file in chunks_dir.glob("*_chunks.json"):
        # Extract doc_id from filename (remove _chunks.json suffix)
        doc_id = chunks_file.stem.replace("_chunks", "")
        doc_ids.append(doc_id)

    return sorted(doc_ids)


def delete_chunks(doc_id: str, chunks_dir: Path) -> None:
    """
    Delete chunks for a document.

    Args:
        doc_id: Document identifier
        chunks_dir: Directory containing chunks files

    Raises:
        FileNotFoundError: If chunks file doesn't exist

    Example:
        >>> delete_chunks("doc1", Path("data/processed"))
    """
    chunks_dir = Path(chunks_dir)
    chunks_path = chunks_dir / f"{doc_id}_chunks.json"

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks not found for document '{doc_id}' at {chunks_path}"
        )

    chunks_path.unlink()
    logger.info(f"Deleted chunks for document '{doc_id}'")
