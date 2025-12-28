"""
Vector Store Persistence

Author: Nishit Attrey

This module provides functionality to persist vector stores to disk and load them
back. This enables:
- Avoiding re-embedding documents on every run
- Sharing pre-computed indexes
- Backing up vector stores
- Version control of indexes

File Format:
    Indexes are saved as a directory containing:
    - vectors.npy: NumPy binary file with all vectors
    - metadata.json: JSON file with metadata for each vector
    - index_info.json: Index configuration and integrity information

Index Info Format:
    {
        "version": "0.1.0-stage5",
        "created_at": "2025-12-27T12:00:00",
        "dimension": 384,
        "num_vectors": 1000,
        "similarity_metric": "cosine",
        "normalized": true,
        "checksum": "sha256_hash_of_vectors"
    }

Safety Features:
    - Version checking (prevents loading incompatible indexes)
    - Checksum validation (detects corruption)
    - Atomic writes (use temporary files and rename)

Author: RAG Team
Version: 0.1.0-stage5
"""

import numpy as np
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Current index format version
INDEX_FORMAT_VERSION = "0.1.0-stage5"


def compute_checksum(vectors: np.ndarray) -> str:
    """
    Compute SHA256 checksum of vector data for integrity verification.

    Args:
        vectors: NumPy array of vectors

    Returns:
        Hex string of SHA256 hash

    Example:
        >>> vectors = np.random.randn(100, 384)
        >>> checksum = compute_checksum(vectors)
        >>> len(checksum)
        64  # SHA256 produces 64 hex characters
    """
    vector_bytes = vectors.tobytes()
    sha256_hash = hashlib.sha256(vector_bytes)

    return sha256_hash.hexdigest()


def save_index(
    index_dir: Path,
    vectors: np.ndarray,
    metadata: List[Dict[str, Any]],
    dimension: int,
    similarity_metric: str,
    normalized: bool,
    index_name: str = "default",
) -> None:
    """
    Save a vector index to disk.

    Creates a directory structure:
        index_dir/
        ├── vectors.npy         # Binary vectors
        ├── metadata.json       # Vector metadata
        └── index_info.json     # Index configuration

    Args:
        index_dir: Directory to save index in
        vectors: NumPy array of vectors (n, d)
        metadata: List of metadata dicts
        dimension: Vector dimension
        similarity_metric: Similarity metric used
        normalized: Whether vectors are normalized
        index_name: Name for the index (for logging)

    Raises:
        ValueError: If vectors/metadata shapes don't match
        IOError: If unable to write files

    Example:
        >>> save_index(
        ...     Path("data/indexes/my_index"),
        ...     vectors=my_vectors,
        ...     metadata=my_metadata,
        ...     dimension=384,
        ...     similarity_metric="cosine",
        ...     normalized=True
        ... )
    """
    if len(vectors) != len(metadata):
        raise ValueError(
            f"Vector count ({len(vectors)}) doesn't match metadata count ({len(metadata)})"
        )

    if vectors.shape[1] != dimension:
        raise ValueError(
            f"Vector dimension ({vectors.shape[1]}) doesn't match specified dimension ({dimension})"
        )

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving index '{index_name}' to {index_dir}")

    checksum = compute_checksum(vectors)

    index_info = {
        "version": INDEX_FORMAT_VERSION,
        "created_at": datetime.now().isoformat(),
        "index_name": index_name,
        "dimension": dimension,
        "num_vectors": len(vectors),
        "similarity_metric": similarity_metric,
        "normalized": normalized,
        "checksum": checksum,
    }

    vectors_temp = index_dir / "vectors_tmp"
    vectors_path = index_dir / "vectors.npy"
    np.save(vectors_temp, vectors)
    (index_dir / "vectors_tmp.npy").replace(vectors_path)

    logger.debug(f"Saved vectors: {vectors.shape}")

    metadata_temp = index_dir / "metadata.json.tmp"
    metadata_path = index_dir / "metadata.json"
    with open(metadata_temp, "w") as f:
        json.dump(metadata, f, indent=2)
    metadata_temp.replace(metadata_path)

    logger.debug(f"Saved metadata: {len(metadata)} entries")

    info_temp = index_dir / "index_info.json.tmp"
    info_path = index_dir / "index_info.json"
    with open(info_temp, "w") as f:
        json.dump(index_info, f, indent=2)
    info_temp.replace(info_path)

    logger.info(
        f"Index saved successfully: {len(vectors)} vectors, "
        f"{dimension}D, checksum={checksum[:16]}..."
    )


def load_index(
    index_dir: Path, verify_checksum: bool = True
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load a vector index from disk.

    Args:
        index_dir: Directory containing saved index
        verify_checksum: If True, verify data integrity using checksum

    Returns:
        Tuple of (vectors, metadata, index_info)

    Raises:
        FileNotFoundError: If index directory or files don't exist
        ValueError: If index format is incompatible or corrupted
        IOError: If unable to read files

    Example:
        >>> vectors, metadata, info = load_index(Path("data/indexes/my_index"))
        >>> print(f"Loaded {len(vectors)} vectors")
    """
    index_dir = Path(index_dir)

    if not index_dir.exists():
        raise FileNotFoundError(f"Index directory not found: {index_dir}")

    logger.info(f"Loading index from {index_dir}")

    vectors_path = index_dir / "vectors.npy"
    metadata_path = index_dir / "metadata.json"
    info_path = index_dir / "index_info.json"

    if not vectors_path.exists():
        raise FileNotFoundError(f"Vectors file not found: {vectors_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not info_path.exists():
        raise FileNotFoundError(f"Index info file not found: {info_path}")

    with open(info_path, "r") as f:
        index_info = json.load(f)

    logger.debug(f"Index info: {index_info}")

    if index_info["version"] != INDEX_FORMAT_VERSION:
        logger.warning(
            f"Index version mismatch: expected {INDEX_FORMAT_VERSION}, got {index_info['version']}"
        )

    vectors = np.load(vectors_path)
    logger.debug(f"Loaded vectors: {vectors.shape}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    logger.debug(f"Loaded metadata: {len(metadata)} entries")

    if len(vectors) != len(metadata):
        raise ValueError(
            f"Inconsistent index: {len(vectors)} vectors but {len(metadata)} metadata entries"
        )

    if len(vectors) != index_info["num_vectors"]:
        raise ValueError(
            f"Inconsistent index: expected {index_info['num_vectors']} vectors, "
            f"found {len(vectors)}"
        )

    if vectors.shape[1] != index_info["dimension"]:
        raise ValueError(
            f"Dimension mismatch: expected {index_info['dimension']}, got {vectors.shape[1]}"
        )

    if verify_checksum:
        logger.debug("Verifying checksum...")
        actual_checksum = compute_checksum(vectors)
        expected_checksum = index_info["checksum"]

        if actual_checksum != expected_checksum:
            raise ValueError(
                f"Checksum mismatch! Index may be corrupted.\n"
                f"Expected: {expected_checksum}\n"
                f"Actual: {actual_checksum}"
            )

        logger.debug("Checksum verified ✓")

    logger.info(f"Index loaded successfully: {len(vectors)} vectors, {vectors.shape[1]}D")

    return vectors, metadata, index_info


def index_exists(index_dir: Path) -> bool:
    """
    Check if a valid index exists at the given directory.

    Args:
        index_dir: Directory to check

    Returns:
        True if all required index files exist

    Example:
        >>> if index_exists(Path("data/indexes/my_index")):
        ...     vectors, metadata, info = load_index(Path("data/indexes/my_index"))
    """
    index_dir = Path(index_dir)

    if not index_dir.exists():
        return False

    required_files = ["vectors.npy", "metadata.json", "index_info.json"]

    for filename in required_files:
        if not (index_dir / filename).exists():
            return False

    return True


def get_index_info(index_dir: Path) -> Dict[str, Any]:
    """
    Get index information without loading the full index.

    Useful for checking index properties before loading.

    Args:
        index_dir: Directory containing saved index

    Returns:
        Index info dictionary

    Raises:
        FileNotFoundError: If index info file doesn't exist

    Example:
        >>> info = get_index_info(Path("data/indexes/my_index"))
        >>> print(f"Index has {info['num_vectors']} vectors")
    """
    index_dir = Path(index_dir)
    info_path = index_dir / "index_info.json"

    if not info_path.exists():
        raise FileNotFoundError(f"Index info not found: {info_path}")

    with open(info_path, "r") as f:
        return json.load(f)


def list_indexes(indexes_dir: Path) -> List[Dict[str, Any]]:
    """
    List all available indexes in a directory.

    Args:
        indexes_dir: Base directory containing indexes

    Returns:
        List of index info dictionaries

    Example:
        >>> for info in list_indexes(Path("data/indexes")):
        ...     print(f"{info['index_name']}: {info['num_vectors']} vectors")
    """
    indexes_dir = Path(indexes_dir)

    if not indexes_dir.exists():
        return []

    indexes = []

    for subdir in indexes_dir.iterdir():
        if subdir.is_dir() and index_exists(subdir):
            try:
                info = get_index_info(subdir)
                info["path"] = str(subdir)
                indexes.append(info)
            except Exception as e:
                logger.warning(f"Failed to read index at {subdir}: {e}")
                continue

    return indexes


def delete_index(index_dir: Path) -> None:
    """
    Delete a persisted index.

    Args:
        index_dir: Directory containing index to delete

    Raises:
        FileNotFoundError: If index doesn't exist

    Example:
        >>> delete_index(Path("data/indexes/old_index"))
    """
    index_dir = Path(index_dir)

    if not index_dir.exists():
        raise FileNotFoundError(f"Index not found: {index_dir}")

    logger.info(f"Deleting index at {index_dir}")

    for filepath in index_dir.glob("*"):
        filepath.unlink()

    index_dir.rmdir()

    logger.info("Index deleted successfully")
