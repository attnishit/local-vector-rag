"""
Embeddings Persistence Module

Author: Nishit Attrey

This module provides functionality to persist embeddings to disk,
enabling:
- Avoiding re-embedding chunks on every run
- Detecting when chunks change and need re-embedding
- Loading previously generated embeddings
- Incremental embedding (only embed new/changed chunks)

File Format:
    Embeddings are saved as .npz (compressed NumPy) files in data/embeddings/:
    - Filename: {collection_name}.npz
    - Contains: embedding matrix, chunk_ids, model metadata

Functions:
    save_embeddings: Save embeddings to disk with metadata
    load_embeddings: Load embeddings from disk
    embeddings_exist: Check if embeddings exist for a collection
    get_embeddings_for_chunks: Get embeddings for specific chunk_ids
    update_embeddings: Add new embeddings to existing collection
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


def save_embeddings(
    embeddings: np.ndarray,
    chunk_ids: List[str],
    output_dir: Path,
    collection_name: str,
    model_name: Optional[str] = None,
    normalized: bool = True,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save embeddings to disk with metadata.

    Creates a compressed .npz file containing embeddings and metadata.

    Args:
        embeddings: NumPy array of shape (n, dimension)
        chunk_ids: List of chunk IDs corresponding to embeddings
        output_dir: Directory to save embeddings in (e.g., data/embeddings/)
        collection_name: Name for this embedding collection
        model_name: Name of the embedding model used
        normalized: Whether embeddings are L2-normalized
        additional_metadata: Optional additional metadata to store

    Raises:
        ValueError: If embeddings and chunk_ids lengths don't match
        IOError: If unable to write file

    Example:
        >>> embeddings = np.random.randn(100, 384)
        >>> chunk_ids = [f"doc1_chunk_{i}_abc" for i in range(100)]
        >>> save_embeddings(
        ...     embeddings,
        ...     chunk_ids,
        ...     Path("data/embeddings"),
        ...     "my_collection",
        ...     model_name="all-MiniLM-L6-v2"
        ... )
    """
    if len(embeddings) != len(chunk_ids):
        raise ValueError(
            f"Embeddings count ({len(embeddings)}) doesn't match "
            f"chunk_ids count ({len(chunk_ids)})"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build metadata
    metadata = {
        "collection_name": collection_name,
        "num_embeddings": len(embeddings),
        "dimension": embeddings.shape[1] if embeddings.ndim > 1 else 0,
        "model_name": model_name,
        "normalized": normalized,
        "created_at": datetime.now().isoformat(),
        "chunk_ids": chunk_ids,  # Store in metadata for easy lookup
    }

    # Merge additional metadata
    if additional_metadata:
        metadata.update(additional_metadata)

    # Save using numpy's compressed format
    output_path = output_dir / f"{collection_name}.npz"
    # Note: numpy adds .npz automatically, so temp file should not have .npz
    temp_base = output_dir / f".{collection_name}_tmp"

    # Save as compressed npz file with both embeddings and metadata
    # We store metadata as a JSON string in the npz file
    np.savez_compressed(
        str(temp_base),  # numpy will add .npz
        embeddings=embeddings,
        metadata=np.array([json.dumps(metadata)])  # Store metadata as string array
    )

    # The actual temp file created by numpy
    temp_path = Path(str(temp_base) + ".npz")

    # Atomic rename
    temp_path.replace(output_path)

    logger.info(
        f"Saved {len(embeddings)} embeddings for collection '{collection_name}' "
        f"({embeddings.shape[1]}D) to {output_path}"
    )


def load_embeddings(
    collection_name: str,
    embeddings_dir: Path,
) -> Dict[str, Any]:
    """
    Load embeddings from disk.

    Args:
        collection_name: Name of the embedding collection
        embeddings_dir: Directory containing embeddings files

    Returns:
        Dictionary with:
        {
            "embeddings": np.ndarray,  # Shape (n, dimension)
            "chunk_ids": List[str],
            "metadata": Dict[str, Any],  # Full metadata
        }

    Raises:
        FileNotFoundError: If embeddings file doesn't exist

    Example:
        >>> data = load_embeddings("my_collection", Path("data/embeddings"))
        >>> embeddings = data["embeddings"]
        >>> chunk_ids = data["chunk_ids"]
        >>> print(f"Loaded {len(embeddings)} embeddings")
    """
    embeddings_dir = Path(embeddings_dir)
    embeddings_path = embeddings_dir / f"{collection_name}.npz"

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found for collection '{collection_name}' at {embeddings_path}"
        )

    # Load npz file
    with np.load(embeddings_path, allow_pickle=True) as data:
        embeddings = data["embeddings"]
        metadata_str = str(data["metadata"][0])  # Extract metadata string
        metadata = json.loads(metadata_str)

    chunk_ids = metadata.get("chunk_ids", [])

    logger.info(
        f"Loaded {len(embeddings)} embeddings for collection '{collection_name}' "
        f"from {embeddings_path}"
    )

    return {
        "embeddings": embeddings,
        "chunk_ids": chunk_ids,
        "metadata": metadata,
    }


def embeddings_exist(collection_name: str, embeddings_dir: Path) -> bool:
    """
    Check if embeddings exist for a collection.

    Args:
        collection_name: Name of the embedding collection
        embeddings_dir: Directory containing embeddings files

    Returns:
        True if embeddings file exists

    Example:
        >>> if not embeddings_exist("my_collection", Path("data/embeddings")):
        ...     # Need to generate embeddings
        ...     pass
    """
    embeddings_dir = Path(embeddings_dir)
    embeddings_path = embeddings_dir / f"{collection_name}.npz"
    return embeddings_path.exists()


def get_embeddings_for_chunks(
    chunk_ids: List[str],
    collection_name: str,
    embeddings_dir: Path,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Get embeddings for specific chunk IDs.

    Returns embeddings for requested chunks and identifies missing chunks.

    Args:
        chunk_ids: List of chunk IDs to get embeddings for
        collection_name: Name of the embedding collection
        embeddings_dir: Directory containing embeddings files

    Returns:
        Tuple of:
        - embeddings (np.ndarray): Array of shape (n, dimension) for found chunks
        - found_chunk_ids (List[str]): Chunk IDs that were found
        - missing_chunk_ids (List[str]): Chunk IDs that were not found

    Example:
        >>> chunk_ids = ["doc1_chunk_0_abc", "doc1_chunk_1_def", "doc2_chunk_0_ghi"]
        >>> embeddings, found, missing = get_embeddings_for_chunks(
        ...     chunk_ids, "my_collection", Path("data/embeddings")
        ... )
        >>> print(f"Found: {len(found)}, Missing: {len(missing)}")
    """
    # Load all embeddings
    try:
        data = load_embeddings(collection_name, embeddings_dir)
    except FileNotFoundError:
        # No embeddings exist, all chunks are missing
        return np.array([]).reshape(0, 0), [], chunk_ids

    all_embeddings = data["embeddings"]
    all_chunk_ids = data["chunk_ids"]

    # Create mapping from chunk_id to index
    chunk_id_to_idx = {cid: idx for idx, cid in enumerate(all_chunk_ids)}

    # Find which chunks exist
    found_indices = []
    found_chunk_ids = []
    missing_chunk_ids = []

    for cid in chunk_ids:
        if cid in chunk_id_to_idx:
            idx = chunk_id_to_idx[cid]
            found_indices.append(idx)
            found_chunk_ids.append(cid)
        else:
            missing_chunk_ids.append(cid)

    # Extract embeddings for found chunks
    if found_indices:
        found_embeddings = all_embeddings[found_indices]
    else:
        # No chunks found, return empty array with correct dimension
        dimension = data["metadata"].get("dimension", 0)
        found_embeddings = np.array([]).reshape(0, dimension)

    logger.info(
        f"Found {len(found_chunk_ids)}/{len(chunk_ids)} chunk embeddings "
        f"in collection '{collection_name}'"
    )

    return found_embeddings, found_chunk_ids, missing_chunk_ids


def update_embeddings(
    new_embeddings: np.ndarray,
    new_chunk_ids: List[str],
    collection_name: str,
    embeddings_dir: Path,
    model_name: Optional[str] = None,
    normalized: bool = True,
) -> None:
    """
    Add new embeddings to existing collection (incremental update).

    If collection doesn't exist, creates it.
    If chunk_ids already exist, replaces their embeddings.

    Args:
        new_embeddings: New embeddings to add (n, dimension)
        new_chunk_ids: Chunk IDs for new embeddings
        collection_name: Name of the embedding collection
        embeddings_dir: Directory containing embeddings files
        model_name: Name of the embedding model
        normalized: Whether embeddings are normalized

    Example:
        >>> # Add new embeddings to existing collection
        >>> new_embeddings = np.random.randn(10, 384)
        >>> new_chunk_ids = [f"doc2_chunk_{i}_xyz" for i in range(10)]
        >>> update_embeddings(
        ...     new_embeddings,
        ...     new_chunk_ids,
        ...     "my_collection",
        ...     Path("data/embeddings")
        ... )
    """
    if len(new_embeddings) != len(new_chunk_ids):
        raise ValueError(
            f"New embeddings count ({len(new_embeddings)}) doesn't match "
            f"new chunk_ids count ({len(new_chunk_ids)})"
        )

    # Try to load existing embeddings
    if embeddings_exist(collection_name, embeddings_dir):
        data = load_embeddings(collection_name, embeddings_dir)
        existing_embeddings = data["embeddings"]
        existing_chunk_ids = data["chunk_ids"]
        existing_metadata = data["metadata"]

        # Create mapping of existing chunks
        chunk_id_to_idx = {cid: idx for idx, cid in enumerate(existing_chunk_ids)}

        # Separate new chunks from updates
        update_indices = []
        update_embeddings = []
        truly_new_chunk_ids = []
        truly_new_embeddings = []

        for cid, emb in zip(new_chunk_ids, new_embeddings):
            if cid in chunk_id_to_idx:
                # Update existing embedding
                idx = chunk_id_to_idx[cid]
                update_indices.append(idx)
                update_embeddings.append(emb)
            else:
                # New chunk
                truly_new_chunk_ids.append(cid)
                truly_new_embeddings.append(emb)

        # Apply updates to existing embeddings
        if update_indices:
            for idx, emb in zip(update_indices, update_embeddings):
                existing_embeddings[idx] = emb
            logger.info(f"Updated {len(update_indices)} existing embeddings")

        # Append new embeddings
        if truly_new_embeddings:
            combined_embeddings = np.vstack([
                existing_embeddings,
                np.array(truly_new_embeddings)
            ])
            combined_chunk_ids = existing_chunk_ids + truly_new_chunk_ids
            logger.info(f"Added {len(truly_new_embeddings)} new embeddings")
        else:
            combined_embeddings = existing_embeddings
            combined_chunk_ids = existing_chunk_ids

        # Preserve model_name from existing if not provided
        if model_name is None:
            model_name = existing_metadata.get("model_name")

    else:
        # No existing embeddings, create new collection
        combined_embeddings = new_embeddings
        combined_chunk_ids = new_chunk_ids
        logger.info(f"Creating new embeddings collection with {len(new_embeddings)} embeddings")

    # Save combined embeddings
    save_embeddings(
        combined_embeddings,
        combined_chunk_ids,
        embeddings_dir,
        collection_name,
        model_name=model_name,
        normalized=normalized,
    )


def delete_embeddings(collection_name: str, embeddings_dir: Path) -> None:
    """
    Delete embeddings for a collection.

    Args:
        collection_name: Name of the embedding collection
        embeddings_dir: Directory containing embeddings files

    Raises:
        FileNotFoundError: If embeddings file doesn't exist

    Example:
        >>> delete_embeddings("my_collection", Path("data/embeddings"))
    """
    embeddings_dir = Path(embeddings_dir)
    embeddings_path = embeddings_dir / f"{collection_name}.npz"

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found for collection '{collection_name}' at {embeddings_path}"
        )

    embeddings_path.unlink()
    logger.info(f"Deleted embeddings for collection '{collection_name}'")


def get_all_embedding_collections(embeddings_dir: Path) -> List[str]:
    """
    Get list of all embedding collections.

    Args:
        embeddings_dir: Directory containing embeddings files

    Returns:
        List of collection names

    Example:
        >>> collections = get_all_embedding_collections(Path("data/embeddings"))
        >>> print(f"Found {len(collections)} collections")
    """
    embeddings_dir = Path(embeddings_dir)

    if not embeddings_dir.exists():
        return []

    collections = []
    for embeddings_file in embeddings_dir.glob("*.npz"):
        # Extract collection name from filename (remove .npz suffix)
        collection_name = embeddings_file.stem
        collections.append(collection_name)

    return sorted(collections)
