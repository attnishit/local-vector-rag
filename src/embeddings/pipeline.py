"""
Embedding Pipeline Module

Author: Nishit Attrey

This module provides batch embedding generation with L2 normalization.

Key Features:
- Batch processing for efficiency
- Optional L2 normalization
- Progress tracking
- Deterministic embeddings

Functions:
    embed_texts: Generate embeddings for a list of texts
    embed_chunks: Generate embeddings for document chunks
    normalize_embeddings: Apply L2 normalization to embeddings

Example:
    >>> from src.embeddings import load_embedding_model, embed_texts
    >>> model = load_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
    >>> texts = ["Hello world", "Vector search is powerful"]
    >>> embeddings = embed_texts(model, texts, normalize=True)
    >>> print(embeddings.shape)
    (2, 384)
"""

import logging
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Apply L2 normalization to embeddings.

    L2 normalization scales each embedding vector to unit length,
    ensuring that all vectors lie on the unit hypersphere. This is
    useful for:
    - Consistent dot product and cosine similarity
    - Numerical stability
    - Comparable magnitudes across different texts

    Args:
        embeddings: NumPy array of shape (n, dimension)
                   where n is number of embeddings

    Returns:
        L2-normalized embeddings of same shape

    Example:
        >>> embeddings = np.array([[3.0, 4.0], [1.0, 0.0]])
        >>> normalized = normalize_embeddings(embeddings)
        >>> np.linalg.norm(normalized[0])  # Should be 1.0
        1.0

    Note:
        L2 normalization: x_normalized = x / ||x||_2
        where ||x||_2 is the Euclidean norm (sqrt(sum(x^2)))
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings / norms

    logger.debug(f"Normalized {len(embeddings)} embeddings")

    return normalized


def embed_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 32,
    normalize: bool = True,
    show_progress: bool = False
) -> np.ndarray:
    """
    Generate embeddings for a list of texts.

    Processes texts in batches for memory efficiency and performance.
    Optionally normalizes embeddings to unit length.

    Args:
        model: Loaded SentenceTransformer model
        texts: List of text strings to embed
        batch_size: Number of texts to process in each batch
                   Default: 32 (good balance for CPU)
        normalize: Whether to apply L2 normalization
                  Default: True (recommended for vector search)
        show_progress: Whether to show progress bar
                      Default: False (use True for large datasets)

    Returns:
        NumPy array of shape (n, dimension) containing embeddings
        where n = len(texts)

    Example:
        >>> from src.embeddings.model import load_embedding_model
        >>> model = load_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
        >>> texts = ["First document", "Second document", "Third document"]
        >>> embeddings = embed_texts(model, texts, batch_size=2, normalize=True)
        >>> print(embeddings.shape)
        (3, 384)

    Note:
        - Batch size affects memory usage and speed
        - Larger batches are faster but use more memory
        - For CPU: batch_size=32 is a good default
        - For GPU: batch_size=128 or higher can be faster
        - Embeddings are deterministic: same text → same embedding
    """
    if not texts:
        logger.warning("Empty text list provided, returning empty array")
        return np.array([])

    logger.info(f"Generating embeddings for {len(texts)} texts")
    logger.info(f"Batch size: {batch_size}, Normalize: {normalize}")

    try:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=False
        )

        logger.info(f"Generated embeddings shape: {embeddings.shape}")

        if normalize:
            embeddings = normalize_embeddings(embeddings)
            logger.debug("Applied L2 normalization")

        return embeddings

    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        raise RuntimeError(
            f"Embedding generation failed. "
            f"Texts: {len(texts)}, Batch size: {batch_size}. "
            f"Error: {e}"
        )


def embed_chunks(
    model: SentenceTransformer,
    chunks: List[Dict[str, Any]],
    batch_size: int = 32,
    normalize: bool = True,
    show_progress: bool = False
) -> List[Dict[str, Any]]:
    """
    Generate embeddings for document chunks.

    Takes chunks from the ingestion pipeline and adds embeddings
    to each chunk dictionary.

    Args:
        model: Loaded SentenceTransformer model
        chunks: List of chunk dictionaries from chunker.chunk_text()
               Each chunk must have a 'text' field
        batch_size: Number of chunks to process in each batch
        normalize: Whether to apply L2 normalization
        show_progress: Whether to show progress bar

    Returns:
        List of chunk dictionaries with added 'embedding' field
        Each chunk['embedding'] is a NumPy array of shape (dimension,)

    Example:
        >>> from src.ingestion import chunk_text
        >>> from src.embeddings import load_embedding_model, embed_chunks
        >>>
        >>> # Create chunks
        >>> chunks = chunk_text("Sample document text", doc_id="doc1")
        >>>
        >>> # Generate embeddings
        >>> model = load_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
        >>> chunks_with_embeddings = embed_chunks(model, chunks)
        >>>
        >>> # Access embeddings
        >>> for chunk in chunks_with_embeddings:
        ...     print(f"{chunk['chunk_id']}: {chunk['embedding'].shape}")

    Note:
        - Original chunks are modified in-place with 'embedding' field
        - Returns the same chunks list for convenience
        - Preserves all original chunk metadata
    """
    if not chunks:
        logger.warning("Empty chunks list provided")
        return chunks

    logger.info(f"Generating embeddings for {len(chunks)} chunks")

    texts = [chunk['text'] for chunk in chunks]

    embeddings = embed_texts(
        model,
        texts,
        batch_size=batch_size,
        normalize=normalize,
        show_progress=show_progress
    )

    for chunk, embedding in zip(chunks, embeddings):
        chunk['embedding'] = embedding

    logger.info(f"Added embeddings to {len(chunks)} chunks")

    return chunks


def compute_embedding_statistics(embeddings: np.ndarray) -> Dict[str, Any]:
    """
    Compute statistics about a set of embeddings.

    Useful for validation and debugging.

    Args:
        embeddings: NumPy array of shape (n, dimension)

    Returns:
        Dictionary with statistics:
        - num_embeddings: Number of embedding vectors
        - dimension: Embedding dimension
        - mean_norm: Average L2 norm of embeddings
        - min_norm: Minimum L2 norm
        - max_norm: Maximum L2 norm
        - mean_value: Mean of all embedding values
        - std_value: Standard deviation of all embedding values

    Example:
        >>> embeddings = embed_texts(model, ["Hello", "World"])
        >>> stats = compute_embedding_statistics(embeddings)
        >>> print(f"Mean norm: {stats['mean_norm']:.3f}")
    """
    if embeddings.size == 0:
        return {
            'num_embeddings': 0,
            'dimension': 0,
            'mean_norm': 0.0,
            'min_norm': 0.0,
            'max_norm': 0.0,
            'mean_value': 0.0,
            'std_value': 0.0,
        }

    norms = np.linalg.norm(embeddings, axis=1)

    return {
        'num_embeddings': len(embeddings),
        'dimension': embeddings.shape[1] if embeddings.ndim > 1 else 0,
        'mean_norm': float(np.mean(norms)),
        'min_norm': float(np.min(norms)),
        'max_norm': float(np.max(norms)),
        'mean_value': float(np.mean(embeddings)),
        'std_value': float(np.std(embeddings)),
    }


def verify_embedding_determinism(
    model: SentenceTransformer,
    text: str,
    num_runs: int = 3
) -> bool:
    """
    Verify that embeddings are deterministic.

    Generates embeddings for the same text multiple times
    and checks that they are identical.

    Args:
        model: Loaded SentenceTransformer model
        text: Text to embed repeatedly
        num_runs: Number of times to generate embedding
                 Default: 3

    Returns:
        True if all embeddings are identical, False otherwise

    Example:
        >>> model = load_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
        >>> is_deterministic = verify_embedding_determinism(model, "Test text")
        >>> print(f"Deterministic: {is_deterministic}")
        True

    Note:
        This is a validation utility for testing.
        Embeddings should ALWAYS be deterministic for reproducibility.
    """
    embeddings = []

    for i in range(num_runs):
        emb = embed_texts(model, [text], batch_size=1, normalize=False)
        embeddings.append(emb[0])

    first_embedding = embeddings[0]

    for i, emb in enumerate(embeddings[1:], start=1):
        if not np.allclose(first_embedding, emb, rtol=1e-7, atol=1e-9):
            logger.warning(
                f"Embedding run {i+1} differs from run 1! "
                f"Max diff: {np.max(np.abs(first_embedding - emb))}"
            )
            return False

    logger.info(f"Embeddings are deterministic ({num_runs} runs)")
    return True
