"""
Embedding Model Management Module

Author: Nishit Attrey

This module handles loading and managing sentence-transformers models
for generating text embeddings.

Key Features:
- Load sentence-transformers models
- CPU-only execution support
- Model caching for reuse
- Dimension validation

Functions:
    load_embedding_model: Load a sentence-transformers model
    get_model_dimension: Get the embedding dimension of a model

Example:
    >>> from src.embeddings.model import load_embedding_model
    >>> model = load_embedding_model("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    >>> dimension = model.get_sentence_embedding_dimension()
    >>> print(f"Embedding dimension: {dimension}")
"""

import logging
from typing import Optional
from sentence_transformers import SentenceTransformer

# Get logger for this module
logger = logging.getLogger(__name__)

_model_cache = {}


def load_embedding_model(
    model_name: str,
    device: str = "cpu",
    cache: bool = True
) -> SentenceTransformer:
    """
    Load a sentence-transformers embedding model.

    This function loads a pre-trained sentence-transformers model
    for generating text embeddings. Models are cached by default
    to avoid reloading on subsequent calls.

    Args:
        model_name: Name of the sentence-transformers model
                   Examples: "sentence-transformers/all-MiniLM-L6-v2",
                            "sentence-transformers/all-mpnet-base-v2"
        device: Device to load model on ("cpu" or "cuda")
                Default: "cpu" for local-first execution
        cache: Whether to cache the model for reuse
               Default: True (recommended for performance)

    Returns:
        Loaded SentenceTransformer model ready for encoding

    Example:
        >>> model = load_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
        >>> embeddings = model.encode(["Hello world", "Test sentence"])
        >>> print(embeddings.shape)
        (2, 384)

    Note:
        - First call downloads the model (cached by Hugging Face)
        - Subsequent calls with cache=True reuse the loaded model
        - CPU execution works on any machine without GPU
        - Model files are stored in ~/.cache/torch/sentence_transformers/
    """
    cache_key = f"{model_name}_{device}"

    if cache and cache_key in _model_cache:
        logger.debug(f"Using cached model: {model_name} on {device}")
        return _model_cache[cache_key]

    logger.info(f"Loading embedding model: {model_name}")
    logger.info(f"Device: {device}")

    try:
        model = SentenceTransformer(model_name, device=device)

        dimension = model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded successfully")
        logger.info(f"Embedding dimension: {dimension}")

        if cache:
            _model_cache[cache_key] = model
            logger.debug(f"Model cached with key: {cache_key}")

        return model

    except Exception as e:
        logger.error(f"Failed to load embedding model: {model_name}")
        logger.error(f"Error: {e}")
        raise RuntimeError(
            f"Failed to load embedding model '{model_name}'. "
            f"Please check the model name and network connection. "
            f"Error: {e}"
        )


def get_model_dimension(model: SentenceTransformer) -> int:
    """
    Get the embedding dimension of a loaded model.

    Args:
        model: Loaded SentenceTransformer model

    Returns:
        Integer dimension of the embedding vectors

    Example:
        >>> model = load_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
        >>> dim = get_model_dimension(model)
        >>> print(dim)
        384
    """
    return model.get_sentence_embedding_dimension()


def clear_model_cache() -> None:
    """
    Clear the model cache.

    Useful for freeing memory or forcing model reload.
    Should rarely be needed in normal operation.

    Example:
        >>> from src.embeddings.model import clear_model_cache
        >>> clear_model_cache()
        >>> # Next load_embedding_model() call will reload from disk
    """
    global _model_cache
    _model_cache.clear()
    logger.info("Model cache cleared")


def validate_model_dimension(
    model: SentenceTransformer,
    expected_dimension: int
) -> None:
    """
    Validate that a model produces embeddings of the expected dimension.

    Args:
        model: Loaded SentenceTransformer model
        expected_dimension: Expected embedding dimension

    Raises:
        ValueError: If dimension doesn't match expected value

    Example:
        >>> model = load_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
        >>> validate_model_dimension(model, 384)  # Passes
        >>> validate_model_dimension(model, 768)  # Raises ValueError
    """
    actual_dimension = get_model_dimension(model)

    if actual_dimension != expected_dimension:
        raise ValueError(
            f"Model dimension mismatch! "
            f"Expected: {expected_dimension}, "
            f"Actual: {actual_dimension}. "
            f"Please check your configuration."
        )

    logger.debug(f"Model dimension validated: {actual_dimension}")
