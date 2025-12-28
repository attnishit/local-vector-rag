"""
Embedding Generation Module

Author: Nishit Attrey

This module provides:
- Loading sentence-transformers models
- Batch embedding generation
- L2 normalization
- CPU-only execution support
- Deterministic embeddings
- Embeddings persistence (save/load from disk)

Main Functions:
    From model:
        - load_embedding_model: Load a sentence-transformers model
        - get_model_dimension: Get embedding dimension
        - validate_model_dimension: Validate dimension matches config

    From pipeline:
        - embed_texts: Generate embeddings for texts
        - embed_chunks: Generate embeddings for document chunks
        - normalize_embeddings: Apply L2 normalization
        - compute_embedding_statistics: Calculate embedding stats
        - verify_embedding_determinism: Verify deterministic behavior

    From persistence:
        - save_embeddings: Save embeddings to disk
        - load_embeddings: Load embeddings from disk
        - embeddings_exist: Check if embeddings exist
        - get_embeddings_for_chunks: Get embeddings for specific chunks
        - update_embeddings: Add new embeddings to collection
        - delete_embeddings: Delete embeddings collection
        - get_all_embedding_collections: List all collections

Default Model:
- sentence-transformers/all-MiniLM-L6-v2
- Embedding dimension: 384
- Multilingual support: Limited (English-focused)

Example:
    >>> from src.embeddings import load_embedding_model, embed_texts
    >>> model = load_embedding_model("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    >>> texts = ["Hello world", "Vector search"]
    >>> embeddings = embed_texts(model, texts, normalize=True)
    >>> print(embeddings.shape)
    (2, 384)
"""

from .model import (
    load_embedding_model,
    get_model_dimension,
    clear_model_cache,
    validate_model_dimension,
)

from .pipeline import (
    embed_texts,
    embed_chunks,
    normalize_embeddings,
    compute_embedding_statistics,
    verify_embedding_determinism,
)

from .persistence import (
    save_embeddings,
    load_embeddings,
    embeddings_exist,
    get_embeddings_for_chunks,
    update_embeddings,
    delete_embeddings,
    get_all_embedding_collections,
)

__all__ = [
    # Model functions
    "load_embedding_model",
    "get_model_dimension",
    "clear_model_cache",
    "validate_model_dimension",
    # Pipeline functions
    "embed_texts",
    "embed_chunks",
    "normalize_embeddings",
    "compute_embedding_statistics",
    "verify_embedding_determinism",
    # Persistence functions
    "save_embeddings",
    "load_embeddings",
    "embeddings_exist",
    "get_embeddings_for_chunks",
    "update_embeddings",
    "delete_embeddings",
    "get_all_embedding_collections",
]
