"""
Query Pipeline Module

Author: Nishit Attrey

This module provides the end-to-end query retrieval pipeline that:
1. Embeds user queries
2. Performs vector search
3. Ranks and formats results
4. Returns relevant document chunks

Key Features:
- Supports both brute-force and HNSW search
- Configurable top-k and score thresholds
- JSON-serializable output format
- Integration with existing embedding and vector store modules

Functions:
    create_query_pipeline: Factory function to create a query pipeline
    QueryPipeline: Main class for query processing

Example:
    >>> from src.query import create_query_pipeline
    >>> from src.embeddings import load_embedding_model
    >>> from src.vectorstore import BruteForceVectorStore
    >>>
    >>> model = load_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
    >>> vector_store = BruteForceVectorStore(dimension=384)
    >>>
    >>> # Add documents to vector store...
    >>>
    >>> pipeline = create_query_pipeline(model, vector_store, top_k=5)
    >>> results = pipeline.query("What is vector search?")
    >>>
    >>> for result in results:
    ...     print(f"Score: {result['score']:.3f} - {result['metadata']['text']}")
"""

import logging
import json
from typing import List, Dict, Any, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from src.embeddings.pipeline import embed_texts
from src.vectorstore.brute_force import BruteForceVectorStore
from src.vectorstore.hnsw import HNSWIndex

logger = logging.getLogger(__name__)


class QueryPipeline:
    """
    End-to-end query retrieval pipeline.

    This class orchestrates the full retrieval process:
    1. Query embedding generation
    2. Vector search in the index
    3. Result formatting and filtering

    Attributes:
        model: SentenceTransformer model for embedding queries
        vector_store: Vector store (BruteForce or HNSW) for search
        top_k: Default number of results to return
        min_score: Minimum similarity score threshold
        normalize: Whether to normalize query embeddings
    """

    def __init__(
        self,
        model: SentenceTransformer,
        vector_store: Union[BruteForceVectorStore, HNSWIndex],
        top_k: int = 5,
        min_score: float = 0.0,
        normalize: bool = True,
        ef_search: Optional[int] = None
    ):
        """
        Initialize query pipeline.

        Args:
            model: Loaded SentenceTransformer model (same as used for docs)
            vector_store: Initialized vector store with indexed documents
            top_k: Default number of results to return
                  Default: 5
            min_score: Minimum similarity score to include in results
                      Default: 0.0 (no filtering)
                      Useful values: 0.3-0.5 for cosine similarity
            normalize: Whether to normalize query embeddings
                      Default: True (should match document normalization)
            ef_search: HNSW search parameter (only used for HNSW indexes)
                      Default: None (uses index default)

        Note:
            - Model and normalization must match the settings used for documents
            - Vector store must be pre-populated with document embeddings
            - min_score filtering happens after search, reducing results
        """
        self.model = model
        self.vector_store = vector_store
        self.top_k = top_k
        self.min_score = min_score
        self.normalize = normalize
        self.ef_search = ef_search

        self.is_hnsw = isinstance(vector_store, HNSWIndex)

        logger.info(f"Query pipeline initialized")
        logger.info(f"Vector store type: {'HNSW' if self.is_hnsw else 'BruteForce'}")
        logger.info(f"Default top_k: {top_k}, min_score: {min_score}")

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query string.

        Args:
            query: User query text

        Returns:
            NumPy array of shape (dimension,) containing query embedding

        Example:
            >>> embedding = pipeline.embed_query("What is machine learning?")
            >>> print(embedding.shape)
            (384,)

        Note:
            - Uses the same model and normalization as documents
            - Batch size is 1 (single query)
            - No progress bar (single item)
        """
        logger.debug(f"Embedding query: {query[:50]}...")

        embeddings = embed_texts(
            self.model,
            [query],
            batch_size=1,
            normalize=self.normalize,
            show_progress=False
        )

        query_embedding = embeddings[0]

        logger.debug(f"Query embedding shape: {query_embedding.shape}")

        return query_embedding

    def search(
        self,
        query_embedding: np.ndarray,
        k: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for nearest neighbors using query embedding.

        Args:
            query_embedding: Query vector (shape: (dimension,))
            k: Number of results to return
              Default: None (uses self.top_k)
            min_score: Minimum score threshold
                      Default: None (uses self.min_score)

        Returns:
            List of result dictionaries, sorted by score (highest first)

        Note:
            - Result format differs between BruteForce and HNSW
            - This method normalizes to a consistent format
            - Scores are similarity (higher = better)
        """
        if k is None:
            k = self.top_k
        if min_score is None:
            min_score = self.min_score

        logger.debug(f"Searching for top {k} results (min_score={min_score})")

        if self.is_hnsw:
            results = self.vector_store.search(
                query_embedding,
                k=k,
                ef_search=self.ef_search
            )
        else:
            results = self.vector_store.search(
                query_embedding,
                k=k,
                return_vectors=True
            )
            for result in results:
                result['node_id'] = result.pop('index')

                score = result['score']
                if self.vector_store.similarity_metric == "cosine":
                    result['distance'] = 1.0 - score
                elif self.vector_store.similarity_metric == "dot":
                    result['distance'] = -score
                else:
                    result['distance'] = score

        if min_score > 0.0:
            original_count = len(results)
            results = [r for r in results if r['score'] >= min_score]
            filtered_count = original_count - len(results)
            if filtered_count > 0:
                logger.debug(f"Filtered {filtered_count} results below min_score={min_score}")

        logger.info(f"Found {len(results)} results")

        return results

    def query(
        self,
        query: str,
        k: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        End-to-end query processing: embed + search.

        This is the main entry point for querying the system.

        Args:
            query: User query text
            k: Number of results to return
              Default: None (uses self.top_k)
            min_score: Minimum similarity score threshold
                      Default: None (uses self.min_score)

        Returns:
            List of result dictionaries, each containing:
            - node_id: Index in vector store
            - score: Similarity score (higher = better)
            - distance: Distance metric (lower = closer)
            - metadata: Document metadata (chunk_id, text, etc.)
            - vector: Document embedding vector

        Example:
            >>> results = pipeline.query("How does HNSW work?", k=3)
            >>> for result in results:
            ...     print(f"{result['score']:.3f}: {result['metadata']['chunk_id']}")
            0.892: doc1_chunk_0
            0.834: doc2_chunk_5
            0.791: doc1_chunk_3

        Note:
            - Results are sorted by score (highest first)
            - Scores depend on similarity metric (cosine, dot, euclidean)
            - For cosine: score ∈ [-1, 1], typically [0, 1] for normalized
            - min_score filtering reduces the result list
        """
        logger.info(f"Processing query: {query[:100]}")

        query_embedding = self.embed_query(query)
        results = self.search(query_embedding, k=k, min_score=min_score)

        logger.info(f"Query completed: {len(results)} results returned")

        return results

    def format_results(
        self,
        results: List[Dict[str, Any]],
        include_vector: bool = False,
        include_distance: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Format search results for output (JSON-serializable).

        Args:
            results: Raw search results from query()
            include_vector: Whether to include embedding vectors
                           Default: False (vectors are large and rarely needed)
            include_distance: Whether to include distance metric
                             Default: True

        Returns:
            List of formatted result dictionaries

        Note:
            - Removes NumPy arrays (not JSON-serializable) unless requested
            - Converts types to Python primitives
            - Suitable for JSON output or API responses
        """
        formatted = []

        for result in results:
            formatted_result = {
                'node_id': int(result['node_id']),
                'score': float(result['score']),
                'metadata': result['metadata'],
            }

            if include_distance:
                formatted_result['distance'] = float(result['distance'])

            if include_vector:
                formatted_result['vector'] = result['vector'].tolist()

            formatted.append(formatted_result)

        return formatted

    def query_to_json(
        self,
        query: str,
        k: Optional[int] = None,
        min_score: Optional[float] = None,
        include_vector: bool = False,
        indent: int = 2
    ) -> str:
        """
        Query and return results as JSON string.

        Args:
            query: User query text
            k: Number of results
            min_score: Minimum score threshold
            include_vector: Whether to include embedding vectors
            indent: JSON indentation (0 for compact, 2 for readable)

        Returns:
            JSON string with query results

        Example:
            >>> json_output = pipeline.query_to_json("vector search", k=2)
            >>> print(json_output)
            {
              "query": "vector search",
              "results": [
                {
                  "node_id": 0,
                  "score": 0.924,
                  "metadata": {"chunk_id": "doc1_chunk_0", "text": "..."}
                },
                ...
              ]
            }
        """
        results = self.query(query, k=k, min_score=min_score)

        formatted_results = self.format_results(
            results,
            include_vector=include_vector
        )

        output = {
            'query': query,
            'num_results': len(formatted_results),
            'results': formatted_results
        }

        json_str = json.dumps(output, indent=indent if indent > 0 else None)

        return json_str


def create_query_pipeline(
    model: SentenceTransformer,
    vector_store: Union[BruteForceVectorStore, HNSWIndex],
    config: Optional[Dict[str, Any]] = None
) -> QueryPipeline:
    """
    Factory function to create a query pipeline.

    Args:
        model: Loaded SentenceTransformer model
        vector_store: Vector store with indexed documents
        config: Optional configuration dictionary
               If None, uses defaults
               Expected keys:
               - top_k: Number of results (default: 5)
               - min_score: Minimum score threshold (default: 0.0)
               - normalize: Whether to normalize queries (default: True)
               - ef_search: HNSW search parameter (default: None)

    Returns:
        Initialized QueryPipeline instance

    Example:
        >>> config = {'top_k': 10, 'min_score': 0.5}
        >>> pipeline = create_query_pipeline(model, vector_store, config)

    Note:
        This is the recommended way to create a pipeline from config.
    """
    default_config = {
        'top_k': 5,
        'min_score': 0.0,
        'normalize': True,
        'ef_search': None,
    }

    if config is not None:
        default_config.update(config)

    logger.info(f"Creating query pipeline with config: {default_config}")

    pipeline = QueryPipeline(
        model=model,
        vector_store=vector_store,
        top_k=default_config['top_k'],
        min_score=default_config['min_score'],
        normalize=default_config['normalize'],
        ef_search=default_config['ef_search']
    )

    return pipeline
