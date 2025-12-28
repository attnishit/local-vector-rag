"""
Collection Management System

Author: Nishit Attrey

This module provides a high-level interface for managing document collections
with persistent storage. A collection represents a searchable set of documents
with their chunks, embeddings, and vector index.

Key Features:
- Create collections from documents
- Incremental document addition (no full rebuild)
- Automatic persistence of all components
- Multiple algorithm support (brute-force, HNSW)
- Collection metadata tracking

Components of a Collection:
- Chunks: Processed document chunks (data/processed/<doc_id>_chunks.json)
- Embeddings: Vector embeddings (data/embeddings/<collection>.npz)
- Index: Vector search index (data/indexes/<algorithm>/<collection>/)
- Metadata: Collection info (data/collections.json)

Example:
    >>> from src.collection import create_collection, load_collection
    >>> from pathlib import Path
    >>>
    >>> # Create a new collection
    >>> collection = create_collection(
    ...     name="my_docs",
    ...     documents_dir=Path("data/raw"),
    ...     algorithm="hnsw"
    ... )
    >>>
    >>> # Search
    >>> results = collection.search("machine learning", k=5)
    >>>
    >>> # Load existing collection
    >>> collection = load_collection("my_docs")
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import numpy as np

# Import all the components
from .ingestion import (
    load_document,
    load_documents_from_directory,
    chunk_text,
    save_chunks,
    load_chunks,
    chunks_exist,
    should_rechunk,
    get_all_chunked_documents,
)
from .embeddings import (
    load_embedding_model,
    embed_texts,
    save_embeddings,
    load_embeddings,
    embeddings_exist,
    get_embeddings_for_chunks,
    update_embeddings,
)
from .vectorstore import BruteForceVectorStore, HNSWIndex, create_hnsw_index
from .config import load_config

logger = logging.getLogger(__name__)


class Collection:
    """
    A searchable collection of documents with persistent storage.

    Manages the entire pipeline: documents → chunks → embeddings → vector index

    Attributes:
        name: Collection name
        algorithm: Vector search algorithm ("brute_force" or "hnsw")
        config: Configuration dictionary
        model: Embedding model
        index: Vector store (BruteForceVectorStore or HNSWIndex)
        chunks_dir: Directory for chunks storage
        embeddings_dir: Directory for embeddings storage
        indexes_dir: Directory for index storage
    """

    def __init__(
        self,
        name: str,
        algorithm: str = "brute_force",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a collection.

        Args:
            name: Collection name
            algorithm: "brute_force" or "hnsw"
            config: Configuration dict (loads from config.yaml if not provided)
        """
        self.name = name
        self.algorithm = algorithm
        self.config = config or load_config()

        # Load embedding model
        model_name = self.config["embeddings"]["model_name"]
        device = self.config["embeddings"].get("device", "cpu")
        self.model = load_embedding_model(model_name, device=device)

        # Set up directories
        base_dir = Path(self.config["paths"]["data_dir"])
        self.chunks_dir = base_dir / "processed"
        self.embeddings_dir = base_dir / "embeddings"
        self.indexes_dir = base_dir / "indexes" / algorithm

        # Create directories
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.indexes_dir.mkdir(parents=True, exist_ok=True)

        # Index will be created/loaded separately
        self.index: Optional[Any] = None

        logger.info(f"Initialized collection '{name}' with algorithm '{algorithm}'")

    def add_documents(
        self,
        documents_dir: Optional[Path] = None,
        document_paths: Optional[List[Path]] = None,
        show_progress: bool = False,
    ) -> Dict[str, Any]:
        """
        Add documents to the collection (incremental).

        Processes documents, generates embeddings, and updates the index.
        Only processes new or changed documents.

        Args:
            documents_dir: Directory containing documents to add
            document_paths: List of specific document paths to add
            show_progress: Whether to show progress bars

        Returns:
            Statistics about the operation

        Example:
            >>> stats = collection.add_documents(Path("data/raw/new_docs"))
            >>> print(f"Added {stats['new_documents']} documents")
        """
        # Get supported formats from config
        supported_formats = self.config["ingestion"].get("supported_formats", ["txt"])

        # Load documents
        if documents_dir:
            docs = load_documents_from_directory(
                documents_dir,
                supported_formats=supported_formats,
                recursive=False
            )
        elif document_paths:
            docs = [load_document(path) for path in document_paths]
        else:
            raise ValueError("Must provide either documents_dir or document_paths")

        logger.info(f"Processing {len(docs)} documents for collection '{self.name}'")

        # Track statistics
        stats = {
            "total_documents": len(docs),
            "new_documents": 0,
            "updated_documents": 0,
            "new_chunks": 0,
            "new_embeddings": 0,
        }

        # Get chunk config
        chunk_size = self.config["ingestion"]["chunk_size"]
        chunk_overlap = self.config["ingestion"]["chunk_overlap"]
        chunk_config = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}

        all_new_chunks = []
        all_new_embeddings = []
        all_new_chunk_ids = []

        # Process each document
        for doc in docs:
            doc_id = doc["doc_id"]
            source_file = Path(doc.get("filepath", ""))

            # Check if we need to rechunk
            if should_rechunk(doc_id, source_file, self.chunks_dir, chunk_config):
                # Chunk the document
                chunks = chunk_text(
                    doc["text"],
                    doc_id,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )

                # Save chunks
                save_chunks(
                    chunks,
                    self.chunks_dir,
                    source_file=source_file,
                    chunk_config=chunk_config,
                )

                stats["new_documents"] += 1
                stats["new_chunks"] += len(chunks)

                # Collect chunks for embedding
                all_new_chunks.extend(chunks)
            else:
                logger.debug(f"Chunks up-to-date for document '{doc_id}', skipping")

        # Generate embeddings for new chunks
        if all_new_chunks:
            logger.info(f"Generating embeddings for {len(all_new_chunks)} new chunks")

            # Extract texts and chunk_ids
            texts = [chunk["text"] for chunk in all_new_chunks]
            chunk_ids = [chunk["chunk_id"] for chunk in all_new_chunks]

            # Generate embeddings
            batch_size = self.config["embeddings"]["batch_size"]
            normalize = self.config["embeddings"]["normalize"]

            embeddings = embed_texts(
                self.model,
                texts,
                batch_size=batch_size,
                normalize=normalize,
                show_progress=show_progress
            )

            all_new_embeddings = embeddings
            all_new_chunk_ids = chunk_ids
            stats["new_embeddings"] = len(embeddings)

            # Update embeddings collection
            model_name = self.config["embeddings"]["model_name"]
            update_embeddings(
                all_new_embeddings,
                all_new_chunk_ids,
                self.name,
                self.embeddings_dir,
                model_name=model_name,
                normalized=normalize,
            )

            logger.info(f"Updated embeddings collection with {len(embeddings)} new embeddings")

            # Update vector index
            self._update_index(all_new_embeddings, all_new_chunks)

        logger.info(f"Completed adding documents to collection '{self.name}': {stats}")
        return stats

    def _update_index(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]) -> None:
        """
        Update the vector index with new embeddings.

        Args:
            embeddings: New embeddings to add
            chunks: Corresponding chunks with metadata
        """
        # Create index if it doesn't exist
        if self.index is None:
            self._load_or_create_index()

        # Add embeddings to index
        metadata_list = [{"chunk_id": c["chunk_id"], "text": c["text"], "doc_id": c["doc_id"]} for c in chunks]

        if self.algorithm == "brute_force":
            self.index.add_batch(embeddings, metadata=metadata_list)
        elif self.algorithm == "hnsw":
            for emb, meta in zip(embeddings, metadata_list):
                self.index.insert(emb, metadata=meta)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # Save index
        index_path = self.indexes_dir / self.name
        self.index.save(index_path, index_name=self.name)

        logger.info(f"Updated and saved index with {len(embeddings)} new vectors")

    def _load_or_create_index(self) -> None:
        """Load existing index or create a new one."""
        index_path = self.indexes_dir / self.name

        dimension = self.config["embeddings"]["dimension"]
        similarity_metric = self.config["vectorstore"]["similarity_metric"]
        normalized = self.config["embeddings"]["normalize"]

        if self.algorithm == "brute_force":
            # Try to load existing index
            if index_path.exists() and (index_path / "index_info.json").exists():
                try:
                    self.index = BruteForceVectorStore.load(index_path)
                    logger.info(f"Loaded existing brute-force index with {len(self.index)} vectors")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load index: {e}, creating new one")

            # Create new index
            self.index = BruteForceVectorStore(
                dimension=dimension,
                similarity_metric=similarity_metric,
                normalized=normalized
            )
            logger.info("Created new brute-force index")

        elif self.algorithm == "hnsw":
            # Try to load existing index
            if index_path.exists() and (index_path / "index_info.json").exists():
                try:
                    self.index = HNSWIndex.load(index_path)
                    logger.info(f"Loaded existing HNSW index with {len(self.index)} vectors")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load index: {e}, creating new one")

            # Create new index
            m = self.config["vectorstore"]["hnsw"]["m"]
            ef_construction = self.config["vectorstore"]["hnsw"]["ef_construction"]
            ef_search = self.config["vectorstore"]["hnsw"]["ef_search"]

            self.index = create_hnsw_index(
                dimension=dimension,
                m=m,
                ef_construction=ef_construction,
                ef_search=ef_search,
                similarity_metric=similarity_metric,
                normalized=normalized
            )
            logger.info("Created new HNSW index")

        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def search(
        self,
        query: str,
        k: int = 5,
        ef_search: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search the collection for similar chunks.

        Args:
            query: Query text
            k: Number of results to return
            ef_search: HNSW ef_search parameter (only for HNSW)

        Returns:
            List of result dictionaries with scores and metadata

        Example:
            >>> results = collection.search("machine learning", k=3)
            >>> for result in results:
            ...     print(f"{result['score']:.3f}: {result['metadata']['text'][:50]}")
        """
        # Ensure index is loaded
        if self.index is None:
            self._load_or_create_index()

        # Embed query
        normalize = self.config["embeddings"]["normalize"]
        query_embedding = embed_texts(
            self.model,
            [query],
            batch_size=1,
            normalize=normalize,
            show_progress=False
        )[0]

        # Search index
        if self.algorithm == "brute_force":
            results = self.index.search(query_embedding, k=k)
        elif self.algorithm == "hnsw":
            if ef_search is None:
                ef_search = self.config["vectorstore"]["hnsw"]["ef_search"]
            results = self.index.search(query_embedding, k=k, ef_search=ef_search)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        return results

    def info(self) -> Dict[str, Any]:
        """
        Get information about the collection.

        Returns:
            Dictionary with collection statistics
        """
        # Count documents
        doc_ids = get_all_chunked_documents(self.chunks_dir)
        # Filter to only docs in this collection (could be improved with metadata)
        num_documents = len(doc_ids)

        # Count chunks (from all docs - approximate)
        num_chunks = 0
        for doc_id in doc_ids:
            try:
                chunk_data = load_chunks(doc_id, self.chunks_dir)
                num_chunks += chunk_data["num_chunks"]
            except:
                pass

        # Get embeddings info
        num_embeddings = 0
        if embeddings_exist(self.name, self.embeddings_dir):
            emb_data = load_embeddings(self.name, self.embeddings_dir)
            num_embeddings = emb_data["metadata"]["num_embeddings"]

        # Get index info
        index_info = {}
        if self.index:
            index_info = self.index.statistics()

        return {
            "name": self.name,
            "algorithm": self.algorithm,
            "num_documents": num_documents,
            "num_chunks": num_chunks,
            "num_embeddings": num_embeddings,
            "index": index_info,
        }

    def delete(self) -> None:
        """
        Delete all data for this collection.

        WARNING: This is irreversible!
        """
        from .embeddings.persistence import delete_embeddings

        logger.warning(f"Deleting collection '{self.name}'")

        # Delete embeddings
        try:
            delete_embeddings(self.name, self.embeddings_dir)
        except FileNotFoundError:
            pass

        # Delete index
        index_path = self.indexes_dir / self.name
        if index_path.exists():
            import shutil
            shutil.rmtree(index_path)

        # Note: We don't delete chunks because they might be shared across collections
        # Users can manually clean up data/processed/ if needed

        logger.info(f"Deleted collection '{self.name}'")


# Module-level functions

def create_collection(
    name: str,
    documents_dir: Optional[Path] = None,
    document_paths: Optional[List[Path]] = None,
    algorithm: str = "brute_force",
    config: Optional[Dict[str, Any]] = None,
    show_progress: bool = False,
) -> Collection:
    """
    Create a new collection from documents.

    Args:
        name: Collection name
        documents_dir: Directory containing documents
        document_paths: List of document paths
        algorithm: "brute_force" or "hnsw"
        config: Optional configuration
        show_progress: Show progress bars

    Returns:
        Collection instance

    Example:
        >>> collection = create_collection(
        ...     "my_docs",
        ...     documents_dir=Path("data/raw"),
        ...     algorithm="hnsw"
        ... )
    """
    collection = Collection(name=name, algorithm=algorithm, config=config)

    if documents_dir or document_paths:
        collection.add_documents(
            documents_dir=documents_dir,
            document_paths=document_paths,
            show_progress=show_progress
        )

    # Save collection metadata
    _save_collection_metadata(collection)

    return collection


def load_collection(
    name: str,
    algorithm: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Collection:
    """
    Load an existing collection.

    Args:
        name: Collection name
        algorithm: Algorithm ("brute_force" or "hnsw"), auto-detected if None
        config: Optional configuration

    Returns:
        Collection instance

    Raises:
        ValueError: If collection doesn't exist

    Example:
        >>> collection = load_collection("my_docs")
        >>> results = collection.search("query", k=5)
    """
    # Try to get algorithm from metadata
    if algorithm is None:
        metadata = get_collection_metadata(name, config=config)
        if metadata:
            algorithm = metadata.get("algorithm", "brute_force")
        else:
            algorithm = "brute_force"

    collection = Collection(name=name, algorithm=algorithm, config=config)
    collection._load_or_create_index()

    return collection


def list_collections(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    List all collections.

    Args:
        config: Optional configuration

    Returns:
        List of collection metadata dictionaries

    Example:
        >>> collections = list_collections()
        >>> for coll in collections:
        ...     print(f"{coll['name']}: {coll['num_embeddings']} embeddings")
    """
    if config is None:
        config = load_config()

    metadata_file = Path(config["paths"]["data_dir"]) / "collections.json"

    if not metadata_file.exists():
        return []

    with open(metadata_file, "r") as f:
        data = json.load(f)

    return list(data.get("collections", {}).values())


def delete_collection(name: str, config: Optional[Dict[str, Any]] = None) -> None:
    """
    Delete a collection.

    Args:
        name: Collection name
        config: Optional configuration

    Example:
        >>> delete_collection("old_collection")
    """
    collection = load_collection(name, config=config)
    collection.delete()

    # Remove from metadata
    _remove_collection_metadata(name, config=config)


def get_collection_metadata(
    name: str,
    config: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Get metadata for a collection without loading it.

    Args:
        name: Collection name
        config: Optional configuration

    Returns:
        Metadata dictionary or None if not found
    """
    collections = list_collections(config=config)

    for coll in collections:
        if coll["name"] == name:
            return coll

    return None


def _save_collection_metadata(collection: Collection) -> None:
    """Save collection metadata to collections.json"""
    config = collection.config
    metadata_file = Path(config["paths"]["data_dir"]) / "collections.json"

    # Load existing metadata
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            data = json.load(f)
    else:
        data = {"collections": {}}

    # Update metadata
    info = collection.info()
    data["collections"][collection.name] = {
        "name": collection.name,
        "algorithm": collection.algorithm,
        "created_at": datetime.now().isoformat(),
        "num_documents": info["num_documents"],
        "num_chunks": info["num_chunks"],
        "num_embeddings": info["num_embeddings"],
    }

    # Save
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_file, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved metadata for collection '{collection.name}'")


def _remove_collection_metadata(name: str, config: Optional[Dict[str, Any]] = None) -> None:
    """Remove collection from metadata"""
    if config is None:
        config = load_config()

    metadata_file = Path(config["paths"]["data_dir"]) / "collections.json"

    if not metadata_file.exists():
        return

    with open(metadata_file, "r") as f:
        data = json.load(f)

    if name in data.get("collections", {}):
        del data["collections"][name]

    with open(metadata_file, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Removed metadata for collection '{name}'")
