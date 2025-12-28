"""
Document Loading Module

Author: Nishit Attrey

This module provides functionality to load documents from the filesystem
and prepare them for chunking.

Functions:
    load_document: Load a single document file
    load_documents_from_directory: Load all documents from a directory
    get_document_id: Generate document ID from filepath

Example:
    >>> from src.ingestion.loader import load_documents_from_directory
    >>> docs = load_documents_from_directory("data/raw")
    >>> for doc in docs:
    ...     print(f"{doc['doc_id']}: {len(doc['text'])} chars")
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from .extractors import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_markdown,
)

logger = logging.getLogger(__name__)


def detect_format(filepath: Path) -> str:
    """
    Detect document format based on file extension.

    Identifies the document type to route to the appropriate text extractor.

    Args:
        filepath: Path to the document file

    Returns:
        Normalized format identifier (e.g., "txt", "pdf", "docx", "markdown")

    Raises:
        ValueError: If file format is not supported

    Example:
        >>> detect_format(Path("document.pdf"))
        'pdf'
        >>> detect_format(Path("notes.md"))
        'markdown'
        >>> detect_format(Path("report.docx"))
        'docx'

    Supported formats:
        - .txt -> "txt"
        - .pdf -> "pdf"
        - .docx, .doc -> "docx"
        - .md, .markdown -> "markdown"
    """
    extension = filepath.suffix.lower()

    # Map file extensions to normalized format names
    format_map = {
        ".txt": "txt",
        ".pdf": "pdf",
        ".docx": "docx",
        ".doc": "docx",  # Treat legacy .doc as docx (requires special handling)
        ".md": "markdown",
        ".markdown": "markdown",
    }

    if extension not in format_map:
        supported = ", ".join(sorted(set(format_map.values())))
        raise ValueError(
            f"Unsupported file format: {extension}\n"
            f"Supported formats: {supported}\n"
            f"File: {filepath}"
        )

    return format_map[extension]


def get_document_id(filepath: Path) -> str:
    """
    Generate a document ID from a filepath.

    The document ID is the filename without extension, making it:
    - Human-readable
    - Stable across runs
    - Unique within a directory

    Args:
        filepath: Path to the document file

    Returns:
        Document ID string (filename without extension)

    Example:
        >>> get_document_id(Path("data/raw/sample_doc.txt"))
        'sample_doc'
        >>> get_document_id(Path("/path/to/report_2024.txt"))
        'report_2024'

    Note:
        If you have files with the same name in different subdirectories,
        consider modifying this to include parent directory in the ID.
    """
    return filepath.stem


def load_document(filepath: Path, encoding: str = "utf-8") -> Optional[Dict[str, Any]]:
    """
    Load a single document from a file.

    Automatically detects file format and routes to the appropriate extractor.
    Supports: .txt, .pdf, .docx, .doc, .md, .markdown

    Args:
        filepath: Path to the document file
        encoding: Text encoding to use for text files (default: utf-8)

    Returns:
        Document dictionary containing:
        - doc_id: Document identifier
        - filepath: Original file path (as string)
        - text: Document text content
        - size: Size in characters
        - format: Detected document format (txt, pdf, docx, markdown)
        Or None if file cannot be read

    Example:
        >>> doc = load_document(Path("data/raw/sample.pdf"))
        >>> print(f"{doc['doc_id']}: {doc['size']} characters ({doc['format']})")

    Note:
        Returns None on error rather than raising exception,
        allowing batch processing to continue even if some files fail.
    """
    try:
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return None

        if not filepath.is_file():
            logger.error(f"Not a file: {filepath}")
            return None

        # Detect document format
        try:
            doc_format = detect_format(filepath)
        except ValueError as e:
            logger.error(str(e))
            return None

        # Route to appropriate extractor based on format
        text = None

        if doc_format == "txt":
            # Plain text file - read directly
            try:
                with open(filepath, "r", encoding=encoding) as f:
                    text = f.read()
            except UnicodeDecodeError as e:
                logger.error(
                    f"Encoding error reading {filepath}: {e}\\n"
                    f"Try a different encoding (current: {encoding})"
                )
                return None

        elif doc_format == "pdf":
            # PDF file - use PyMuPDF extractor
            try:
                text = extract_text_from_pdf(filepath)
            except ImportError as e:
                logger.error(f"Missing dependency: {e}")
                return None
            except ValueError as e:
                logger.error(f"PDF error: {e}")
                return None

        elif doc_format == "docx":
            # DOCX/DOC file - use python-docx extractor
            try:
                text = extract_text_from_docx(filepath)
            except ImportError as e:
                logger.error(f"Missing dependency: {e}")
                return None
            except ValueError as e:
                logger.error(f"DOCX error: {e}")
                return None

        elif doc_format == "markdown":
            # Markdown file - use markdown extractor
            try:
                text = extract_text_from_markdown(filepath, encoding=encoding)
            except UnicodeDecodeError as e:
                logger.error(f"Encoding error reading {filepath}: {e}")
                return None

        else:
            logger.error(f"Unsupported format: {doc_format} for file {filepath}")
            return None

        # Check if extraction succeeded
        if text is None:
            logger.error(f"Text extraction failed for {filepath}")
            return None

        # Normalize whitespace (already done in extractors, but ensure consistency)
        text = text.strip()

        doc_id = get_document_id(filepath)

        document = {
            "doc_id": doc_id,
            "filepath": str(filepath),
            "text": text,
            "size": len(text),
            "format": doc_format,
        }

        logger.debug(
            f"Loaded document {doc_id}: {len(text)} characters "
            f"from {filepath} (format: {doc_format})"
        )

        return document

    except PermissionError:
        logger.error(f"Permission denied reading {filepath}")
        return None

    except Exception as e:
        logger.error(f"Unexpected error reading {filepath}: {e}")
        return None


def load_documents_from_directory(
    directory: str,
    supported_formats: List[str] = None,
    encoding: str = "utf-8",
    recursive: bool = False,
) -> List[Dict[str, Any]]:
    """
    Load all supported documents from a directory.

    Scans a directory for files matching supported formats and loads them.

    Args:
        directory: Path to directory containing documents
        supported_formats: List of file extensions to load (e.g., ['txt', 'md'])
                          If None, defaults to ['txt']
        encoding: Text encoding to use (default: utf-8)
        recursive: Whether to search subdirectories (default: False)

    Returns:
        List of document dictionaries, each containing:
        - doc_id: Document identifier
        - filepath: Original file path
        - text: Document text content
        - size: Size in characters

        Documents that failed to load are excluded from the list.

    Example:
        >>> docs = load_documents_from_directory("data/raw")
        >>> print(f"Loaded {len(docs)} documents")

        >>> # Load only txt files recursively
        >>> docs = load_documents_from_directory(
        ...     "data/raw",
        ...     supported_formats=['txt'],
        ...     recursive=True
        ... )

    Note:
        - Files are processed in sorted order for determinism
        - Failed loads are logged but don't stop processing
        - Empty files are loaded but will produce no chunks
    """
    if supported_formats is None:
        supported_formats = ["txt"]

    dir_path = Path(directory)

    if not dir_path.exists():
        logger.error(f"Directory not found: {directory}")
        return []

    if not dir_path.is_dir():
        logger.error(f"Not a directory: {directory}")
        return []

    pattern = "**/*" if recursive else "*"
    all_files = sorted(dir_path.glob(pattern))

    supported_exts = [f".{fmt.lower()}" for fmt in supported_formats]
    matching_files = [f for f in all_files if f.is_file() and f.suffix.lower() in supported_exts]

    logger.info(
        f"Found {len(matching_files)} files in {directory} "
        f"(formats: {supported_formats}, recursive: {recursive})"
    )

    documents = []
    failed_count = 0

    for filepath in matching_files:
        doc = load_document(filepath, encoding=encoding)
        if doc is not None:
            documents.append(doc)
        else:
            failed_count += 1

    logger.info(f"Loaded {len(documents)} documents from {directory} ({failed_count} failed)")

    if failed_count > 0:
        logger.warning(f"{failed_count} files failed to load. Check logs for details.")

    return documents


def document_statistics(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate statistics about loaded documents.

    Useful for debugging and validation.

    Args:
        documents: List of document dictionaries from load_documents_from_directory()

    Returns:
        Dictionary with statistics:
        - num_documents: Total number of documents
        - total_chars: Total characters across all documents
        - avg_doc_size: Average document size
        - min_doc_size: Size of smallest document
        - max_doc_size: Size of largest document

    Example:
        >>> docs = load_documents_from_directory("data/raw")
        >>> stats = document_statistics(docs)
        >>> print(f"Loaded {stats['num_documents']} documents, "
        ...       f"{stats['total_chars']} total characters")
    """
    if not documents:
        return {
            "num_documents": 0,
            "total_chars": 0,
            "avg_doc_size": 0,
            "min_doc_size": 0,
            "max_doc_size": 0,
        }

    sizes = [doc["size"] for doc in documents]

    return {
        "num_documents": len(documents),
        "total_chars": sum(sizes),
        "avg_doc_size": sum(sizes) / len(sizes),
        "min_doc_size": min(sizes),
        "max_doc_size": max(sizes),
    }
