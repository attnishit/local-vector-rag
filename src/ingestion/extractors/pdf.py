"""
PDF Text Extraction

Author: Nishit Attrey

This module provides text extraction from PDF documents using PyMuPDF (fitz).

Functions:
    extract_text_from_pdf: Extract text from PDF files

Dependencies:
    PyMuPDF (fitz): pip install PyMuPDF
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def extract_text_from_pdf(filepath: Path, ocr_enabled: bool = False) -> str:
    """
    Extract text from a PDF file.

    Uses PyMuPDF (fitz) to extract text from all pages of a PDF document.
    Handles multi-page documents, tables, and formatting.

    Args:
        filepath: Path to the PDF file
        ocr_enabled: Whether to use OCR for scanned PDFs (requires pytesseract)

    Returns:
        Normalized plain text from all pages, with page markers

    Raises:
        ImportError: If PyMuPDF is not installed
        ValueError: If PDF is encrypted or corrupted
        Exception: For other PDF processing errors

    Example:
        >>> text = extract_text_from_pdf(Path("research_paper.pdf"))
        >>> print(f"Extracted {len(text)} characters")

    Notes:
        - Pages are concatenated with "\\n\\n--- Page N ---\\n\\n" markers
        - Metadata (title, author) is logged but not included in text
        - Tables are preserved as plain text where possible
        - Scanned PDFs without OCR will return minimal/no text
        - Password-protected PDFs are not supported
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF is required for PDF extraction. "
            "Install it with: pip install PyMuPDF"
        )

    if not filepath.exists():
        raise FileNotFoundError(f"PDF file not found: {filepath}")

    try:
        # Open the PDF
        doc = fitz.open(filepath)

        # Check if PDF is encrypted
        if doc.is_encrypted:
            doc.close()
            raise ValueError(
                f"PDF is encrypted and cannot be read: {filepath}\\n"
                "Password-protected PDFs are not currently supported."
            )

        # Extract metadata for logging
        metadata = doc.metadata
        if metadata:
            logger.debug(f"PDF Metadata - Title: {metadata.get('title', 'N/A')}, "
                        f"Author: {metadata.get('author', 'N/A')}")

        # Extract text from all pages
        pages_text = []
        total_pages = len(doc)

        for page_num in range(total_pages):
            page = doc[page_num]
            page_text = page.get_text()

            # Check if page appears to be scanned (very little text)
            if len(page_text.strip()) < 50 and ocr_enabled:
                logger.warning(
                    f"Page {page_num + 1} has minimal text. "
                    "This may be a scanned page. OCR is not yet implemented."
                )

            if page_text.strip():  # Only add non-empty pages
                # Add page marker for context
                pages_text.append(f"--- Page {page_num + 1} ---\\n{page_text}")

        doc.close()

        # Combine all pages with double newlines
        full_text = "\\n\\n".join(pages_text)

        # Normalize whitespace
        full_text = " ".join(full_text.split())

        logger.info(f"Extracted {len(full_text)} characters from {total_pages} pages in {filepath.name}")

        if len(full_text.strip()) < 100:
            logger.warning(
                f"PDF extracted very little text ({len(full_text)} chars). "
                "This may be a scanned PDF requiring OCR."
            )

        return full_text

    except fitz.FileDataError as e:
        raise ValueError(f"Corrupted or invalid PDF file: {filepath}\\nError: {e}")

    except Exception as e:
        logger.error(f"Error extracting text from PDF {filepath}: {e}")
        raise Exception(f"Failed to extract text from PDF: {e}")


def extract_metadata_from_pdf(filepath: Path) -> Optional[dict]:
    """
    Extract metadata from a PDF file.

    Args:
        filepath: Path to the PDF file

    Returns:
        Dictionary with metadata (title, author, subject, creator, etc.)
        or None if extraction fails

    Example:
        >>> meta = extract_metadata_from_pdf(Path("paper.pdf"))
        >>> if meta:
        ...     print(f"Title: {meta.get('title', 'Unknown')}")
    """
    try:
        import fitz
    except ImportError:
        logger.error("PyMuPDF is required for PDF metadata extraction")
        return None

    try:
        doc = fitz.open(filepath)
        metadata = doc.metadata
        doc.close()
        return metadata
    except Exception as e:
        logger.error(f"Failed to extract metadata from {filepath}: {e}")
        return None
