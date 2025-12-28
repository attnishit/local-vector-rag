"""
Document Format Extractors

Author: Nishit Attrey

This module provides format-specific text extraction for various document types.

Modules:
    pdf: PDF text extraction using PyMuPDF
    docx: Word document extraction using python-docx
    markdown: Markdown parsing and text extraction
"""

from .pdf import extract_text_from_pdf
from .docx import extract_text_from_docx
from .markdown import extract_text_from_markdown

__all__ = [
    "extract_text_from_pdf",
    "extract_text_from_docx",
    "extract_text_from_markdown",
]
