"""
Markdown Text Extraction

Author: Nishit Attrey

This module provides text extraction from Markdown documents.

Functions:
    extract_text_from_markdown: Extract clean text from Markdown files

Dependencies:
    None required (uses regex-based parsing)
    Optional: markdown library for advanced parsing
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_text_from_markdown(
    filepath: Path,
    preserve_headers: bool = True,
    preserve_code_blocks: bool = True,
    preserve_links: bool = False,
    encoding: str = "utf-8"
) -> str:
    """
    Extract clean text from a Markdown file.

    Removes Markdown syntax while optionally preserving structural elements
    like headers and code blocks for better context.

    Args:
        filepath: Path to the Markdown file
        preserve_headers: Keep header markers (# Header -> "# Header") (default: True)
        preserve_code_blocks: Include code block content (default: True)
        preserve_links: Keep link URLs in footnotes (default: False)
        encoding: Text encoding (default: utf-8)

    Returns:
        Cleaned plain text with optional structural markers

    Raises:
        FileNotFoundError: If file doesn't exist
        UnicodeDecodeError: If encoding is incorrect
        Exception: For other reading errors

    Example:
        >>> text = extract_text_from_markdown(Path("README.md"))
        >>> print(f"Extracted {len(text)} characters")

    Notes:
        - Headers are preserved with "# " prefix for context
        - Links: [text](url) -> "text" (URL discarded by default)
        - Images: ![alt](url) -> "alt text"
        - Code blocks: ``` content ``` -> "content" or removed
        - Inline code: `code` -> "code"
        - Bold/italic: **text** -> "text"
        - Lists: Bullets removed, text preserved
        - Tables: Converted to plain text rows
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Markdown file not found: {filepath}")

    try:
        with open(filepath, "r", encoding=encoding) as f:
            content = f.read()

        # Store extracted links if needed
        links = []

        # Extract code blocks first (to handle separately)
        code_blocks = []
        if preserve_code_blocks:
            # Match fenced code blocks with optional language
            code_pattern = r'```(?:\w+)?\n(.*?)```'
            for match in re.finditer(code_pattern, content, re.DOTALL):
                code_blocks.append(match.group(1).strip())
            # Remove code blocks temporarily
            content = re.sub(code_pattern, "<<<CODE_BLOCK>>>", content, flags=re.DOTALL)
        else:
            # Remove code blocks entirely
            content = re.sub(r'```(?:\w+)?\n.*?```', '', content, flags=re.DOTALL)

        # Process headers
        if preserve_headers:
            # Keep headers with # prefix but clean the syntax
            # "### Header" -> "# Header"
            content = re.sub(r'^(#{1,6})\s+(.+)$', r'# \2', content, flags=re.MULTILINE)
        else:
            # Remove header markers but keep text
            content = re.sub(r'^#{1,6}\s+', '', content, flags=re.MULTILINE)

        # Extract and process links
        if preserve_links:
            # [text](url) -> "text [URL: url]"
            def link_replacer(match):
                text = match.group(1)
                url = match.group(2)
                links.append(url)
                return f"{text} [URL: {url}]"
            content = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', link_replacer, content)
        else:
            # [text](url) -> "text"
            content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)

        # Process images: ![alt](url) -> "alt"
        content = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', r'\1', content)

        # Remove inline code backticks: `code` -> "code"
        content = re.sub(r'`([^`]+)`', r'\1', content)

        # Remove bold/italic markers
        content = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', content)  # ***bold italic***
        content = re.sub(r'\*\*(.+?)\*\*', r'\1', content)      # **bold**
        content = re.sub(r'\*(.+?)\*', r'\1', content)          # *italic*
        content = re.sub(r'___(.+?)___', r'\1', content)        # ___bold italic___
        content = re.sub(r'__(.+?)__', r'\1', content)          # __bold__
        content = re.sub(r'_(.+?)_', r'\1', content)            # _italic_

        # Remove strikethrough: ~~text~~ -> "text"
        content = re.sub(r'~~(.+?)~~', r'\1', content)

        # Process lists: remove bullet markers
        content = re.sub(r'^\s*[-*+]\s+', '', content, flags=re.MULTILINE)  # Unordered
        content = re.sub(r'^\s*\d+\.\s+', '', content, flags=re.MULTILINE)  # Ordered

        # Process blockquotes: remove ">" marker
        content = re.sub(r'^\s*>\s+', '', content, flags=re.MULTILINE)

        # Process horizontal rules: remove entirely
        content = re.sub(r'^[\-*_]{3,}$', '', content, flags=re.MULTILINE)

        # Process tables: convert to plain text rows
        content = _process_markdown_tables(content)

        # Restore code blocks if preserved
        if preserve_code_blocks and code_blocks:
            for code in code_blocks:
                content = content.replace("<<<CODE_BLOCK>>>", f"\\n[CODE]\\n{code}\\n[/CODE]\\n", 1)

        # Remove HTML tags if any
        content = re.sub(r'<[^>]+>', '', content)

        # Normalize whitespace
        content = re.sub(r'\n{3,}', '\\n\\n', content)  # Max 2 consecutive newlines
        content = " ".join(content.split())

        logger.info(f"Extracted {len(content)} characters from Markdown file {filepath.name}")

        if len(content.strip()) < 10:
            logger.warning(f"Markdown extracted very little text ({len(content)} chars)")

        return content

    except UnicodeDecodeError as e:
        raise UnicodeDecodeError(
            e.encoding,
            e.object,
            e.start,
            e.end,
            f"Failed to decode {filepath} with encoding {encoding}. Try a different encoding."
        )

    except Exception as e:
        logger.error(f"Error extracting text from Markdown {filepath}: {e}")
        raise Exception(f"Failed to extract text from Markdown: {e}")


def _process_markdown_tables(content: str) -> str:
    """
    Convert Markdown tables to plain text rows.

    Args:
        content: Markdown content with tables

    Returns:
        Content with tables converted to plain text

    Example:
        | Col1 | Col2 |
        |------|------|
        | A    | B    |

        Becomes: "Col1 Col2 A B"
    """
    # Match table rows (lines with | separators)
    table_pattern = r'^(\|.+\|)\s*$'

    lines = content.split('\\n')
    processed_lines = []

    in_table = False
    for line in lines:
        # Check if this is a table row
        if re.match(table_pattern, line):
            # Skip separator rows (|---|---|)
            if re.match(r'^\|[\s\-:|]+\|$', line):
                continue

            in_table = True
            # Extract cell content: | A | B | -> "A B"
            cells = [cell.strip() for cell in line.split('|') if cell.strip()]
            processed_lines.append(' '.join(cells))
        else:
            if in_table:
                # Add spacing after table
                processed_lines.append('')
                in_table = False
            processed_lines.append(line)

    return '\\n'.join(processed_lines)


def extract_headers_from_markdown(filepath: Path, encoding: str = "utf-8") -> list:
    """
    Extract just the headers from a Markdown file.

    Useful for creating a table of contents or understanding document structure.

    Args:
        filepath: Path to the Markdown file
        encoding: Text encoding (default: utf-8)

    Returns:
        List of tuples: (level, header_text)
        where level is 1-6 (number of # symbols)

    Example:
        >>> headers = extract_headers_from_markdown(Path("README.md"))
        >>> for level, text in headers:
        ...     print(f"{'  ' * (level-1)}{text}")
    """
    try:
        with open(filepath, "r", encoding=encoding) as f:
            content = f.read()

        headers = []
        header_pattern = r'^(#{1,6})\s+(.+)$'

        for match in re.finditer(header_pattern, content, re.MULTILINE):
            level = len(match.group(1))
            text = match.group(2).strip()
            headers.append((level, text))

        return headers

    except Exception as e:
        logger.error(f"Failed to extract headers from {filepath}: {e}")
        return []
