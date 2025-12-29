"""
Citation System

Author: Nishit Attrey

This module handles citation injection and formatting for RAG responses.
When retrieving context chunks, this module adds citation markers ([1], [2])
that LLMs can reference in their answers.

Key Features:
- Add citation markers to retrieved chunks
- Track which sources are cited in responses
- Format sources for display
- Extract citations from LLM responses

Example:
    >>> from src.generation.citations import prepare_context_with_citations
    >>> results = collection.search("What is HNSW?", k=3)
    >>> context, sources = prepare_context_with_citations(results)
    >>> print(context)
    [1] HNSW is a graph-based algorithm...
    [2] The algorithm uses a hierarchical structure...
    [3] Vector search with HNSW provides fast...
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Set

logger = logging.getLogger(__name__)


def prepare_context_with_citations(
    results: List[Dict[str, Any]],
    max_context_length: int = 4000
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Prepare context string with citation markers from search results.

    Takes search results from Collection.search() and formats them with
    numbered citation markers that the LLM can reference.

    Args:
        results: Search results from Collection.search()
                Each result has: {score, metadata: {text, chunk_id, doc_id}}
        max_context_length: Maximum context length in characters

    Returns:
        Tuple of (context_string, sources_list)

        context_string: Formatted context with [1], [2] markers
        sources_list: List of source dicts with citation info

    Example:
        >>> results = [
        ...     {
        ...         'score': 0.92,
        ...         'metadata': {
        ...             'text': 'HNSW is a graph algorithm...',
        ...             'chunk_id': 'doc1_chunk_0',
        ...             'doc_id': 'doc1'
        ...         }
        ...     }
        ... ]
        >>> context, sources = prepare_context_with_citations(results)
        >>> print(context)
        [1] HNSW is a graph algorithm...
        >>> print(sources)
        [{'citation_num': 1, 'chunk_id': 'doc1_chunk_0', ...}]
    """
    if not results:
        logger.warning("No results provided for context preparation")
        return "", []

    context_parts = []
    sources = []
    total_length = 0

    for i, result in enumerate(results, 1):
        # Extract metadata
        metadata = result.get('metadata', {})
        text = metadata.get('text', '')
        chunk_id = metadata.get('chunk_id', f'unknown_chunk_{i}')
        doc_id = metadata.get('doc_id', 'unknown_doc')
        score = result.get('score', 0.0)

        # Format with citation marker
        cited_text = f"[{i}] {text}"

        # Check if adding this would exceed limit
        if max_context_length and (total_length + len(cited_text)) > max_context_length:
            logger.info(
                f"Context length limit reached at {i-1} chunks "
                f"({total_length} chars)"
            )
            break

        context_parts.append(cited_text)
        total_length += len(cited_text)

        # Track source info
        sources.append({
            'citation_num': i,
            'chunk_id': chunk_id,
            'doc_id': doc_id,
            'text': text,
            'score': score,
            'cited': False,  # Will be updated after LLM response
        })

    # Join with double newlines for readability
    context_string = "\n\n".join(context_parts)

    logger.info(
        f"Prepared context: {len(sources)} chunks, {len(context_string)} chars"
    )

    return context_string, sources


def extract_citations_from_response(response_text: str) -> Set[int]:
    """
    Extract citation numbers from LLM response.

    Finds all [N] patterns in the response where N is a number.

    Args:
        response_text: Generated answer from LLM

    Returns:
        Set of citation numbers found in the response

    Example:
        >>> response = "HNSW [1] is fast. It uses graphs [2] for search."
        >>> citations = extract_citations_from_response(response)
        >>> print(citations)
        {1, 2}
    """
    # Pattern: [N] where N is a number
    pattern = r'\[(\d+)\]'
    matches = re.findall(pattern, response_text)

    # Convert to integers
    citation_nums = {int(num) for num in matches}

    logger.debug(f"Extracted citations: {sorted(citation_nums)}")

    return citation_nums


def mark_cited_sources(
    sources: List[Dict[str, Any]],
    response_text: str
) -> List[Dict[str, Any]]:
    """
    Mark which sources were actually cited in the response.

    Updates the 'cited' field in sources list based on what
    citations appear in the LLM's response.

    Args:
        sources: List of source dictionaries from prepare_context_with_citations()
        response_text: Generated answer from LLM

    Returns:
        Updated sources list with 'cited' field set correctly

    Example:
        >>> sources = [{'citation_num': 1, 'cited': False}, ...]
        >>> response = "According to [1], HNSW is fast."
        >>> updated = mark_cited_sources(sources, response)
        >>> print(updated[0]['cited'])
        True
    """
    # Extract citations from response
    cited_nums = extract_citations_from_response(response_text)

    # Update sources
    for source in sources:
        citation_num = source['citation_num']
        source['cited'] = citation_num in cited_nums

    cited_count = sum(1 for s in sources if s['cited'])
    logger.info(f"Marked {cited_count}/{len(sources)} sources as cited")

    return sources


def format_sources(
    sources: List[Dict[str, Any]],
    only_cited: bool = False,
    max_text_length: int = 150
) -> str:
    """
    Format sources for display to user.

    Creates a readable list of sources with citation numbers,
    chunk IDs, scores, and text previews.

    Args:
        sources: List of source dictionaries
        only_cited: If True, only show sources that were cited
        max_text_length: Maximum text preview length

    Returns:
        Formatted string for display

    Example:
        >>> sources = [
        ...     {
        ...         'citation_num': 1,
        ...         'chunk_id': 'doc1_chunk_0',
        ...         'score': 0.92,
        ...         'text': 'HNSW is a graph algorithm...',
        ...         'cited': True
        ...     }
        ... ]
        >>> print(format_sources(sources))
        [1] doc1_chunk_0 (score: 0.92)
        "HNSW is a graph algorithm..."
    """
    if not sources:
        return "No sources available"

    # Filter if needed
    display_sources = sources
    if only_cited:
        display_sources = [s for s in sources if s.get('cited', False)]

        if not display_sources:
            return "No sources were cited in the response"

    lines = []

    for source in display_sources:
        citation_num = source['citation_num']
        chunk_id = source['chunk_id']
        score = source['score']
        text = source['text']
        cited = source.get('cited', False)

        # Truncate text preview
        if len(text) > max_text_length:
            text_preview = text[:max_text_length] + "..."
        else:
            text_preview = text

        # Format line
        cited_marker = "✓" if cited else " "
        line = (
            f"{cited_marker} [{citation_num}] {chunk_id} "
            f"(score: {score:.2f})\n"
            f'    "{text_preview}"'
        )
        lines.append(line)

    return "\n\n".join(lines)


def get_source_statistics(sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate statistics about sources and citations.

    Args:
        sources: List of source dictionaries

    Returns:
        Dictionary with statistics:
        - total_sources: Total number of sources
        - cited_sources: Number of sources that were cited
        - citation_rate: Percentage of sources cited
        - avg_score: Average retrieval score
        - min_score: Minimum score
        - max_score: Maximum score

    Example:
        >>> stats = get_source_statistics(sources)
        >>> print(f"Citation rate: {stats['citation_rate']:.1f}%")
    """
    if not sources:
        return {
            'total_sources': 0,
            'cited_sources': 0,
            'citation_rate': 0.0,
            'avg_score': 0.0,
            'min_score': 0.0,
            'max_score': 0.0,
        }

    cited_count = sum(1 for s in sources if s.get('cited', False))
    scores = [s['score'] for s in sources]

    return {
        'total_sources': len(sources),
        'cited_sources': cited_count,
        'citation_rate': (cited_count / len(sources)) * 100,
        'avg_score': sum(scores) / len(scores),
        'min_score': min(scores),
        'max_score': max(scores),
    }


def validate_citations(
    response_text: str,
    num_sources: int
) -> List[str]:
    """
    Validate citations in response.

    Checks for:
    - Citations referencing non-existent sources ([99] when only 5 sources)
    - Proper citation format

    Args:
        response_text: Generated answer
        num_sources: Number of sources that were provided

    Returns:
        List of validation warnings (empty if all valid)

    Example:
        >>> warnings = validate_citations("See [1] and [99]", num_sources=3)
        >>> print(warnings)
        ['Citation [99] references non-existent source (only 3 sources provided)']
    """
    warnings = []

    # Extract citations
    cited_nums = extract_citations_from_response(response_text)

    # Check for out-of-range citations
    for num in cited_nums:
        if num < 1 or num > num_sources:
            warnings.append(
                f"Citation [{num}] references non-existent source "
                f"(only {num_sources} sources provided)"
            )

    if warnings:
        logger.warning(f"Citation validation warnings: {warnings}")

    return warnings
