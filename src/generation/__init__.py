"""
LLM Generation Module

Author: Nishit Attrey

This module provides local LLM generation capabilities for the RAG system.
Uses Ollama as the backend to run models like Mistral-7B, Llama2-7B locally.

Key Features:
- Local LLM integration (no external API calls)
- Streaming and batch generation
- Citation injection in answers
- Prompt templates (Q&A, summarization, chat)
- Conversation history management
- Confidence scoring

Main Components:
    From client:
        - OllamaClient: HTTP client for Ollama API
        - ModelInfo: Model metadata

    From model:
        - get_ollama_client: Get cached client instance
        - ensure_model_available: Verify model is downloaded
        - list_available_models: List models on server
        - check_ollama_installed: Check if Ollama is installed
        - get_recommended_model: Get recommended model for use case

    From citations:
        - prepare_context_with_citations: Add citation markers to context
        - extract_citations_from_response: Extract citation numbers from text
        - mark_cited_sources: Mark which sources were actually cited
        - format_sources: Format sources for display
        - get_source_statistics: Get citation statistics
        - validate_citations: Validate citation integrity

    From prompts:
        - PromptTemplate: Template class for Jinja2 rendering
        - load_prompt_template: Load built-in template by name
        - load_custom_template: Load custom template from file
        - validate_template: Validate template has required variables
        - DEFAULT_TEMPLATES: Built-in template dictionary

    From confidence:
        - calculate_confidence: Compute confidence score based on retrieval
        - get_confidence_level: Get human-readable level (Low/Medium/High)
        - explain_confidence: Generate explanation of confidence score

    From conversation:
        - ConversationHistory: Multi-turn conversation manager with pruning

    From cache:
        - AnswerCache: LRU cache for generated answers
        - get_global_cache: Get global cache instance
        - clear_global_cache: Clear global cache

Example:
    >>> from src.generation import get_ollama_client, ensure_model_available
    >>> from src.generation import load_prompt_template
    >>>
    >>> # Get client
    >>> client = get_ollama_client()
    >>> ensure_model_available(client, "llama2:7b")
    >>>
    >>> # Generate answer
    >>> response = client.generate("What is vector search?", model="llama2:7b")
    >>> print(response)
    >>>
    >>> # Streaming generation
    >>> for token in client.generate("Explain HNSW", stream=True):
    ...     print(token, end='', flush=True)
    >>>
    >>> # Use prompt templates
    >>> template = load_prompt_template("qa")
    >>> prompt = template.render(query="What is HNSW?", context="[1] HNSW is...")
    >>> answer = client.generate(prompt, model="llama2:7b")
"""

# Import from client
from .client import (
    OllamaClient,
    ModelInfo,
)

# Import from model
from .model import (
    get_ollama_client,
    ensure_model_available,
    list_available_models,
    check_ollama_installed,
    get_recommended_model,
    clear_client_cache,
    validate_generation_config,
    DEFAULT_MODEL,
    RECOMMENDED_MODELS,
)

# Import from citations
from .citations import (
    prepare_context_with_citations,
    extract_citations_from_response,
    mark_cited_sources,
    format_sources,
    get_source_statistics,
    validate_citations,
)

# Import from prompts
from .prompts import (
    DEFAULT_TEMPLATES,
    PromptTemplate,
    load_prompt_template,
    load_custom_template,
    validate_template,
)

# Import from confidence
from .confidence import (
    calculate_confidence,
    get_confidence_level,
    explain_confidence,
)

# Import from conversation
from .conversation import ConversationHistory

# Import from cache
from .cache import (
    AnswerCache,
    get_global_cache,
    clear_global_cache,
)

__all__ = [
    # Client classes
    "OllamaClient",
    "ModelInfo",
    # Model management functions
    "get_ollama_client",
    "ensure_model_available",
    "list_available_models",
    "check_ollama_installed",
    "get_recommended_model",
    "clear_client_cache",
    "validate_generation_config",
    # Constants
    "DEFAULT_MODEL",
    "RECOMMENDED_MODELS",
    # Citations
    "prepare_context_with_citations",
    "extract_citations_from_response",
    "mark_cited_sources",
    "format_sources",
    "get_source_statistics",
    "validate_citations",
    # Prompts
    "DEFAULT_TEMPLATES",
    "PromptTemplate",
    "load_prompt_template",
    "load_custom_template",
    "validate_template",
    # Confidence
    "calculate_confidence",
    "get_confidence_level",
    "explain_confidence",
    # Conversation
    "ConversationHistory",
    # Cache
    "AnswerCache",
    "get_global_cache",
    "clear_global_cache",
]
