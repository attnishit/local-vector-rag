"""
LLM Model Management

Author: Nishit Attrey

This module manages Ollama client instances and model validation.
Follows the same pattern as src/embeddings/model.py for consistency.

Key Features:
- Module-level client caching
- Ollama installation/running checks
- Model availability validation
- Helpful error messages for setup issues

Example:
    >>> from src.generation import get_ollama_client, ensure_model_available
    >>> client = get_ollama_client()
    >>> ensure_model_available(client, "llama2:7b")
    >>> response = client.generate("Hello", model="llama2:7b")
"""

import shutil
import logging
from typing import Optional
from .client import OllamaClient

logger = logging.getLogger(__name__)

# Module-level cache (like embedding model cache)
_client_cache: Optional[OllamaClient] = None

# Recommended models for different use cases
RECOMMENDED_MODELS = {
    "small": "phi:2.7b",          # Fast, lower quality
    "medium": "llama2:7b",        # Balanced (default)
    "large": "llama2:13b",         # Best quality, slower
    "chat": "llama2:7b-instruct", # Fine-tuned for chat
}

DEFAULT_MODEL = "llama2:7b"


def check_ollama_installed() -> bool:
    """
    Check if Ollama is installed on the system.

    Returns:
        True if ollama command is found in PATH, False otherwise

    Example:
        >>> if not check_ollama_installed():
        ...     print("Please install Ollama")
    """
    is_installed = shutil.which("ollama") is not None
    logger.debug(f"Ollama installed: {is_installed}")
    return is_installed


def get_ollama_client(
    base_url: str = "http://localhost:11434",
    timeout: int = 120,
    cache: bool = True
) -> OllamaClient:
    """
    Get Ollama client instance with caching.

    This function follows the same pattern as load_embedding_model():
    - Module-level caching for efficiency
    - Health checks before returning
    - Helpful error messages

    Args:
        base_url: Ollama API endpoint (default: http://localhost:11434)
        timeout: Request timeout in seconds (default: 120)
        cache: Whether to cache the client instance (default: True)

    Returns:
        OllamaClient instance

    Raises:
        RuntimeError: If Ollama is not installed or not running

    Example:
        >>> client = get_ollama_client()
        >>> models = client.list_models()
        >>> print(f"Available: {models}")
    """
    global _client_cache

    # Return cached client if available
    if cache and _client_cache is not None:
        logger.debug("Using cached Ollama client")
        return _client_cache

    # Check if Ollama is installed
    if not check_ollama_installed():
        raise RuntimeError(
            "Ollama not found in PATH.\n\n"
            "Please install Ollama:\n"
            "  macOS:  brew install ollama\n"
            "  Linux:  curl https://ollama.ai/install.sh | sh\n"
            "  Or download from: https://ollama.ai\n\n"
            "After installation, start the server:\n"
            "  ollama serve\n"
            "  Or launch the Ollama app (macOS)"
        )

    # Create client
    logger.info(f"Initializing Ollama client: {base_url}")
    client = OllamaClient(base_url=base_url, timeout=timeout)

    # Check if server is running
    if not client.is_running():
        raise RuntimeError(
            "Ollama server is not running.\n\n"
            "Please start the Ollama server:\n"
            "  Option 1: Run 'ollama serve' in a terminal\n"
            "  Option 2: Launch the Ollama app (macOS)\n\n"
            "Then try again."
        )

    logger.info("Ollama client initialized successfully")

    # Cache if enabled
    if cache:
        _client_cache = client

    return client


def ensure_model_available(
    client: OllamaClient,
    model_name: str,
    auto_download: bool = False
) -> None:
    """
    Ensure that a model is available on the Ollama server.

    Args:
        client: OllamaClient instance
        model_name: Name of the model (e.g., "llama2:7b")
        auto_download: If True, print download command (default: False)

    Raises:
        RuntimeError: If model is not available

    Example:
        >>> client = get_ollama_client()
        >>> ensure_model_available(client, "llama2:7b")
    """
    logger.debug(f"Checking if model '{model_name}' is available")

    available_models = client.list_models()

    if model_name not in available_models:
        error_msg = (
            f"Model '{model_name}' is not available.\n\n"
            f"Available models: {available_models}\n\n"
            f"Download the model with:\n"
            f"  ollama pull {model_name}\n\n"
        )

        if not available_models:
            error_msg += (
                "No models found. You need to download at least one model.\n"
                "Recommended models:\n"
                f"  ollama pull {RECOMMENDED_MODELS['medium']}  (balanced)\n"
                f"  ollama pull {RECOMMENDED_MODELS['small']}  (fast)\n"
                f"  ollama pull {RECOMMENDED_MODELS['large']}  (best quality)\n"
            )

        logger.error(f"Model not available: {model_name}")
        raise RuntimeError(error_msg)

    logger.info(f"Model '{model_name}' is available")


def list_available_models(client: Optional[OllamaClient] = None) -> list[str]:
    """
    List all models available on Ollama server.

    Args:
        client: Optional OllamaClient instance (creates new if None)

    Returns:
        List of model names

    Example:
        >>> models = list_available_models()
        >>> for model in models:
        ...     print(f"  - {model}")
    """
    if client is None:
        client = get_ollama_client()

    models = client.list_models()
    logger.info(f"Found {len(models)} available models")
    return models


def get_recommended_model(use_case: str = "medium") -> str:
    """
    Get recommended model for a use case.

    Args:
        use_case: One of "small", "medium", "large", "chat"
                 (default: "medium")

    Returns:
        Model name

    Example:
        >>> model = get_recommended_model("chat")
        >>> print(f"Recommended: {model}")
        Recommended: llama2:7b-instruct
    """
    if use_case not in RECOMMENDED_MODELS:
        logger.warning(
            f"Unknown use case '{use_case}', "
            f"defaulting to 'medium'"
        )
        use_case = "medium"

    return RECOMMENDED_MODELS[use_case]


def clear_client_cache() -> None:
    """
    Clear the cached Ollama client.

    Useful for testing or when you need to reconnect.

    Example:
        >>> clear_client_cache()
        >>> client = get_ollama_client()  # Creates fresh client
    """
    global _client_cache
    _client_cache = None
    logger.debug("Cleared Ollama client cache")


def validate_generation_config(config: dict) -> None:
    """
    Validate generation configuration.

    Args:
        config: Configuration dictionary with generation settings

    Raises:
        ValueError: If configuration is invalid

    Example:
        >>> config = {"model": "llama2:7b", "temperature": 0.7}
        >>> validate_generation_config(config)
    """
    # Check temperature
    if "temperature" in config:
        temp = config["temperature"]
        if not (0.0 <= temp <= 2.0):
            raise ValueError(
                f"Temperature must be between 0.0 and 2.0, got {temp}"
            )

    # Check max_tokens
    if "max_tokens" in config:
        max_tokens = config["max_tokens"]
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError(
                f"max_tokens must be a positive integer, got {max_tokens}"
            )

    # Check model name
    if "model" in config:
        model = config["model"]
        if not isinstance(model, str) or not model.strip():
            raise ValueError(
                f"model must be a non-empty string, got {model}"
            )

    logger.debug("Generation config validated successfully")
