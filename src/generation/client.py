"""
Ollama HTTP Client

Author: Nishit Attrey

This module provides an HTTP client for communicating with the Ollama API.
Ollama is a local LLM server that provides an OpenAI-compatible API for
running models like Mistral, Llama2, etc.

Key Features:
- HTTP API wrapper with session pooling
- Streaming and batch generation
- Model listing and health checks
- Error handling and retries

API Endpoints:
- POST /api/generate - Generate completion
- GET /api/tags - List available models
- GET /api/show - Get model details

Example:
    >>> client = OllamaClient()
    >>> if client.is_running():
    ...     response = client.generate("What is vector search?", model="llama2:7b")
    ...     print(response)
"""

import requests
import json
import logging
from typing import Optional, Dict, Any, Iterator, Union, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about an Ollama model."""
    name: str
    size: Optional[int] = None
    modified_at: Optional[str] = None
    digest: Optional[str] = None


class OllamaClient:
    """
    HTTP client for Ollama API.

    Provides methods for generating completions, listing models,
    and checking server status.

    Attributes:
        base_url: Ollama API base URL (default: http://localhost:11434)
        timeout: Request timeout in seconds
        session: Requests session for connection pooling

    Example:
        >>> client = OllamaClient()
        >>> models = client.list_models()
        >>> response = client.generate("Hello", model="llama2:7b")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: int = 120
    ):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama server URL
            timeout: Request timeout in seconds (default: 120)
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()

        logger.debug(f"Initialized OllamaClient: {self.base_url}")

    def is_running(self) -> bool:
        """
        Check if Ollama server is running.

        Returns:
            True if server is accessible, False otherwise

        Example:
            >>> client = OllamaClient()
            >>> if not client.is_running():
            ...     print("Ollama is not running")
        """
        try:
            response = self.session.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            is_up = response.status_code == 200
            logger.debug(f"Ollama health check: {'UP' if is_up else 'DOWN'}")
            return is_up
        except requests.exceptions.RequestException as e:
            logger.warning(f"Ollama not accessible: {e}")
            return False

    def list_models(self) -> List[str]:
        """
        List available models on Ollama server.

        Returns:
            List of model names (e.g., ["llama2:7b", "llama2:7b"])

        Raises:
            RuntimeError: If Ollama is not running
            requests.exceptions.RequestException: If API call fails

        Example:
            >>> client = OllamaClient()
            >>> models = client.list_models()
            >>> print(f"Available models: {models}")
        """
        try:
            response = self.session.get(
                f"{self.base_url}/api/tags",
                timeout=10
            )
            response.raise_for_status()

            data = response.json()
            models = [model['name'] for model in data.get('models', [])]

            logger.info(f"Found {len(models)} models: {models}")
            return models

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to list models: {e}")
            raise RuntimeError(
                f"Failed to connect to Ollama API at {self.base_url}. "
                f"Is Ollama running? Error: {e}"
            )

    def get_model_info(self, model_name: str) -> ModelInfo:
        """
        Get detailed information about a model.

        Args:
            model_name: Name of the model (e.g., "llama2:7b")

        Returns:
            ModelInfo object with model details

        Raises:
            ValueError: If model not found

        Example:
            >>> info = client.get_model_info("llama2:7b")
            >>> print(f"Model size: {info.size} bytes")
        """
        models = self.list_models()

        if model_name not in models:
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available models: {models}"
            )

        # Note: Ollama API doesn't have a detailed show endpoint yet
        # Return basic info from tags endpoint
        return ModelInfo(name=model_name)

    def generate(
        self,
        prompt: str,
        model: str = "llama2:7b",
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None,
        **kwargs
    ) -> Union[str, Iterator[str]]:
        """
        Generate completion from prompt.

        Args:
            prompt: Input prompt text
            model: Model name (default: "llama2:7b")
            stream: Whether to stream response (default: False)
            temperature: Sampling temperature 0-1 (default: 0.7)
            max_tokens: Maximum tokens to generate
            system: Optional system message
            **kwargs: Additional Ollama parameters

        Returns:
            Generated text (str) or token iterator (Iterator[str]) if streaming

        Raises:
            RuntimeError: If generation fails

        Example (batch):
            >>> response = client.generate("What is AI?", model="llama2:7b")
            >>> print(response)

        Example (streaming):
            >>> for token in client.generate("What is AI?", stream=True):
            ...     print(token, end='', flush=True)
        """
        # Build request payload
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
            }
        }

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        if system:
            payload["system"] = system

        # Merge any additional options
        payload.update(kwargs)

        logger.debug(
            f"Generating with model={model}, stream={stream}, "
            f"temp={temperature}, prompt_len={len(prompt)}"
        )

        # Route to streaming or batch
        if stream:
            return self._generate_stream(payload)
        else:
            return self._generate_batch(payload)

    def _generate_stream(self, payload: Dict[str, Any]) -> Iterator[str]:
        """
        Generate streaming response (internal).

        Args:
            payload: Request payload

        Yields:
            Response tokens as they arrive

        Raises:
            RuntimeError: If streaming fails
        """
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=self.timeout
            )
            response.raise_for_status()

            # Ollama streams newline-delimited JSON
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)

                        # Extract token from response
                        if 'response' in data:
                            yield data['response']

                        # Check if done
                        if data.get('done', False):
                            logger.debug("Streaming generation completed")
                            break

                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse stream line: {e}")
                        continue

        except requests.exceptions.RequestException as e:
            logger.error(f"Streaming generation failed: {e}")
            raise RuntimeError(
                f"Failed to generate streaming response: {e}"
            )

    def _generate_batch(self, payload: Dict[str, Any]) -> str:
        """
        Generate batch response (internal).

        Args:
            payload: Request payload

        Returns:
            Complete generated text

        Raises:
            RuntimeError: If generation fails
        """
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()
            text = data.get('response', '')

            logger.debug(f"Batch generation completed: {len(text)} chars")
            return text

        except requests.exceptions.Timeout:
            logger.error("Generation timeout")
            raise RuntimeError(
                f"Generation timed out after {self.timeout}s. "
                f"Try increasing timeout or using a smaller model."
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"Batch generation failed: {e}")
            raise RuntimeError(
                f"Failed to generate response: {e}"
            )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"OllamaClient(base_url='{self.base_url}', timeout={self.timeout})"
