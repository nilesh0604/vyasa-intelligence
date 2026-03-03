"""LLM factory for provider abstraction.

This module provides a factory pattern to switch between different LLM providers
(Ollama for local development, Groq for production) while maintaining the same interface.
"""

import logging
import os
from typing import Optional

from langchain_core.language_models.llms import BaseLLM
from langchain_groq import ChatGroq
from langchain_ollama import OllamaLLM

logger = logging.getLogger(__name__)


def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: Optional[int] = None,
    **kwargs,
) -> BaseLLM:
    """Get LLM instance based on provider configuration.

    Args:
        provider: LLM provider ('ollama' or 'groq'). If None, reads from env.
        model: Model name. If None, uses provider-specific default.
        temperature: Sampling temperature (0.0 to 1.0)
        max_tokens: Maximum tokens to generate
        **kwargs: Additional provider-specific parameters

    Returns:
        Configured LLM instance

    Raises:
        ValueError: If provider is unknown or not configured
    """
    # Determine provider
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    logger.info(f"Initializing LLM with provider: {provider}")

    if provider == "ollama":
        return _get_ollama_llm(model, temperature, **kwargs)
    elif provider == "groq":
        return _get_groq_llm(model, temperature, max_tokens, **kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def _get_ollama_llm(model: Optional[str], temperature: float, **kwargs) -> OllamaLLM:
    """Get Ollama LLM instance.

    Args:
        model: Model name (e.g., 'llama3.2', 'llama3.1:8b')
        temperature: Sampling temperature
        **kwargs: Additional Ollama-specific parameters

    Returns:
        Configured Ollama LLM instance
    """
    # Default model for Ollama
    if model is None:
        model = os.getenv("OLLAMA_MODEL", "llama3.2")

    # Get base URL
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    logger.info(f"Using Ollama model: {model} at {base_url}")

    return OllamaLLM(
        model=model,
        base_url=base_url,
        temperature=temperature,
        **kwargs,
    )


def _get_groq_llm(
    model: Optional[str],
    temperature: float,
    max_tokens: Optional[int],
    **kwargs,
) -> ChatGroq:
    """Get Groq LLM instance.

    Args:
        model: Model name (e.g., 'llama-3.3-70b-versatile')
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        **kwargs: Additional Groq-specific parameters

    Returns:
        Configured Groq LLM instance

    Raises:
        ValueError: If GROQ_API_KEY is not set
    """
    # Check for API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable is required for Groq provider"
        )

    # Default model for Groq
    if model is None:
        model = "llama-3.3-70b-versatile"

    # Default max tokens for Groq
    if max_tokens is None:
        max_tokens = 1024

    logger.info(f"Using Groq model: {model}")

    return ChatGroq(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )


def get_available_models(provider: Optional[str] = None) -> list[str]:
    """Get list of available models for a provider.

    Args:
        provider: LLM provider. If None, reads from env.

    Returns:
        List of available model names
    """
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "ollama":
        return [
            "llama3.2",  # 2B, fastest
            "llama3.1:8b",  # 8B, balanced
            "mistral:7b",  # Alternative
            "qwen2.5:7b",  # Another option
        ]
    elif provider == "groq":
        return [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ]
    else:
        return []
