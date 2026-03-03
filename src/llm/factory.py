"""LLM factory for provider abstraction."""

import os
from typing import Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from langchain_groq import ChatGroq
from langchain_ollama import OllamaLLM


def get_llm() -> Union[BaseLLM, BaseChatModel]:
    """Get LLM instance based on environment configuration.

    Returns:
        LLM instance (Ollama or Groq based on LLM_PROVIDER env var)
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "ollama":
        return OllamaLLM(
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.1,
        )
    elif provider == "groq":
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1,
            max_tokens=1024,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def get_local_llm(model: str = "llama3.2", temperature: float = 0.1) -> OllamaLLM:
    """Get local Ollama LLM instance.

    Args:
        model: Model name to use
        temperature: Sampling temperature

    Returns:
        Ollama LLM instance
    """
    return OllamaLLM(
        model=model,
        base_url="http://localhost:11434",
        temperature=temperature,
    )
