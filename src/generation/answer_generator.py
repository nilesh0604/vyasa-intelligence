"""Answer generator for Mahabharata RAG system.

This module handles the generation of answers using retrieved context
and configured LLM providers.
"""

import logging
import time
from typing import Dict, List, Optional

from langchain_core.callbacks import AsyncCallbackHandler

from .llm_factory import get_llm
from .prompt_assembler import PromptAssembler

logger = logging.getLogger(__name__)


class GenerationCallbackHandler(AsyncCallbackHandler):
    """Callback handler for logging generation metrics."""

    def __init__(self):
        self.start_time = None
        self.tokens_generated = 0
        self.metrics = {}

    async def on_llm_start(self, serialized: Dict, prompts: List[str], **kwargs):
        """Called when LLM starts generation."""
        self.start_time = time.time()
        logger.info(f"Starting LLM generation with {len(prompts)} prompts")

    async def on_llm_end(self, response, **kwargs):
        """Called when LLM finishes generation."""
        if self.start_time:
            duration = time.time() - self.start_time
            self.metrics["generation_time"] = duration

            # Estimate tokens (rough approximation)
            if hasattr(response, "generations") and response.generations:
                text = response.generations[0][0].text
                self.tokens_generated = len(text.split())
                self.metrics["tokens_generated"] = self.tokens_generated
                self.metrics["tokens_per_second"] = (
                    self.tokens_generated / duration if duration > 0 else 0
                )

            logger.info(
                f"Generation completed in {duration:.2f}s, ~{self.tokens_generated} tokens"
            )


class AnswerGenerator:
    """Generates answers using retrieved context and LLM."""

    def __init__(
        self,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        enable_tracing: bool = True,
    ):
        """Initialize the answer generator.

        Args:
            llm_provider: LLM provider ('ollama' or 'groq')
            llm_model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            enable_tracing: Whether to enable LangSmith tracing
        """
        self.llm = get_llm(
            provider=llm_provider,
            model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.prompt_assembler = PromptAssembler()
        self.enable_tracing = enable_tracing
        self.callback_handler = GenerationCallbackHandler()

    def generate_answer(
        self,
        question: str,
        context_docs: List[Dict],
        user_role: str = "public",
        conversation_history: Optional[List[Dict]] = None,
    ) -> Dict[str, any]:
        """Generate an answer for a question using provided context.

        Args:
            question: User's question
            context_docs: List of retrieved documents
            user_role: User's role (public, scholar, admin)
            conversation_history: Optional conversation history

        Returns:
            Dictionary with answer, citations, and metadata
        """
        if not context_docs:
            return {
                "answer": "I cannot find relevant information in the Mahabharata text to answer your question.",
                "citations": [],
                "sources": [],
                "metadata": {
                    "context_used": False,
                    "generation_time": 0,
                    "tokens_generated": 0,
                },
            }

        # Assemble prompt
        prompt = self.prompt_assembler.assemble_prompt(
            question=question,
            context_docs=context_docs,
            user_role=user_role,
            conversation_history=conversation_history,
        )

        # Generate answer
        start_time = time.time()

        try:
            # Generate answer without callbacks to avoid the error
            answer = self.llm.invoke(prompt)
            generation_time = time.time() - start_time

            # Handle different response types
            if hasattr(answer, "content"):
                # For ChatGroq and other chat models
                answer_text = answer.content
            else:
                # For LLMs that return strings directly
                answer_text = str(answer)

            # Extract citations
            citations = self.prompt_assembler.extract_citations_from_answer(answer_text)

            # Validate citations
            validation_result = self.prompt_assembler.validate_answer_citations(
                answer_text, context_docs
            )

            # Build sources list
            sources = self._build_sources_list(context_docs, citations)

            # Prepare metadata
            metadata = {
                "context_used": True,
                "context_count": len(context_docs),
                "generation_time": generation_time,
                "tokens_generated": self.callback_handler.tokens_generated,
                "citations_valid": validation_result["all_valid"],
                "missing_citations": validation_result["missing_citations"],
            }

            # Add callback metrics if available
            if hasattr(self.callback_handler, "metrics"):
                metadata.update(self.callback_handler.metrics)

            return {
                "answer": answer_text.strip(),
                "citations": citations,
                "sources": sources,
                "metadata": metadata,
            }

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": f"An error occurred while generating the answer: {str(e)}",
                "citations": [],
                "sources": [],
                "metadata": {
                    "context_used": False,
                    "error": str(e),
                    "generation_time": time.time() - start_time,
                },
            }

    def _build_sources_list(
        self, context_docs: List[Dict], citations: List[str]
    ) -> List[str]:
        """Build a formatted list of sources.

        Args:
            context_docs: Retrieved documents
            citations: Citations found in answer

        Returns:
            List of formatted source strings
        """
        sources = []
        seen = set()

        for doc in context_docs:
            metadata = doc.get("metadata", {})
            parva = metadata.get("parva", "Unknown")
            section = metadata.get("section", "Unknown")
            citation_key = f"[{parva}, {section}]"

            # Only include sources that were actually cited
            if citation_key in citations and citation_key not in seen:
                # Add additional context if available
                title = metadata.get("title", "")
                if title:
                    sources.append(f"{citation_key} - {title}")
                else:
                    sources.append(citation_key)
                seen.add(citation_key)

        return sources

    def generate_batch(
        self,
        questions: List[str],
        context_docs_list: List[List[Dict]],
        user_role: str = "public",
    ) -> List[Dict[str, any]]:
        """Generate answers for multiple questions.

        Args:
            questions: List of questions
            context_docs_list: List of context document lists for each question
            user_role: User role for all questions

        Returns:
            List of generation results
        """
        if len(questions) != len(context_docs_list):
            raise ValueError(
                "Number of questions must match number of context document lists"
            )

        results = []
        for question, context_docs in zip(questions, context_docs_list):
            result = self.generate_answer(
                question=question,
                context_docs=context_docs,
                user_role=user_role,
            )
            results.append(result)

        return results

    def update_llm_config(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """Update LLM configuration.

        Args:
            provider: New LLM provider
            model: New model name
            temperature: New temperature
            max_tokens: New max tokens
        """
        import os

        new_provider = provider or os.getenv("LLM_PROVIDER", "ollama")
        new_temperature = temperature if temperature is not None else 0.1
        new_max_tokens = max_tokens if max_tokens is not None else None

        self.llm = get_llm(
            provider=new_provider,
            model=model,
            temperature=new_temperature,
            max_tokens=new_max_tokens,
        )

        logger.info(
            f"Updated LLM configuration: provider={new_provider}, model={model or 'default'}"
        )
