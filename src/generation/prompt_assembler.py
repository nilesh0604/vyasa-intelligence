"""Prompt assembler for Mahabharata RAG generation.

This module handles the construction of prompts with proper context formatting
and citation requirements for the Mahabharata knowledge base.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class PromptAssembler:
    """Assembles prompts with context and citation requirements."""

    def __init__(
        self,
        max_context_length: int = 4000,
        citation_format: str = "[Parva, Section]",
    ):
        """Initialize the prompt assembler.

        Args:
            max_context_length: Maximum characters of context to include
            citation_format: Format for citations (e.g., "[Parva, Section]")
        """
        self.max_context_length = max_context_length
        self.citation_format = citation_format

    def assemble_prompt(
        self,
        question: str,
        context_docs: List[Dict],
        user_role: str = "public",
        conversation_history: Optional[List[Dict]] = None,
    ) -> str:
        """Assemble a complete prompt with context and instructions.

        Args:
            question: User's question
            context_docs: List of retrieved documents with metadata
            user_role: User's role (public, scholar, admin)
            conversation_history: Optional list of previous Q&A pairs

        Returns:
            Complete prompt string
        """
        # Build system prompt based on user role
        system_prompt = self._build_system_prompt(user_role)

        # Format context with citations
        formatted_context = self._format_context(context_docs)

        # Add conversation history if provided
        history_section = self._format_conversation_history(conversation_history)

        # Assemble final prompt
        prompt_parts = [
            system_prompt,
            history_section,
            f"Context from the Mahabharata:\n{formatted_context}\n",
            f"Question: {question}\n",
            "Answer:",
        ]

        return "\n".join(filter(None, prompt_parts))

    def _build_system_prompt(self, user_role: str) -> str:
        """Build system prompt based on user role.

        Args:
            user_role: User's role (public, scholar, admin)

        Returns:
            System prompt string
        """
        base_instructions = """You are Vyasa, an AI assistant specialized in the Mahabharata.
Your role is to answer questions based solely on the provided context from the Mahabharata text.

CRITICAL REQUIREMENTS:
1. Answer ONLY using information from the provided context
2. If the context doesn't contain the answer, say "I cannot find this information in the Mahabharata text provided"
3. ALWAYS include citations for your answers using the format [Parva, Section]
4. Do not invent or extrapolate information beyond the text
5. Maintain accuracy and fidelity to the source text"""

        role_specific = {
            "public": "\n\nProvide clear, accessible answers suitable for general readers.",
            "scholar": "\n\nProvide detailed answers with specific references, suitable for academic research.",
            "admin": "\n\nProvide comprehensive answers with full context and analysis.",
        }

        return base_instructions + role_specific.get(user_role, role_specific["public"])

    def _format_context(self, context_docs: List[Dict]) -> str:
        """Format context documents with citations.

        Args:
            context_docs: List of documents with content and metadata

        Returns:
            Formatted context string
        """
        if not context_docs:
            return "No relevant context found."

        formatted_sections = []
        current_length = 0

        for i, doc in enumerate(context_docs):
            # Get citation information
            metadata = doc.get("metadata", {})
            parva = metadata.get("parva", "Unknown")
            section = metadata.get("section", str(i + 1))

            # Format citation
            citation = f"[{parva}, {section}]"

            # Get content
            content = doc.get("content", "").strip()

            # Check length limit
            if current_length + len(content) > self.max_context_length:
                # Truncate last document to fit limit
                remaining = self.max_context_length - current_length - 100
                if remaining > 100:
                    content = content[:remaining] + "..."
                else:
                    break

            # Add formatted section
            formatted_section = f"{citation}\n{content}\n"
            formatted_sections.append(formatted_section)
            current_length += len(formatted_section)

        return "\n".join(formatted_sections)

    def _format_conversation_history(
        self, conversation_history: Optional[List[Dict]]
    ) -> str:
        """Format conversation history.

        Args:
            conversation_history: List of Q&A pairs

        Returns:
            Formatted history string or empty string
        """
        if not conversation_history:
            return ""

        history_lines = ["Previous conversation:"]
        for i, turn in enumerate(conversation_history[-3:]):  # Last 3 turns
            history_lines.append(f"Q{i+1}: {turn.get('question', '')}")
            history_lines.append(f"A{i+1}: {turn.get('answer', '')}")

        return "\n".join(history_lines) + "\n\n"

    def extract_citations_from_answer(self, answer: str) -> List[str]:
        """Extract citations from an answer.

        Args:
            answer: Generated answer text

        Returns:
            List of citations found in the answer
        """
        import re

        # Pattern to match citations like [Bhishma Parva, 25]
        pattern = r"\[([^\]]+)\]"
        matches = re.findall(pattern, answer)

        # Filter and clean citations
        citations = []
        for match in matches:
            if "," in match:  # Likely a valid citation
                citations.append(f"[{match}]")

        return citations

    def validate_answer_citations(
        self, answer: str, provided_context: List[Dict]
    ) -> Dict[str, any]:
        """Validate that answer citations match provided context.

        Args:
            answer: Generated answer
            provided_context: Context documents used

        Returns:
            Validation result with found and missing citations
        """
        answer_citations = set(self.extract_citations_from_answer(answer))
        context_citations = set()

        # Build set of valid citations from context
        for doc in provided_context:
            metadata = doc.get("metadata", {})
            parva = metadata.get("parva", "Unknown")
            section = metadata.get("section", "Unknown")
            context_citations.add(f"[{parva}, {section}]")

        # Find missing citations
        missing_citations = answer_citations - context_citations
        valid_citations = answer_citations & context_citations

        return {
            "valid_citations": list(valid_citations),
            "missing_citations": list(missing_citations),
            "all_valid": len(missing_citations) == 0 and len(valid_citations) > 0,
        }
