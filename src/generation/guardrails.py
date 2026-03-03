"""Guardrails for content safety and output validation.

This module implements content safety checks to ensure appropriate responses
and prevent misuse of the Mahabharata RAG system.
"""

import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)


class ContentGuardrails:
    """Implements content safety and validation guardrails."""

    def __init__(self):
        """Initialize the guardrails with default policies."""
        # Blocked content categories
        self.blocked_categories = {
            "hate_speech": True,
            "violence": True,
            "adult_content": True,
            "personal_questions": False,  # Allow but monitor
            "political": True,
            "religious_conversion": True,
        }

        # Patterns for blocked content
        self.blocked_patterns = [
            # Hate speech patterns
            r"\b(hate|kill|destroy|eliminate).*\b(people|group|community|caste|religion)",
            r"\b(superior|inferior).*\b(race|caste|religion)",
            # Violence patterns
            r"\b(how to|instructions for).*(bomb|weapon|violence|attack)",
            # Adult content patterns
            r"\b(sex|nude|porn|adult).*\b(content|video|image)",
            # Political patterns
            r"\b(vote|election|campaign|political party)",
            # Religious conversion patterns
            r"\b(convert).*\b(to christianity|to islam|to hinduism|religion)",
        ]

        # Mahabharata-specific allowed content
        self.mahabharata_context_patterns = [
            r"\b(kuru|pandava|kaurava|hastinapura|indraprastha)",
            r"\b(dharma|adharma|karma|yoga|veda)",
            r"\b(krishna|arjuna|bhishma|drona|karna|duryodhana)",
            r"\b(kurukshetra|battle|war).*\bmahabharata",
        ]

        # Compile regex patterns
        self.compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.blocked_patterns
        ]
        self.context_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.mahabharata_context_patterns
        ]

    def check_input(self, question: str, user_role: str = "public") -> Dict[str, any]:
        """Check if user input violates any guardrails.

        Args:
            question: User's question
            user_role: User's role (public, scholar, admin)

        Returns:
            Dictionary with check results
        """
        result = {
            "allowed": True,
            "blocked_categories": [],
            "warnings": [],
            "is_mahabharata_related": self._is_mahabharata_related(question),
        }

        # Check for blocked patterns
        for i, pattern in enumerate(self.compiled_patterns):
            if pattern.search(question):
                category = (
                    list(self.blocked_categories.keys())[i]
                    if i < len(self.blocked_categories)
                    else "unknown"
                )
                if self.blocked_categories.get(category, True):
                    result["blocked_categories"].append(category)
                    result["allowed"] = False

        # Role-based checks
        if user_role == "public":
            # Additional restrictions for public users
            if self._contains_personal_info_request(question):
                result["warnings"].append("Question asks for personal information")

        elif user_role == "admin":
            # Admins have fewer restrictions
            result["allowed"] = True

        # Log violations
        if not result["allowed"]:
            logger.warning(
                f"Input blocked: {question[:100]}... Categories: {result['blocked_categories']}"
            )

        return result

    def check_output(self, answer: str, context_used: bool = True) -> Dict[str, any]:
        """Check if generated output violates any guardrails.

        Args:
            answer: Generated answer
            context_used: Whether answer was based on retrieved context

        Returns:
            Dictionary with check results
        """
        result = {
            "allowed": True,
            "violations": [],
            "warnings": [],
            "has_citations": self._has_citations(answer),
            "length_appropriate": self._check_length(answer),
        }

        # Check for blocked patterns in output
        for i, pattern in enumerate(self.compiled_patterns):
            if pattern.search(answer):
                category = (
                    list(self.blocked_categories.keys())[i]
                    if i < len(self.blocked_categories)
                    else "unknown"
                )
                if self.blocked_categories.get(category, True):
                    result["violations"].append(category)
                    result["allowed"] = False

        # Check for hallucination indicators
        if context_used and not result["has_citations"]:
            result["warnings"].append("Answer lacks proper citations")

        # Check for refusal patterns
        if self._is_refusal(answer):
            result["warnings"].append("Answer appears to be a refusal")

        # Check for length
        if not result["length_appropriate"]:
            result["warnings"].append("Answer length may be inappropriate")

        # Log violations
        if not result["allowed"]:
            logger.warning(
                f"Output blocked: {answer[:100]}... Violations: {result['violations']}"
            )

        return result

    def _is_mahabharata_related(self, text: str) -> bool:
        """Check if text is related to Mahabharata.

        Args:
            text: Text to check

        Returns:
            True if text appears related to Mahabharata
        """
        text_lower = text.lower()

        # Direct mentions
        if "mahabharata" in text_lower:
            return True

        # Check for character/place names
        for pattern in self.context_patterns:
            if pattern.search(text):
                return True

        return False

    def _contains_personal_info_request(self, text: str) -> bool:
        """Check if text requests personal information.

        Args:
            text: Text to check

        Returns:
            True if personal info is requested
        """
        personal_patterns = [
            r"\b(your name|who are you|tell me about yourself)",
            r"\b(personal|private|confidential)",
            r"\b(email|phone|address)\b",
        ]

        for pattern in personal_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def _has_citations(self, answer: str) -> bool:
        """Check if answer contains proper citations.

        Args:
            answer: Answer text

        Returns:
            True if citations are present
        """
        # Look for citation format [Parva, Section]
        citation_pattern = r"\[[^]]+\]"
        return bool(re.search(citation_pattern, answer))

    def _check_length(self, answer: str) -> bool:
        """Check if answer length is appropriate.

        Args:
            answer: Answer text

        Returns:
            True if length is appropriate
        """
        # Too short
        if len(answer) < 20:
            return False

        # Too long (potential rambling)
        if len(answer) > 2000:
            return False

        return True

    def _is_refusal(self, answer: str) -> bool:
        """Check if answer is a refusal.

        Args:
            answer: Answer text

        Returns:
            True if answer appears to be a refusal
        """
        refusal_patterns = [
            r"\b(i cannot|cannot answer|unable to|not able to)",
            r"\b(i'm sorry|sorry, i)",
            r"\b(inappropriate|not appropriate)",
            r"\b(don't have|do not have).*(information|data)",
        ]

        for pattern in refusal_patterns:
            if re.search(pattern, answer, re.IGNORECASE):
                return True

        return False

    def sanitize_input(self, text: str) -> str:
        """Sanitize input text by removing potentially harmful content.

        Args:
            text: Input text

        Returns:
            Sanitized text
        """
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Remove or replace special characters that might cause issues
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

        # Limit length
        if len(text) > 1000:
            text = text[:1000] + "..."

        return text

    def get_allowed_topics(self) -> List[str]:
        """Get list of allowed topics for Mahabharata discussions.

        Returns:
            List of allowed topics
        """
        return [
            "Characters and their roles",
            "Philosophical concepts (dharma, karma, moksha)",
            "Historical events and battles",
            "Genealogy and lineages",
            "Moral and ethical dilemmas",
            "Teachings and parables",
            "Cultural and social practices",
            "Interpretations and analysis",
        ]

    def update_policy(self, category: str, allowed: bool):
        """Update guardrail policy for a category.

        Args:
            category: Category name
            allowed: Whether to allow this category
        """
        if category in self.blocked_categories:
            self.blocked_categories[category] = not allowed
            logger.info(
                f"Updated guardrail policy: {category} = {'allowed' if allowed else 'blocked'}"
            )
        else:
            logger.warning(f"Unknown guardrail category: {category}")
