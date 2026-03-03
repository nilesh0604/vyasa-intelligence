"""Secure document loader with PII redaction.

This module provides secure document loading with automatic PII (Personally
Identifiable Information) detection and redaction using Microsoft Presidio.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig, RecognizerResult

logger = logging.getLogger(__name__)


class PIIRedactor:
    """Redacts PII from text using Presidio."""

    def __init__(self, language: str = "en"):
        """Initialize the PII redactor.

        Args:
            language: Language code for PII detection
        """
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.language = language

        # Define which PII types to redact
        self.pii_types = [
            "PERSON",  # Person names
            "EMAIL_ADDRESS",  # Email addresses
            "PHONE_NUMBER",  # Phone numbers
            "IBAN_CODE",  # Bank account numbers
            "CREDIT_CARD",  # Credit card numbers
            "IP_ADDRESS",  # IP addresses
            "LOCATION",  # Specific locations (not general places)
            "DATE_TIME",  # Specific dates
            "NRP",  # National registration numbers
            "URL",  # URLs
            "US_SSN",  # US Social Security Numbers
            "UK_NHS",  # UK NHS numbers
        ]

        # Configure anonymization operators
        self.anonymize_operators = {
            "PERSON": OperatorConfig("replace", {"new_value": "[PERSON]"}),
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL]"}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[PHONE]"}),
            "IBAN_CODE": OperatorConfig("replace", {"new_value": "[ACCOUNT]"}),
            "CREDIT_CARD": OperatorConfig("replace", {"new_value": "[CARD]"}),
            "IP_ADDRESS": OperatorConfig("replace", {"new_value": "[IP]"}),
            "LOCATION": OperatorConfig("replace", {"new_value": "[LOCATION]"}),
            "DATE_TIME": OperatorConfig("replace", {"new_value": "[DATE]"}),
            "NRP": OperatorConfig("replace", {"new_value": "[ID]"}),
            "URL": OperatorConfig("replace", {"new_value": "[URL]"}),
            "US_SSN": OperatorConfig("replace", {"new_value": "[SSN]"}),
            "UK_NHS": OperatorConfig("replace", {"new_value": "[NHS]"}),
        }

    def analyze_pii(self, text: str) -> List[RecognizerResult]:
        """Analyze text for PII.

        Args:
            text: Text to analyze

        Returns:
            List of PII detection results
        """
        try:
            results = self.analyzer.analyze(
                text=text, entities=self.pii_types, language=self.language
            )
            return results
        except Exception as e:
            logger.error(f"Error analyzing PII: {e}")
            return []

    def redact_pii(self, text: str) -> Tuple[str, Dict]:
        """Redact PII from text.

        Args:
            text: Text to redact

        Returns:
            Tuple of (redacted_text, pii_summary)
        """
        # First analyze for PII
        pii_results = self.analyze_pii(text)

        # Create summary of found PII
        pii_summary = self._create_pii_summary(pii_results)

        # Redact the PII
        try:
            redacted_text = self.anonymizer.anonymize(
                text=text,
                analyzer_results=pii_results,
                operators=self.anonymize_operators,
            )

            logger.info(f"Redacted {len(pii_results)} PII instances")
            return redacted_text.text, pii_summary

        except Exception as e:
            logger.error(f"Error redacting PII: {e}")
            return text, pii_summary

    def _create_pii_summary(self, pii_results: List[RecognizerResult]) -> Dict:
        """Create a summary of detected PII.

        Args:
            pii_results: List of PII detection results

        Returns:
            Summary dictionary
        """
        summary = {
            "total_pii_count": len(pii_results),
            "pii_types": {},
            "pii_instances": [],
        }

        for result in pii_results:
            # Count by type
            entity_type = result.entity_type
            if entity_type not in summary["pii_types"]:
                summary["pii_types"][entity_type] = 0
            summary["pii_types"][entity_type] += 1

            # Add instance details (without the actual value for privacy)
            summary["pii_instances"].append(
                {
                    "type": entity_type,
                    "start": result.start,
                    "end": result.end,
                    "confidence": result.score,
                }
            )

        return summary

    def is_safe_for_processing(self, text: str, max_pii_threshold: int = 10) -> bool:
        """Check if text is safe for processing based on PII count.

        Args:
            text: Text to check
            max_pii_threshold: Maximum allowed PII instances

        Returns:
            True if text is safe to process
        """
        pii_results = self.analyze_pii(text)
        pii_count = len(pii_results)

        # Check for high-risk PII types
        high_risk_types = [
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "IBAN_CODE",
            "CREDIT_CARD",
            "US_SSN",
            "UK_NHS",
        ]
        high_risk_count = sum(
            1 for r in pii_results if r.entity_type in high_risk_types
        )

        if high_risk_count > 0:
            logger.warning(f"High-risk PII detected: {high_risk_count} instances")
            return False

        if pii_count > max_pii_threshold:
            logger.warning(
                f"Too many PII instances detected: {pii_count} > {max_pii_threshold}"
            )
            return False

        return True


class SecureDocumentLoader:
    """Secure document loader with PII protection."""

    def __init__(self, pii_redaction: bool = True):
        """Initialize the secure loader.

        Args:
            pii_redaction: Whether to enable PII redaction
        """
        self.pii_redaction = pii_redaction
        if pii_redaction:
            self.redactor = PIIRedactor()

        self.stats = {
            "documents_processed": 0,
            "pii_instances_redacted": 0,
            "documents_with_pii": 0,
        }

    def load_and_secure_document(
        self, file_path: Path, encoding: str = "utf-8"
    ) -> Tuple[str, Optional[Dict]]:
        """Load and secure a document.

        Args:
            file_path: Path to the document
            encoding: File encoding

        Returns:
            Tuple of (secured_content, pii_summary)
        """
        try:
            # Read the file
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()

            pii_summary = None

            # Apply PII redaction if enabled
            if self.pii_redaction:
                # Check if content is safe
                if not self.redactor.is_safe_for_processing(content):
                    logger.warning(
                        f"Document {file_path} contains excessive PII, skipping"
                    )
                    return None, None

                # Redact PII
                secured_content, pii_summary = self.redactor.redact_pii(content)

                # Update stats
                self.stats["documents_processed"] += 1
                if pii_summary["total_pii_count"] > 0:
                    self.stats["documents_with_pii"] += 1
                    self.stats["pii_instances_redacted"] += pii_summary[
                        "total_pii_count"
                    ]

                logger.info(
                    f"Processed {file_path}: {pii_summary['total_pii_count']} PII instances redacted"
                )
            else:
                secured_content = content
                self.stats["documents_processed"] += 1

            return secured_content, pii_summary

        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            return None, None

    def get_processing_stats(self) -> Dict:
        """Get statistics about document processing.

        Returns:
            Processing statistics
        """
        return self.stats.copy()

    def reset_stats(self):
        """Reset processing statistics."""
        self.stats = {
            "documents_processed": 0,
            "pii_instances_redacted": 0,
            "documents_with_pii": 0,
        }


if __name__ == "__main__":
    # Example usage
    loader = SecureDocumentLoader(pii_redaction=True)

    # Create a sample text with PII
    sample_text = """
    John Smith can be contacted at john.smith@email.com or 555-123-4567.
    His credit card number is 4532-1234-5678-9012.
    He lives at 123 Main Street, New York, NY 10001.
    His SSN is 123-45-6789.
    """

    # Redact PII
    redacted_text, pii_summary = loader.redactor.redact_pii(sample_text)

    print("Original Text:")
    print(sample_text)
    print("\nRedacted Text:")
    print(redacted_text)
    print("\nPII Summary:")
    print(f"Total PII: {pii_summary['total_pii_count']}")
    print(f"PII Types: {pii_summary['pii_types']}")
