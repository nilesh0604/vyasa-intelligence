"""Document loader for Mahabharata corpus.

This module handles loading raw text files containing the Mahabharata
translation by K.M. Ganguli, with support for different file formats
and proper encoding handling.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class MahabharataDocument:
    """Represents a single document (Parva) of the Mahabharata."""

    def __init__(
        self,
        content: str,
        parva: str,
        source_file: Path,
        metadata: Optional[Dict] = None,
    ):
        self.content = content
        self.parva = parva
        self.source_file = source_file
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"MahabharataDocument(parva='{self.parva}', length={len(self.content)})"


class DocumentLoader:
    """Loads Mahabharata documents from raw text files."""

    def __init__(self, corpus_dir: Path):
        """Initialize the document loader.

        Args:
            corpus_dir: Directory containing raw text files
        """
        self.corpus_dir = Path(corpus_dir)
        if not self.corpus_dir.exists():
            raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    def load_documents(self) -> List[MahabharataDocument]:
        """Load all documents from the corpus directory.

        Returns:
            List of MahabharataDocument objects
        """
        documents = []

        # Look for .txt files in the corpus directory
        for file_path in self.corpus_dir.glob("*.txt"):
            try:
                doc = self._load_single_file(file_path)
                if doc:
                    documents.append(doc)
                    logger.info(f"Loaded {doc.parva} from {file_path.name}")
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                continue

        # Sort by Parva number for consistent ordering
        documents.sort(key=lambda x: self._get_parva_number(x.parva))

        logger.info(f"Loaded {len(documents)} documents total")
        return documents

    def _load_single_file(self, file_path: Path) -> Optional[MahabharataDocument]:
        """Load a single text file.

        Args:
            file_path: Path to the text file

        Returns:
            MahabharataDocument or None if loading failed
        """
        # Try different encodings
        encodings = ["utf-8", "latin-1", "cp1252"]

        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read()

                # Extract parva name from filename
                parva = self._extract_parva_name(file_path.stem)

                # Create metadata
                metadata = {
                    "source_file": str(file_path),
                    "encoding": encoding,
                    "file_size": file_path.stat().st_size,
                    "line_count": content.count("\n") + 1,
                }

                return MahabharataDocument(
                    content=content,
                    parva=parva,
                    source_file=file_path,
                    metadata=metadata,
                )
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error loading {file_path} with {encoding}: {e}")
                continue

        logger.error(f"Failed to decode {file_path} with any encoding")
        return None

    def _extract_parva_name(self, filename: str) -> str:
        """Extract parva name from filename.

        Args:
            filename: Name of the file without extension

        Returns:
            Formatted parva name
        """
        # Common mappings for filename to parva name
        parva_mappings = {
            "adi": "Adi Parva",
            "sabha": "Sabha Parva",
            "vana": "Vana Parva",
            "virata": "Virata Parva",
            "udyoga": "Udyoga Parva",
            "bhishma": "Bhishma Parva",
            "drona": "Drona Parva",
            "karna": "Karna Parva",
            "shalya": "Shalya Parva",
            "sauptika": "Sauptika Parva",
            "stree": "Stree Parva",
            "shanti": "Shanti Parva",
            "anushasana": "Anushasana Parva",
            "ashvamedhika": "Ashvamedhika Parva",
            "ashramavasika": "Ashramavasika Parva",
            "mausala": "Mausala Parva",
            "mahaprasthanika": "Mahaprasthanika Parva",
            "svargarohanika": "Svargarohanika Parva",
        }

        # Convert to lowercase and remove special characters
        normalized = filename.lower().replace("_", "").replace("-", "").replace(" ", "")

        # Check mappings
        for key, value in parva_mappings.items():
            if key in normalized:
                return value

        # If no mapping found, format the filename
        return filename.replace("_", " ").replace("-", " ").title()

    def _get_parva_number(self, parva: str) -> int:
        """Get the numerical order of a Parva.

        Args:
            parva: Parva name

        Returns:
            Parva number (1-18)
        """
        parva_order = {
            "Adi Parva": 1,
            "Sabha Parva": 2,
            "Vana Parva": 3,
            "Virata Parva": 4,
            "Udyoga Parva": 5,
            "Bhishma Parva": 6,
            "Drona Parva": 7,
            "Karna Parva": 8,
            "Shalya Parva": 9,
            "Sauptika Parva": 10,
            "Stree Parva": 11,
            "Shanti Parva": 12,
            "Anushasana Parva": 13,
            "Ashvamedhika Parva": 14,
            "Ashramavasika Parva": 15,
            "Mausala Parva": 16,
            "Mahaprasthanika Parva": 17,
            "Svargarohanika Parva": 18,
        }

        return parva_order.get(parva, 99)

    def get_document_stats(self) -> Dict:
        """Get statistics about the loaded documents.

        Returns:
            Dictionary with document statistics
        """
        documents = self.load_documents()

        total_chars = sum(len(doc.content) for doc in documents)
        total_words = sum(len(doc.content.split()) for doc in documents)
        total_lines = sum(doc.metadata.get("line_count", 0) for doc in documents)

        return {
            "document_count": len(documents),
            "total_characters": total_chars,
            "total_words": total_words,
            "total_lines": total_lines,
            "avg_document_size": total_chars / len(documents) if documents else 0,
            "parvas": [doc.parva for doc in documents],
        }


if __name__ == "__main__":
    # Example usage
    loader = DocumentLoader(Path("data/raw"))
    docs = loader.load_documents()
    stats = loader.get_document_stats()

    print(f"Loaded {stats['document_count']} documents")
    print(f"Total words: {stats['total_words']:,}")
    print(f"Parvas: {', '.join(stats['parvas'])}")
