"""Mahabharata-aware text splitter.

This module implements hierarchical chunking that respects the Mahabharata's
structure of Parvas (books) and Adhyayas (chapters), ensuring better
context preservation for RAG retrieval.
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata for each chunk."""

    chunk_id: str
    parva: str
    adhyaya: Optional[str]
    chapter_number: Optional[int]
    section_start: Optional[int]
    section_end: Optional[int]
    characters_mentioned: List[str]
    places_mentioned: List[str]
    is_dialogue: bool
    chunk_type: str  # 'narrative', 'dialogue', 'description', 'philosophy'


class MahabharataSplitter:
    """Splits Mahabharata text while preserving hierarchical structure."""

    def __init__(
        self, chunk_size: int = 500, chunk_overlap: int = 50, min_chunk_size: int = 100
    ):
        """Initialize the splitter.

        Args:
            chunk_size: Target size of each chunk in tokens
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum size for a valid chunk
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        # Initialize base splitter for final chunking
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",  # Paragraph breaks
                "\n",  # Line breaks
                ". ",  # Sentence ends
                " ",  # Words
                "",  # Characters
            ],
        )

        # Compile regex patterns for structure detection
        self.adhyaya_pattern = re.compile(
            r"(?i)chapter\s+(\d+)|adhyaya\s+(\d+)|section\s+(\d+)|part\s+(\d+)",
            re.MULTILINE,
        )

        self.parva_pattern = re.compile(
            r"(?i)(adi|sabha|vana|virata|udyoga|bhishma|drona|karna|shalya|"
            r"sauptika|stree|shanti|anushasana|ashvamedhika|ashramavasika|"
            r"mausala|mahaprasthanika|svargarohanika)\s+parva",
            re.MULTILINE,
        )

        self.dialogue_pattern = re.compile(
            r'^["\']|"([^"]*)"$|^\'([^\']*)\'$', re.MULTILINE
        )

        # Common character names for entity detection
        self.character_names = {
            "arjuna",
            "krishna",
            "yudhishthira",
            "bhima",
            "nakula",
            "sahadeva",
            "duryodhana",
            "dushasana",
            "karna",
            "bhishma",
            "drona",
            "kripa",
            "ashwatthama",
            "shakuni",
            "dhritarashtra",
            "gandhari",
            "kunti",
            "draupadi",
            "subhadra",
            "gandhari",
            "karna",
            "vidura",
            "sanjaya",
            "balarama",
            "rishyasringa",
            "vyasa",
            "sage vyasa",
            "ganga",
            "shantanu",
            "satyavati",
            "bhishma",
            "chitrangada",
            "vichitravirya",
            "ambika",
            "ambalika",
            "pandu",
            "madri",
            "dhritarashtra",
        }

        # Common place names
        self.place_names = {
            "hastinapura",
            "indraprastha",
            "kurukshetra",
            "dwarka",
            "mathura",
            "varanavata",
            "panchala",
            "gandhara",
            "magadha",
            "kashi",
            "vidarbha",
            "matsya",
            "virata",
            "drupada",
            "kamboja",
            "sindhu",
            "sauvira",
            "ganga",
            "yamuna",
            "saraswati",
            "himalayas",
            "vindhya",
            "mount meru",
        }

    def split_documents(
        self, documents: List[Document], parva_name: str
    ) -> List[Document]:
        """Split documents into chunks while preserving structure.

        Args:
            documents: List of LangChain Document objects
            parva_name: Name of the Parva being processed

        Returns:
            List of chunked Document objects with enhanced metadata
        """
        all_chunks = []

        for doc in documents:
            # First, split by chapters/adhyayas
            chapters = self._split_into_chapters(doc.page_content)

            for i, (chapter_title, chapter_content) in enumerate(chapters):
                # Extract chapter number
                chapter_num = self._extract_chapter_number(chapter_title)

                # Split chapter into smaller chunks
                chunks = self.base_splitter.split_text(chapter_content)

                for j, chunk_text in enumerate(chunks):
                    # Skip very small chunks
                    if len(chunk_text) < self.min_chunk_size:
                        continue

                    # Create enhanced metadata
                    metadata = self._create_chunk_metadata(
                        chunk_text=chunk_text,
                        parva=parva_name,
                        chapter_title=chapter_title,
                        chapter_number=chapter_num,
                        chunk_index=j,
                        total_chunks=len(chunks),
                    )

                    # Create Document object
                    chunk = Document(page_content=chunk_text, metadata=metadata)

                    all_chunks.append(chunk)

        logger.info(
            f"Split into {len(all_chunks)} chunks from {len(documents)} documents"
        )
        return all_chunks

    def _split_into_chapters(self, text: str) -> List[Tuple[str, str]]:
        """Split text into chapters based on chapter markers.

        Args:
            text: Full text to split

        Returns:
            List of (chapter_title, chapter_content) tuples
        """
        chapters = []

        # Find all chapter headings
        chapter_matches = list(self.adhyaya_pattern.finditer(text))

        if not chapter_matches:
            # No clear chapter divisions, treat as single chapter
            chapters.append(("Chapter 1", text))
            return chapters

        # Extract chapters based on matches
        for i, match in enumerate(chapter_matches):
            start_pos = match.start()
            end_pos = (
                chapter_matches[i + 1].start()
                if i + 1 < len(chapter_matches)
                else len(text)
            )

            chapter_title = f"Chapter {i + 1}"
            chapter_content = text[start_pos:end_pos].strip()

            if chapter_content:
                chapters.append((chapter_title, chapter_content))

        return chapters

    def _extract_chapter_number(self, chapter_title: str) -> Optional[int]:
        """Extract chapter number from chapter title.

        Args:
            chapter_title: Title of the chapter

        Returns:
            Chapter number if found, None otherwise
        """
        match = re.search(r"\d+", chapter_title)
        return int(match.group()) if match else None

    def _create_chunk_metadata(
        self,
        chunk_text: str,
        parva: str,
        chapter_title: str,
        chapter_number: Optional[int],
        chunk_index: int,
        total_chunks: int,
    ) -> Dict:
        """Create comprehensive metadata for a chunk.

        Args:
            chunk_text: The text content of the chunk
            parva: Parva name
            chapter_title: Title of the chapter
            chapter_number: Chapter number if available
            chunk_index: Index of this chunk within the chapter
            total_chunks: Total number of chunks in the chapter

        Returns:
            Metadata dictionary
        """
        # Extract entities
        characters = self._extract_entities(chunk_text, self.character_names)
        places = self._extract_entities(chunk_text, self.place_names)

        # Detect if chunk is primarily dialogue
        is_dialogue = self._is_dialogue_chunk(chunk_text)

        # Classify chunk type
        chunk_type = self._classify_chunk_type(chunk_text, is_dialogue)

        # Create unique chunk ID
        chunk_id = f"{parva.lower().replace(' ', '_')}_"
        if chapter_number:
            chunk_id += f"ch{chapter_number}_"
        chunk_id += f"chunk{chunk_index + 1}"

        return {
            "chunk_id": chunk_id,
            "parva": parva,
            "adhyaya": chapter_title,
            "chapter_number": chapter_number,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "characters_mentioned": sorted(list(characters)) or ["none"],
            "places_mentioned": sorted(list(places)) or ["none"],
            "is_dialogue": is_dialogue,
            "chunk_type": chunk_type,
            "token_count": len(chunk_text.split()),
            "char_count": len(chunk_text),
        }

    def _extract_entities(self, text: str, entity_set: set) -> set:
        """Extract entities from text using fuzzy matching.

        Args:
            text: Text to search for entities
            entity_set: Set of known entity names

        Returns:
            Set of found entities
        """
        found = set()
        text_lower = text.lower()

        for entity in entity_set:
            # Check for exact match
            if entity in text_lower:
                found.add(entity.title())
            # Check for partial matches (e.g., "Arjuna" in "Arjuna's")
            elif f" {entity}" in text_lower or f"{entity} " in text_lower:
                found.add(entity.title())

        return found

    def _is_dialogue_chunk(self, text: str) -> bool:
        """Determine if a chunk is primarily dialogue.

        Args:
            text: Text to analyze

        Returns:
            True if chunk is mostly dialogue
        """
        # Count dialogue markers
        dialogue_markers = text.count('"') + text.count("'")
        sentences = text.count(".") + text.count("!") + text.count("?")

        # If more than 30% of sentences have dialogue markers, consider it dialogue
        if sentences > 0:
            dialogue_ratio = dialogue_markers / (2 * sentences)  # Divide by 2 for pairs
            return dialogue_ratio > 0.3

        return False

    def _classify_chunk_type(self, text: str, is_dialogue: bool) -> str:
        """Classify the type of content in a chunk.

        Args:
            text: Text to classify
            is_dialogue: Whether the chunk is primarily dialogue

        Returns:
            Chunk type string
        """
        text_lower = text.lower()

        # Philosophy keywords
        philosophy_keywords = [
            "dharma",
            "duty",
            "karma",
            "moksha",
            "yoga",
            "soul",
            "atman",
            "brahman",
            "reincarnation",
            "ethics",
            "morality",
            "wisdom",
            "knowledge",
            "truth",
            "reality",
            "spiritual",
            "divine",
        ]

        # Battle/description keywords
        battle_keywords = [
            "battle",
            "war",
            "fight",
            "weapon",
            "arrow",
            "sword",
            "chariot",
            "army",
            "soldier",
            "attack",
            "defend",
            "victory",
            "defeat",
        ]

        if is_dialogue:
            return "dialogue"
        elif any(keyword in text_lower for keyword in philosophy_keywords):
            return "philosophy"
        elif any(keyword in text_lower for keyword in battle_keywords):
            return "description"
        else:
            return "narrative"

    def get_splitting_stats(self, chunks: List[Document]) -> Dict:
        """Get statistics about the splitting results.

        Args:
            chunks: List of chunked documents

        Returns:
            Statistics dictionary
        """
        if not chunks:
            return {}

        # Calculate statistics
        total_chunks = len(chunks)
        avg_tokens = sum(c.metadata["token_count"] for c in chunks) / total_chunks
        chunk_types = {}
        parvas = set()

        for chunk in chunks:
            # Count chunk types
            chunk_type = chunk.metadata["chunk_type"]
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1

            # Collect parvas
            parvas.add(chunk.metadata["parva"])

        return {
            "total_chunks": total_chunks,
            "avg_tokens_per_chunk": avg_tokens,
            "chunk_types": chunk_types,
            "parvas_processed": list(parvas),
            "chunks_with_dialogue": sum(1 for c in chunks if c.metadata["is_dialogue"]),
            "chunks_with_characters": sum(
                1 for c in chunks if c.metadata["characters_mentioned"]
            ),
            "chunks_with_places": sum(
                1 for c in chunks if c.metadata["places_mentioned"]
            ),
        }


if __name__ == "__main__":
    # Example usage
    from langchain_core.documents import Document

    splitter = MahabharataSplitter(chunk_size=500, chunk_overlap=50)

    # Create a sample document
    sample_text = """
    CHAPTER 1
    Arjuna said: "O Krishna, what is the duty of a warrior?"
    Krishna replied: "The duty is to fight for dharma."
    
    CHAPTER 2  
    On the battlefield of Kurukshetra, the armies gathered.
    The Pandavas stood on one side, the Kauravas on the other.
    """

    doc = Document(page_content=sample_text)
    chunks = splitter.split_documents([doc], "Bhagavad Gita")

    print(f"Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}:")
        print(f"  Type: {chunk.metadata['chunk_type']}")
        print(f"  Characters: {chunk.metadata['characters_mentioned']}")
        print(f"  Preview: {chunk.page_content[:100]}...")
