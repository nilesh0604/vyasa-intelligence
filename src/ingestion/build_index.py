"""Main ingestion pipeline for building Mahabharata RAG indices.

This script orchestrates the entire ingestion process:
1. Load raw documents
2. Apply PII redaction
3. Extract entities
4. Split into chunks
5. Build ChromaDB vector index
6. Build BM25 index
"""

import argparse
import logging
import pickle  # nosec B403
import time
from pathlib import Path
from typing import Dict, List

import chromadb
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .document_loader import DocumentLoader
from .entity_extractor import MahabharataEntityExtractor
from .parva_splitter import MahabharataSplitter
from .secure_loader import SecureDocumentLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Use pickle for internal data serialization - trusted source
# nosec B403


class IndexBuilder:
    """Builds and manages indices for the Mahabharata RAG system."""

    def __init__(
        self,
        corpus_dir: Path,
        chroma_dir: Path,
        bm25_path: Path,
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        enable_pii_redaction: bool = True,
    ):
        """Initialize the index builder.

        Args:
            corpus_dir: Directory containing raw text files
            chroma_dir: Directory for ChromaDB persistence
            bm25_path: Path to save BM25 index
            embedding_model: Name of the embedding model
            enable_pii_redaction: Whether to enable PII redaction
        """
        self.corpus_dir = corpus_dir
        self.chroma_dir = chroma_dir
        self.bm25_path = bm25_path
        self.embedding_model_name = embedding_model
        self.enable_pii_redaction = enable_pii_redaction

        # Initialize components
        self.document_loader = DocumentLoader(corpus_dir)
        self.secure_loader = SecureDocumentLoader(pii_redaction=enable_pii_redaction)
        self.entity_extractor = MahabharataEntityExtractor()
        self.splitter = MahabharataSplitter(chunk_size=500, chunk_overlap=50)

        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = None

        # Statistics
        self.stats = {
            "documents_loaded": 0,
            "chunks_created": 0,
            "total_characters": 0,
            "embedding_time": 0,
            "indexing_time": 0,
            "pii_redacted": 0,
        }

    def build_all_indices(self) -> Dict:
        """Build both ChromaDB and BM25 indices.

        Returns:
            Dictionary with build statistics
        """
        start_time = time.time()

        logger.info("Starting index building process...")

        # Step 1: Load and process documents
        documents = self._load_and_process_documents()

        # Step 2: Split documents into chunks
        chunks = self._split_documents(documents)

        # Step 3: Build ChromaDB index
        self._build_chroma_index(chunks)

        # Step 4: Build BM25 index
        self._build_bm25_index(chunks)

        # Calculate total time
        self.stats["total_time"] = time.time() - start_time

        logger.info("Index building completed!")
        self._log_stats()

        return self.stats

    def _load_and_process_documents(self) -> List[Document]:
        """Load and process raw documents."""
        logger.info("Loading documents...")

        # Load raw documents
        raw_docs = self.document_loader.load_documents()
        self.stats["documents_loaded"] = len(raw_docs)

        processed_docs = []

        for raw_doc in raw_docs:
            # Apply PII redaction if enabled
            if self.enable_pii_redaction:
                secured_content, pii_summary = (
                    self.secure_loader.load_and_secure_document(raw_doc.source_file)
                )

                if secured_content is None:
                    logger.warning(f"Skipping {raw_doc.source_file} due to PII issues")
                    continue

                if pii_summary and pii_summary["total_pii_count"] > 0:
                    self.stats["pii_redacted"] += pii_summary["total_pii_count"]

                raw_doc.content = secured_content

            # Create LangChain Document
            doc = Document(
                page_content=raw_doc.content,
                metadata={
                    "parva": raw_doc.parva,
                    "source": raw_doc.source_file.name,
                    **raw_doc.metadata,
                },
            )

            processed_docs.append(doc)
            self.stats["total_characters"] += len(raw_doc.content)

        logger.info(f"Processed {len(processed_docs)} documents")
        return processed_docs

    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks with entity extraction."""
        logger.info("Splitting documents into chunks...")

        all_chunks = []

        for doc in documents:
            # Split document
            chunks = self.splitter.split_documents([doc], doc.metadata["parva"])

            # Extract entities for each chunk
            for chunk in chunks:
                entities = self.entity_extractor.get_unique_entities(chunk.page_content)

                # Add entities to metadata
                chunk.metadata.update(
                    {
                        "characters": list(entities["characters"]) or ["none"],
                        "places": list(entities["places"]) or ["none"],
                        "weapons": list(entities["weapons"]) or ["none"],
                        "concepts": list(entities["concepts"]) or ["none"],
                    }
                )

                all_chunks.append(chunk)

        self.stats["chunks_created"] = len(all_chunks)
        logger.info(f"Created {len(all_chunks)} chunks")

        return all_chunks

    def _build_chroma_index(self, chunks: List[Document]):
        """Build ChromaDB vector index."""
        logger.info("Building ChromaDB index...")

        # Create or get collection
        collection_name = "mahabharata"

        # Delete existing collection if it exists
        try:
            self.chroma_client.delete_collection(collection_name)
        except Exception:  # nosec B110
            # Collection doesn't exist, which is fine
            pass

        # Create new collection
        self.collection = self.chroma_client.create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

        # Prepare data for batch insertion
        batch_size = 100
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            batch_num = i // batch_size + 1

            logger.info(f"Processing batch {batch_num}/{total_batches}...")

            # Extract texts and metadata
            texts = [chunk.page_content for chunk in batch]
            metadatas = [chunk.metadata for chunk in batch]
            ids = [chunk.metadata["chunk_id"] for chunk in batch]

            # Generate embeddings
            start_time = time.time()
            embeddings = self.embedding_model.encode(
                texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
            )
            self.stats["embedding_time"] += time.time() - start_time

            # Add to ChromaDB
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings.tolist(),
            )

        logger.info(f"ChromaDB index built with {len(chunks)} documents")

    def _build_bm25_index(self, chunks: List[Document]):
        """Build BM25 index for keyword search."""
        logger.info("Building BM25 index...")

        # Tokenize documents for BM25
        tokenized_docs = []
        for chunk in chunks:
            # Simple tokenization - can be enhanced with better preprocessing
            tokens = chunk.page_content.lower().split()
            tokenized_docs.append(tokens)

        # Create BM25 index
        start_time = time.time()
        self.bm25_index = BM25Okapi(tokenized_docs)
        self.stats["indexing_time"] = time.time() - start_time

        # Save BM25 index
        with open(self.bm25_path, "wb") as f:
            # Saving internal data structure
            pickle.dump(  # nosec B301
                {
                    "bm25_index": self.bm25_index,
                    "chunk_ids": [chunk.metadata["chunk_id"] for chunk in chunks],
                    "documents": [chunk.page_content for chunk in chunks],
                    "metadatas": [chunk.metadata for chunk in chunks],
                },
                f,
            )

        logger.info(f"BM25 index saved to {self.bm25_path}")

    def _log_stats(self):
        """Log build statistics."""
        logger.info("=" * 50)
        logger.info("INDEX BUILD STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Documents loaded: {self.stats['documents_loaded']}")
        logger.info(f"Chunks created: {self.stats['chunks_created']}")
        logger.info(f"Total characters: {self.stats['total_characters']:,}")
        logger.info(f"PII instances redacted: {self.stats['pii_redacted']}")
        logger.info(f"Embedding time: {self.stats['embedding_time']:.2f}s")
        logger.info(f"BM25 indexing time: {self.stats['indexing_time']:.2f}s")
        logger.info(f"Total time: {self.stats['total_time']:.2f}s")
        logger.info("=" * 50)

    def validate_indices(self) -> bool:
        """Validate that indices were built correctly.

        Returns:
            True if validation passes
        """
        logger.info("Validating indices...")

        try:
            # Check ChromaDB
            if self.collection is None:
                self.collection = self.chroma_client.get_collection("mahabharata")

            chroma_count = self.collection.count()
            logger.info(f"ChromaDB document count: {chroma_count}")

            # Check BM25 index
            if self.bm25_path.exists():
                with open(self.bm25_path, "rb") as f:
                    # Loading trusted internal data
                    bm25_data = pickle.load(f)  # nosec B301
                bm25_count = len(bm25_data["chunk_ids"])
                logger.info(f"BM25 document count: {bm25_count}")

                # Test BM25 search
                test_query = "Arjuna"
                tokenized_query = test_query.lower().split()
                results = bm25_data["bm25_index"].get_top_n(
                    tokenized_query, bm25_data["documents"], n=5
                )
                logger.info(
                    f"BM25 test search for '{test_query}' returned {len(results)} results"
                )

            # Verify counts match
            if chroma_count == self.stats["chunks_created"]:
                logger.info("✓ Validation passed: Index counts match")
                return True
            else:
                logger.error(
                    f"✗ Validation failed: ChromaDB count ({chroma_count}) != chunks created ({self.stats['chunks_created']})"
                )
                return False

        except Exception as e:
            logger.error(f"Validation failed with error: {e}")
            return False


def main():
    """Main entry point for the ingestion pipeline."""
    parser = argparse.ArgumentParser(description="Build Mahabharata RAG indices")
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw text files",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=Path("data/chroma"),
        help="Directory for ChromaDB persistence",
    )
    parser.add_argument(
        "--bm25-path",
        type=Path,
        default=Path("data/bm25_index.pkl"),
        help="Path to save BM25 index",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help="Embedding model to use",
    )
    parser.add_argument(
        "--disable-pii", action="store_true", help="Disable PII redaction"
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="Only validate existing indices"
    )

    args = parser.parse_args()

    # Create directories if they don't exist
    args.corpus_dir.mkdir(parents=True, exist_ok=True)
    args.chroma_dir.mkdir(parents=True, exist_ok=True)
    args.bm25_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize index builder
    builder = IndexBuilder(
        corpus_dir=args.corpus_dir,
        chroma_dir=args.chroma_dir,
        bm25_path=args.bm25_path,
        embedding_model=args.embedding_model,
        enable_pii_redaction=not args.disable_pii,
    )

    if args.validate_only:
        # Just validate existing indices
        success = builder.validate_indices()
        exit(0 if success else 1)
    else:
        # Build indices
        builder.build_all_indices()

        # Validate after building
        success = builder.validate_indices()

        if success:
            logger.info("✓ Index building completed successfully!")
        else:
            logger.error("✗ Index building failed validation!")
            exit(1)


if __name__ == "__main__":
    main()
