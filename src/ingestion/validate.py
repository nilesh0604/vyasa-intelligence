"""Validation script for Mahabharata ingestion pipeline.

This script validates that the ingestion process completed correctly
by checking indices, metadata, and performing test searches.
"""

import logging
import pickle  # nosec B403
import time
from pathlib import Path
from typing import Dict

import chromadb
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use pickle for internal data serialization - trusted source
# nosec B403


class IngestionValidator:
    """Validates the results of the ingestion pipeline."""

    def __init__(
        self,
        chroma_dir: Path,
        bm25_path: Path,
        embedding_model: str = "BAAI/bge-base-en-v1.5",
    ):
        """Initialize the validator.

        Args:
            chroma_dir: Directory containing ChromaDB
            bm25_path: Path to BM25 index file
            embedding_model: Name of embedding model
        """
        self.chroma_dir = chroma_dir
        self.bm25_path = bm25_path
        self.embedding_model_name = embedding_model

        # Initialize components
        self.chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
        self.embedding_model = SentenceTransformer(embedding_model)

        # Validation results
        self.results = {
            "chroma_validation": {},
            "bm25_validation": {},
            "search_tests": {},
            "metadata_checks": {},
            "overall_status": "unknown",
        }

    def validate_all(self) -> Dict:
        """Run all validation checks.

        Returns:
            Dictionary with validation results
        """
        logger.info("Starting ingestion validation...")

        # Check ChromaDB
        self._validate_chroma()

        # Check BM25 index
        self._validate_bm25()

        # Test search functionality
        self._test_searches()

        # Check metadata quality
        self._check_metadata()

        # Determine overall status
        self._determine_overall_status()

        # Print results
        self._print_results()

        return self.results

    def _validate_chroma(self) -> bool:
        """Validate ChromaDB index."""
        logger.info("Validating ChromaDB...")

        try:
            # Get collection
            collection = self.chroma_client.get_collection("mahabharata")

            # Check document count
            doc_count = collection.count()
            self.results["chroma_validation"]["document_count"] = doc_count

            if doc_count == 0:
                logger.error("ChromaDB contains no documents!")
                return False

            # Get a sample of documents
            sample = collection.get(limit=10, include=["metadatas", "documents"])

            # Check metadata structure
            if sample["metadatas"]:
                first_meta = sample["metadatas"][0]
                required_fields = [
                    "chunk_id",
                    "parva",
                    "adhyaya",
                    "characters",
                    "places",
                ]

                missing_fields = [f for f in required_fields if f not in first_meta]
                if missing_fields:
                    logger.error(f"Missing metadata fields: {missing_fields}")
                    self.results["chroma_validation"]["missing_fields"] = missing_fields
                    return False

                self.results["chroma_validation"]["metadata_ok"] = True

            # Test vector search
            query = "What is dharma?"
            query_embedding = self.embedding_model.encode([query]).tolist()

            search_results = collection.query(
                query_embeddings=query_embedding, n_results=5
            )

            if search_results["documents"] and len(search_results["documents"][0]) > 0:
                self.results["chroma_validation"]["vector_search_ok"] = True
                logger.info("✓ ChromaDB vector search working")
            else:
                logger.error("ChromaDB vector search failed!")
                return False

            self.results["chroma_validation"]["status"] = "pass"
            return True

        except Exception as e:
            logger.error(f"ChromaDB validation failed: {e}")
            self.results["chroma_validation"]["status"] = "fail"
            self.results["chroma_validation"]["error"] = str(e)
            return False

    def _validate_bm25(self) -> bool:
        """Validate BM25 index."""
        logger.info("Validating BM25 index...")

        try:
            # Load BM25 index
            if not self.bm25_path.exists():
                logger.error(f"BM25 index file not found: {self.bm25_path}")
                return False

            with open(self.bm25_path, "rb") as f:
                # Loading trusted internal data
                bm25_data = pickle.load(f)  # nosec B301

            # Check components
            required_keys = ["bm25_index", "chunk_ids", "documents", "metadatas"]
            missing_keys = [k for k in required_keys if k not in bm25_data]

            if missing_keys:
                logger.error(f"BM25 index missing keys: {missing_keys}")
                return False

            # Check document count
            doc_count = len(bm25_data["documents"])
            self.results["bm25_validation"]["document_count"] = doc_count

            if doc_count == 0:
                logger.error("BM25 index contains no documents!")
                return False

            # Test BM25 search
            test_query = "Arjuna"
            tokenized_query = test_query.lower().split()

            results = bm25_data["bm25_index"].get_top_n(
                tokenized_query, bm25_data["documents"], n=5
            )

            if len(results) > 0:
                self.results["bm25_validation"]["search_ok"] = True
                logger.info(f"✓ BM25 search returned {len(results)} results")
            else:
                logger.error("BM25 search returned no results!")
                return False

            self.results["bm25_validation"]["status"] = "pass"
            return True

        except Exception as e:
            logger.error(f"BM25 validation failed: {e}")
            self.results["bm25_validation"]["status"] = "fail"
            self.results["bm25_validation"]["error"] = str(e)
            return False

    def _test_searches(self):
        """Test various search scenarios."""
        logger.info("Testing search functionality...")

        test_queries = [
            ("Arjuna", "entity_search"),
            ("dharma", "concept_search"),
            ("Kurukshetra battle", "event_search"),
            ("Gandiva bow", "weapon_search"),
            ("Hastinapura", "place_search"),
        ]

        search_results = {}

        # Load BM25 for testing
        with open(self.bm25_path, "rb") as f:
            # Loading trusted internal data
            bm25_data = pickle.load(f)  # nosec B301

        # Get ChromaDB collection
        collection = self.chroma_client.get_collection("mahabharata")

        for query, query_type in test_queries:
            start_time = time.time()

            # Test BM25
            tokenized_query = query.lower().split()
            bm25_results = bm25_data["bm25_index"].get_top_n(
                tokenized_query, bm25_data["documents"], n=5
            )

            # Test vector search
            query_embedding = self.embedding_model.encode([query]).tolist()
            vector_results = collection.query(
                query_embeddings=query_embedding, n_results=5
            )

            elapsed = time.time() - start_time

            search_results[query_type] = {
                "query": query,
                "bm25_count": len(bm25_results),
                "vector_count": (
                    len(vector_results["documents"][0])
                    if vector_results["documents"]
                    else 0
                ),
                "latency_ms": elapsed * 1000,
            }

            logger.info(
                f"Query '{query}': BM25={len(bm25_results)}, Vector={len(vector_results['documents'][0]) if vector_results['documents'] else 0}"
            )

        self.results["search_tests"] = search_results

    def _check_metadata(self):
        """Check metadata quality and completeness."""
        logger.info("Checking metadata quality...")

        # Get sample from ChromaDB
        collection = self.chroma_client.get_collection("mahabharata")
        sample = collection.get(limit=100, include=["metadatas"])

        if not sample["metadatas"]:
            logger.error("No metadata found!")
            return

        # Analyze metadata
        parvas = set()
        chapters = set()
        chunks_with_characters = 0
        chunks_with_places = 0
        chunks_with_weapons = 0
        avg_chunk_size = 0

        for meta in sample["metadatas"]:
            parvas.add(meta.get("parva", "unknown"))
            chapters.add(meta.get("adhyaya", "unknown"))

            if meta.get("characters"):
                chunks_with_characters += 1
            if meta.get("places"):
                chunks_with_places += 1
            if meta.get("weapons"):
                chunks_with_weapons += 1

            avg_chunk_size += meta.get("token_count", 0)

        total_chunks = len(sample["metadatas"])
        avg_chunk_size /= total_chunks if total_chunks > 0 else 1

        self.results["metadata_checks"] = {
            "total_parvas": len(parvas),
            "parvas": sorted(list(parvas)),
            "total_chapters": len(chapters),
            "chunks_with_characters": chunks_with_characters,
            "chunks_with_places": chunks_with_places,
            "chunks_with_weapons": chunks_with_weapons,
            "character_coverage": (chunks_with_characters / total_chunks) * 100,
            "place_coverage": (chunks_with_places / total_chunks) * 100,
            "weapon_coverage": (chunks_with_weapons / total_chunks) * 100,
            "avg_chunk_size_tokens": avg_chunk_size,
        }

        logger.info(f"Found {len(parvas)} parvas and {len(chapters)} chapters")
        logger.info(
            f"Character coverage: {self.results['metadata_checks']['character_coverage']:.1f}%"
        )
        logger.info(f"Average chunk size: {avg_chunk_size:.0f} tokens")

    def _determine_overall_status(self):
        """Determine overall validation status."""
        chroma_ok = self.results["chroma_validation"].get("status") == "pass"
        bm25_ok = self.results["bm25_validation"].get("status") == "pass"

        if chroma_ok and bm25_ok:
            self.results["overall_status"] = "pass"
        else:
            self.results["overall_status"] = "fail"

    def _print_results(self):
        """Print validation results."""
        print("\n" + "=" * 60)
        print("INGESTION VALIDATION RESULTS")
        print("=" * 60)

        # ChromaDB results
        print("\nChromaDB Validation:")
        chroma_status = self.results["chroma_validation"].get("status", "unknown")
        print(f"  Status: {chroma_status.upper()}")
        if "document_count" in self.results["chroma_validation"]:
            print(f"  Documents: {self.results['chroma_validation']['document_count']}")

        # BM25 results
        print("\nBM25 Validation:")
        bm25_status = self.results["bm25_validation"].get("status", "unknown")
        print(f"  Status: {bm25_status.upper()}")
        if "document_count" in self.results["bm25_validation"]:
            print(f"  Documents: {self.results['bm25_validation']['document_count']}")

        # Search tests
        print("\nSearch Tests:")
        for query_type, result in self.results["search_tests"].items():
            print(f"  {result['query']}:")
            print(f"    BM25: {result['bm25_count']} results")
            print(f"    Vector: {result['vector_count']} results")
            print(f"    Latency: {result['latency_ms']:.1f}ms")

        # Metadata checks
        if self.results["metadata_checks"]:
            print("\nMetadata Quality:")
            mc = self.results["metadata_checks"]
            print(f"  Parvas found: {mc['total_parvas']}")
            print(f"  Character coverage: {mc['character_coverage']:.1f}%")
            print(f"  Place coverage: {mc['place_coverage']:.1f}%")
            print(f"  Average chunk size: {mc['avg_chunk_size_tokens']:.0f} tokens")

        # Overall status
        print("\n" + "=" * 60)
        overall = self.results["overall_status"].upper()
        print(f"OVERALL STATUS: {overall}")
        print("=" * 60)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate Mahabharata ingestion")
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=Path("data/chroma"),
        help="ChromaDB directory",
    )
    parser.add_argument(
        "--bm25-path",
        type=Path,
        default=Path("data/bm25_index.pkl"),
        help="BM25 index path",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help="Embedding model",
    )

    args = parser.parse_args()

    # Run validation
    validator = IngestionValidator(
        chroma_dir=args.chroma_dir,
        bm25_path=args.bm25_path,
        embedding_model=args.embedding_model,
    )

    results = validator.validate_all()

    # Exit with appropriate code
    exit(0 if results["overall_status"] == "pass" else 1)


if __name__ == "__main__":
    main()
