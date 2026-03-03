"""Integration tests for the retrieval pipeline."""

import pickle
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import chromadb
import numpy as np
import pytest

from src.retrieval.pipeline import RetrievalPipeline


class TestRetrievalPipeline:
    """Integration tests for the complete retrieval pipeline."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test indices."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            {
                "chunk_id": "chunk_001",
                "content": "Arjuna, the third Pandava, was a master archer known as Dhananjaya.",
                "metadata": {
                    "parva": "Adi Parva",
                    "section": "Chapter 1",
                    "characters": ["Arjuna"],
                    "places": ["Hastinapura"],
                },
            },
            {
                "chunk_id": "chunk_002",
                "content": "The Bhagavad Gita is a philosophical dialogue between Krishna and Arjuna on the battlefield of Kurukshetra.",
                "metadata": {
                    "parva": "Bhishma Parva",
                    "section": "Bhagavad Gita",
                    "characters": ["Krishna", "Arjuna"],
                    "places": ["Kurukshetra"],
                    "concepts": ["dharma", "karma"],
                },
            },
            {
                "chunk_id": "chunk_003",
                "content": "Dharma represents moral duty and righteousness in the Mahabharata, guiding the actions of all characters.",
                "metadata": {
                    "parva": "Shanti Parva",
                    "section": "Chapter 100",
                    "concepts": ["dharma", "righteousness"],
                },
            },
            {
                "chunk_id": "chunk_004",
                "content": "The Kurukshetra war lasted for eighteen days, resulting in great loss of life on both sides.",
                "metadata": {
                    "parva": "Karna Parva",
                    "section": "War Description",
                    "places": ["Kurukshetra"],
                    "events": ["war"],
                },
            },
        ]

    @pytest.fixture
    def setup_indices(self, temp_dir, sample_documents):
        """Set up test indices (ChromaDB and BM25)."""
        chroma_dir = temp_dir / "chroma"
        bm25_path = temp_dir / "bm25.pkl"

        # Create ChromaDB index
        chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = chroma_client.create_collection(
            name="mahabharata", metadata={"hnsw:space": "cosine"}
        )

        # Add documents to ChromaDB
        texts = [doc["content"] for doc in sample_documents]
        metadatas = [doc["metadata"] for doc in sample_documents]
        ids = [doc["chunk_id"] for doc in sample_documents]

        # Mock embeddings
        embeddings = np.random.rand(len(texts), 384).tolist()

        collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings,
        )

        # Create BM25 index
        tokenized_docs = [doc["content"].lower().split() for doc in sample_documents]
        from rank_bm25 import BM25Okapi

        bm25_index = BM25Okapi(tokenized_docs)

        # Save BM25 index
        bm25_data = {
            "bm25_index": bm25_index,
            "chunk_ids": ids,
            "documents": texts,
            "metadatas": metadatas,
        }

        with open(bm25_path, "wb") as f:
            pickle.dump(bm25_data, f)

        return chroma_dir, bm25_path

    @pytest.fixture
    def pipeline(self, setup_indices):
        """Create a test retrieval pipeline."""
        chroma_dir, bm25_path = setup_indices

        with (
            patch("sentence_transformers.SentenceTransformer") as mock_model,
            patch("src.retrieval.pipeline.QueryClassifier") as mock_classifier,
            patch("src.retrieval.pipeline.Reranker") as mock_reranker,
        ):

            # Mock embedding model
            mock_model.return_value.encode.return_value = np.random.rand(1, 384)

            # Mock query classifier
            mock_classifier.return_value.get_retrieval_strategy.return_value = {
                "query_type": "entity",
                "confidence": 0.8,
                "use_bm25": True,
                "use_dense": True,
                "bm25_weight": 0.6,
                "dense_weight": 0.4,
                "rerank": False,
                "top_k": 5,
                "rerank_top_k": 3,
                "query_expansion": False,
                "hyde": False,
            }

            # Mock reranker
            mock_reranker.return_value.rerank.side_effect = lambda q, docs, **kw: docs

            pipeline = RetrievalPipeline(
                chroma_dir=chroma_dir,
                bm25_path=bm25_path,
                enable_reranking=True,
                enable_query_classification=True,
            )

            return pipeline

    def test_pipeline_initialization(self, setup_indices):
        """Test pipeline initialization."""
        chroma_dir, bm25_path = setup_indices

        with (
            patch("sentence_transformers.SentenceTransformer"),
            patch("src.retrieval.query_classifier.SentenceTransformer"),
            patch("src.retrieval.pipeline.QueryClassifier"),
            patch("src.retrieval.pipeline.Reranker"),
        ):

            pipeline = RetrievalPipeline(
                chroma_dir=chroma_dir,
                bm25_path=bm25_path,
            )

            assert pipeline.chroma_dir == chroma_dir
            assert pipeline.bm25_path == bm25_path
            assert pipeline.hybrid_searcher is not None
            assert pipeline.rank_fusion is not None
            assert pipeline.query_classifier is not None
            assert pipeline.reranker is not None

    def test_basic_retrieval(self, pipeline):
        """Test basic retrieval functionality."""
        query = "Who is Arjuna?"

        result = pipeline.retrieve(query, top_k=3)

        assert "query" in result
        assert "results" in result
        assert "strategy" in result
        assert "num_retrieved" in result
        assert "num_returned" in result

        assert result["query"] == query
        assert len(result["results"]) <= 3
        assert result["num_returned"] <= result["num_retrieved"]

        # Check result structure
        if result["results"]:
            first_result = result["results"][0]
            assert "chunk_id" in first_result
            assert "content" in first_result
            assert "metadata" in first_result

    def test_retrieval_with_filters(self, pipeline):
        """Test retrieval with metadata filters."""
        query = "Tell me about characters"
        filters = {"characters": ["Arjuna"]}

        result = pipeline.retrieve(query, top_k=5, filters=filters)

        # All results should have Arjuna in characters
        for doc in result["results"]:
            assert "Arjuna" in doc["metadata"].get("characters", [])

    def test_retrieval_with_conversation_history(self, pipeline):
        """Test contextual retrieval with conversation history."""
        query = "What did he teach?"
        history = ["Who is Krishna?", "Tell me about his role"]

        result = pipeline.retrieve(query, top_k=3, conversation_history=history)

        assert len(result["results"]) <= 3
        # The reranker should have been called with context
        if pipeline.reranker:
            pipeline.reranker.contextual_rerank.assert_called()

    def test_batch_retrieval(self, pipeline):
        """Test batch retrieval for multiple queries."""
        queries = ["Who is Arjuna?", "What is dharma?", "Kurukshetra war"]

        results = pipeline.batch_retrieve(queries, top_k=2)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result["query"] == queries[i]
            assert "results" in result
            assert len(result["results"]) <= 2

    def test_pipeline_statistics(self, pipeline):
        """Test pipeline statistics."""
        stats = pipeline.get_pipeline_statistics()

        assert "components" in stats
        assert "models" in stats

        assert "query_classifier" in stats["components"]
        assert "hybrid_searcher" in stats["components"]
        assert "rank_fusion" in stats["components"]
        assert "reranker" in stats["components"]

        assert "embedding_model" in stats["models"]
        assert "device" in stats["models"]

    def test_health_check(self, pipeline):
        """Test pipeline health check."""
        health = pipeline.health_check()

        assert "overall" in health
        assert "components" in health

        assert health["overall"] in ["healthy", "degraded", "unhealthy"]
        assert len(health["components"]) == 5  # All 5 components

    def test_retrieval_without_query_classification(self, setup_indices):
        """Test retrieval with query classification disabled."""
        chroma_dir, bm25_path = setup_indices

        with (
            patch("sentence_transformers.SentenceTransformer"),
            patch("src.retrieval.query_classifier.SentenceTransformer"),
            patch("src.retrieval.pipeline.Reranker") as mock_reranker,
        ):

            mock_reranker.return_value.rerank.side_effect = lambda q, docs, **kw: docs

            pipeline = RetrievalPipeline(
                chroma_dir=chroma_dir,
                bm25_path=bm25_path,
                enable_query_classification=False,
            )

            result = pipeline.retrieve("Test query", top_k=2)

            assert "results" in result
            assert result["strategy"]["query_type"] == "unknown"

    def test_retrieval_without_reranking(self, setup_indices):
        """Test retrieval with reranking disabled."""
        chroma_dir, bm25_path = setup_indices

        with (
            patch("src.retrieval.pipeline.SentenceTransformer"),
            patch("src.retrieval.pipeline.QueryClassifier"),
        ):

            pipeline = RetrievalPipeline(
                chroma_dir=chroma_dir,
                bm25_path=bm25_path,
                enable_reranking=False,
            )

            result = pipeline.retrieve("Test query", top_k=2)

            assert "results" in result
            assert result["strategy"]["rerank"] is False
            assert pipeline.reranker is None

    def test_error_handling(self, setup_indices, temp_dir):
        """Test error handling in retrieval."""
        chroma_dir, bm25_path = setup_indices

        # Create pipeline with invalid BM25 path
        invalid_path = temp_dir / "invalid.pkl"

        with pytest.raises(FileNotFoundError):
            RetrievalPipeline(
                chroma_dir=chroma_dir,
                bm25_path=invalid_path,
            )

    def test_query_expansion(self, setup_indices):
        """Test query expansion functionality."""
        chroma_dir, bm25_path = setup_indices

        with (
            patch("src.retrieval.pipeline.SentenceTransformer"),
            patch("src.retrieval.pipeline.QueryClassifier") as mock_classifier,
            patch("src.retrieval.pipeline.Reranker"),
        ):

            # Configure strategy with query expansion
            mock_classifier.return_value.get_retrieval_strategy.return_value = {
                "query_type": "entity",
                "confidence": 0.8,
                "use_bm25": True,
                "use_dense": True,
                "bm25_weight": 0.5,
                "dense_weight": 0.5,
                "rerank": False,
                "top_k": 5,
                "rerank_top_k": 3,
                "query_expansion": True,
                "hyde": False,
            }

            pipeline = RetrievalPipeline(
                chroma_dir=chroma_dir,
                bm25_path=bm25_path,
                enable_reranking=False,
            )

            result = pipeline.retrieve("krishna", top_k=3)

            assert "expanded_query" in result
            assert "krishna" in result["expanded_query"].lower()

    def test_hyde_query_transformation(self, setup_indices):
        """Test HyDE query transformation."""
        chroma_dir, bm25_path = setup_indices

        with (
            patch("src.retrieval.pipeline.SentenceTransformer"),
            patch("src.retrieval.pipeline.QueryClassifier") as mock_classifier,
            patch("src.retrieval.pipeline.Reranker"),
        ):

            # Configure strategy with HyDE
            mock_classifier.return_value.get_retrieval_strategy.return_value = {
                "query_type": "philosophical",
                "confidence": 0.8,
                "use_bm25": True,
                "use_dense": True,
                "bm25_weight": 0.3,
                "dense_weight": 0.7,
                "rerank": False,
                "top_k": 5,
                "rerank_top_k": 3,
                "query_expansion": False,
                "hyde": True,
            }

            pipeline = RetrievalPipeline(
                chroma_dir=chroma_dir,
                bm25_path=bm25_path,
                enable_reranking=False,
            )

            result = pipeline.retrieve("dharma", top_k=3)

            assert "hyde_query" in result
            assert "according to the mahabharata" in result["hyde_query"].lower()


if __name__ == "__main__":
    pytest.main([__file__])
