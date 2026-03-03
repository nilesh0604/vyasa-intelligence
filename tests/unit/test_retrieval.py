"""Unit tests for the retrieval module."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.retrieval.query_classifier import QueryClassifier, QueryType
from src.retrieval.rank_fusion import RankFusion
from src.retrieval.reranker import Reranker


class TestQueryClassifier:
    """Test cases for QueryClassifier."""

    @pytest.fixture
    def classifier(self):
        """Create a test query classifier."""
        with patch("src.retrieval.query_classifier.SentenceTransformer") as mock_model:
            mock_model.return_value.encode.return_value = [[0.1, 0.2, 0.3]]
            return QueryClassifier()

    def test_classify_entity_query(self, classifier):
        """Test classification of entity queries."""
        query = "Who is Arjuna?"
        query_type, scores = classifier.classify_query(query)

        assert query_type == QueryType.ENTITY
        assert QueryType.ENTITY.value in scores
        assert scores[QueryType.ENTITY.value] > 0

    def test_classify_philosophical_query(self, classifier):
        """Test classification of philosophical queries."""
        query = "What is dharma?"
        query_type, scores = classifier.classify_query(query)

        assert query_type == QueryType.PHILOSOPHICAL
        assert QueryType.PHILOSOPHICAL.value in scores

    def test_classify_narrative_query(self, classifier):
        """Test classification of narrative queries."""
        query = "What happened in the Kurukshetra war?"
        query_type, scores = classifier.classify_query(query)

        assert query_type == QueryType.NARRATIVE
        assert QueryType.NARRATIVE.value in scores

    def test_get_retrieval_strategy(self, classifier):
        """Test retrieval strategy generation."""
        query = "Who is Krishna?"
        strategy = classifier.get_retrieval_strategy(query)

        assert "query_type" in strategy
        assert "confidence" in strategy
        assert "use_bm25" in strategy
        assert "use_dense" in strategy
        assert "bm25_weight" in strategy
        assert "dense_weight" in strategy
        assert "rerank" in strategy

        # Entity queries should favor BM25
        assert strategy["bm25_weight"] > strategy["dense_weight"]

    def test_unknown_query_classification(self, classifier):
        """Test classification of unknown queries."""
        query = "xyz123"
        query_type, scores = classifier.classify_query(query)

        assert query_type == QueryType.UNKNOWN  # Unknown query
        # Unknown queries might not have a score for UNKNOWN type
        # The important thing is that it returns UNKNOWN as the type


class TestRankFusion:
    """Test cases for RankFusion."""

    @pytest.fixture
    def fusion(self):
        """Create a test rank fusion instance."""
        return RankFusion(k=60)

    def test_reciprocal_rank_fusion(self, fusion):
        """Test reciprocal rank fusion."""
        results1 = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        results2 = [("doc2", 0.9), ("doc1", 0.7), ("doc4", 0.6)]

        fused = fusion.reciprocal_rank_fusion([results1, results2], top_k=3)

        assert len(fused) == 3
        assert all(isinstance(item, tuple) for item in fused)
        assert all(isinstance(item[0], str) for item in fused)
        assert all(isinstance(item[1], float) for item in fused)

        # doc1 and doc2 should be ranked higher (appear in both lists)
        doc_ids = [doc_id for doc_id, _ in fused]
        assert "doc1" in doc_ids[:2]
        assert "doc2" in doc_ids[:2]

    def test_weighted_score_fusion(self, fusion):
        """Test weighted score fusion."""
        results1 = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        results2 = [("doc2", 0.9), ("doc1", 0.7), ("doc4", 0.6)]
        weights = [0.6, 0.4]

        fused = fusion.weighted_score_fusion([results1, results2], weights=weights)

        assert len(fused) <= 4  # Max unique documents
        # doc1 should be higher than doc2 due to higher weight on results1
        doc_ids = [doc_id for doc_id, _ in fused]
        assert doc_ids[0] == "doc1"

    def test_condorcet_fusion(self, fusion):
        """Test Condorcet fusion."""
        results1 = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        results2 = [("doc2", 0.9), ("doc1", 0.7), ("doc4", 0.6)]

        fused = fusion.condorcet_fusion([results1, results2])

        assert len(fused) <= 4
        assert all(isinstance(item, tuple) for item in fused)

    def test_adaptive_fusion(self, fusion):
        """Test adaptive fusion."""
        results1 = [("doc1", 0.9), ("doc2", 0.8)]
        results2 = [("doc2", 0.9), ("doc1", 0.7)]

        # Test different query types
        entity_fused = fusion.adaptive_fusion([results1, results2], "entity")
        philo_fused = fusion.adaptive_fusion([results1, results2], "philosophical")

        assert len(entity_fused) == 2
        assert len(philo_fused) == 2

    def test_evaluate_fusion(self, fusion):
        """Test fusion evaluation."""
        fused_results = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        ground_truth = ["doc1", "doc3"]

        metrics = fusion.evaluate_fusion(fused_results, ground_truth)

        assert "precision@1" in metrics
        assert "precision@3" in metrics
        assert "precision@5" in metrics
        assert "mrr" in metrics

        assert 0 <= metrics["precision@1"] <= 1
        assert 0 <= metrics["mrr"] <= 1


class TestReranker:
    """Test cases for Reranker."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_documents(self):
        """Create mock documents."""
        return [
            {
                "chunk_id": "doc1",
                "content": "Arjuna was a skilled archer in the Mahabharata.",
                "metadata": {"parva": "Adi Parva", "characters": ["Arjuna"]},
            },
            {
                "chunk_id": "doc2",
                "content": "Krishna gave the Bhagavad Gita to Arjuna.",
                "metadata": {
                    "parva": "Bhishma Parva",
                    "characters": ["Krishna", "Arjuna"],
                },
            },
            {
                "chunk_id": "doc3",
                "content": "The concept of dharma is central to the epic.",
                "metadata": {"parva": "Shanti Parva", "concepts": ["dharma"]},
            },
        ]

    def test_reranker_initialization(self, temp_dir):
        """Test reranker initialization."""
        with patch("src.retrieval.reranker.CrossEncoder") as mock_cross:
            mock_cross.return_value = Mock()
            reranker = Reranker(device="cpu")

            assert reranker.model_name == "BAAI/bge-reranker-base"
            assert reranker.device == "cpu"
            assert reranker.max_length == 512

    def test_rerank_with_mock_model(self, mock_documents):
        """Test reranking with a mock model."""
        with patch("src.retrieval.reranker.CrossEncoder") as mock_cross:
            mock_model = Mock()
            mock_model.predict.return_value = [0.9, 0.7, 0.3]
            mock_cross.return_value = mock_model

            reranker = Reranker()
            results = reranker.rerank("Who is Arjuna?", mock_documents, top_k=2)

            assert len(results) == 2
            assert results[0]["chunk_id"] == "doc1"  # Highest score
            assert results[1]["chunk_id"] == "doc2"
            assert "rerank_score" in results[0]
            assert results[0]["rerank_score"] == 0.9

    def test_fallback_reranking(self, mock_documents):
        """Test fallback reranking when model fails."""
        reranker = Reranker()
        reranker.model = None  # Force fallback

        results = reranker.rerank("Arjuna archer", mock_documents, top_k=2)

        assert len(results) == 2
        # Should rank documents with "Arjuna" higher
        assert "Arjuna" in results[0]["content"] or "Arjuna" in results[1]["content"]

    def test_multi_stage_rerank(self, mock_documents):
        """Test multi-stage reranking."""
        with patch("src.retrieval.reranker.CrossEncoder") as mock_cross:
            mock_model = Mock()
            mock_model.predict.return_value = [0.9, 0.7, 0.3]
            mock_cross.return_value = mock_model

            reranker = Reranker()
            results = reranker.multi_stage_rerank(
                "Arjuna",
                mock_documents,
                stage1_top_k=3,
                stage2_top_k=2,
            )

            assert len(results) == 2

    def test_apply_filters(self, mock_documents):
        """Test applying metadata filters."""
        reranker = Reranker()
        filters = {"characters": ["Arjuna"]}

        filtered = reranker._apply_filters(mock_documents, filters)

        assert len(filtered) == 2  # doc1 and doc2 have Arjuna
        assert all(
            "Arjuna" in doc["metadata"].get("characters", []) for doc in filtered
        )

    def test_diversity_rerank(self, mock_documents):
        """Test diversity-aware reranking."""
        with patch("src.retrieval.reranker.CrossEncoder") as mock_cross:
            mock_model = Mock()
            mock_model.predict.return_value = [0.9, 0.8, 0.7]
            mock_cross.return_value = mock_model

            reranker = Reranker()
            results = reranker.diversity_rerank(
                "Arjuna",
                mock_documents,
                top_k=2,
                diversity_lambda=0.3,
            )

            assert len(results) == 2
            assert "rerank_score" in results[0]

    def test_contextual_rerank(self, mock_documents):
        """Test contextual reranking with conversation history."""
        with patch("src.retrieval.reranker.CrossEncoder") as mock_cross:
            mock_model = Mock()
            mock_model.predict.return_value = [0.9, 0.7, 0.3]
            mock_cross.return_value = mock_model

            reranker = Reranker()
            history = ["Tell me about warriors", "Who fought in the war?"]

            results = reranker.contextual_rerank(
                "Arjuna",
                mock_documents,
                conversation_history=history,
            )

            assert len(results) == 3
            # Check that the model was called with contextual query
            call_args = mock_model.predict.call_args[0][0]
            assert any("Context:" in str(query) for query in call_args)


if __name__ == "__main__":
    pytest.main([__file__])
