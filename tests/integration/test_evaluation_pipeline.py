"""Integration tests for the evaluation pipeline."""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.evaluation import (MahabharataEvaluator, QualityGate,
                            QualityGateEvaluator)


class TestQualityGate:
    """Test individual quality gate functionality."""

    def test_quality_gate_gte(self):
        """Test quality gate with greater than or equal comparison."""
        gate = QualityGate(name="test_metric", threshold=0.8, comparison="gte")

        # Test passing case
        result = gate.evaluate(0.85)
        assert result["passed"] is True
        assert abs(result["delta"] - 0.05) < 1e-6

        # Test failing case
        result = gate.evaluate(0.75)
        assert result["passed"] is False
        assert abs(result["delta"] + 0.05) < 1e-6

        # Test edge case
        result = gate.evaluate(0.8)
        assert result["passed"] is True
        assert result["delta"] == 0.0

    def test_quality_gate_lte(self):
        """Test quality gate with less than or equal comparison."""
        gate = QualityGate(name="error_rate", threshold=0.1, comparison="lte")

        # Test passing case
        result = gate.evaluate(0.05)
        assert result["passed"] is True
        assert abs(result["delta"] - 0.05) < 1e-6

        # Test failing case
        result = gate.evaluate(0.15)
        assert result["passed"] is False
        assert abs(result["delta"] + 0.05) < 1e-6

    def test_quality_gate_with_weight(self):
        """Test quality gate with weight."""
        gate = QualityGate(
            name="weighted_metric",
            threshold=0.7,
            weight=0.5,
            description="A weighted metric",
        )

        result = gate.evaluate(0.8)
        assert result["weight"] == 0.5
        assert result["description"] == "A weighted metric"


class TestQualityGateEvaluator:
    """Test quality gate evaluator functionality."""

    def test_default_gates(self):
        """Test default quality gates for Mahabharata system."""
        evaluator = QualityGateEvaluator()

        assert len(evaluator.gates) == 5
        gate_names = [gate.name for gate in evaluator.gates]
        assert "faithfulness" in gate_names
        assert "answer_relevancy" in gate_names

        # Check faithfulness gate
        faithfulness_gate = next(g for g in evaluator.gates if g.name == "faithfulness")
        assert faithfulness_gate.threshold == 0.85
        assert faithfulness_gate.weight == 0.3

    def test_evaluate_all_passed(self):
        """Test evaluation when all gates pass."""
        evaluator = QualityGateEvaluator()

        scores = {
            "faithfulness": 0.90,
            "answer_relevancy": 0.85,
            "context_precision": 0.88,
            "context_recall": 0.82,
            "answer_similarity": 0.80,
        }

        result = evaluator.evaluate(scores)

        assert result["overall_passed"] is True
        assert result["overall_score"] == 1.0
        assert len(result["passed_gates"]) == 5
        assert len(result["failed_gates"]) == 0
        assert result["summary"]["pass_rate"] == 1.0

    def test_evaluate_some_failed(self):
        """Test evaluation when some gates fail."""
        evaluator = QualityGateEvaluator()

        scores = {
            "faithfulness": 0.80,  # Below threshold
            "answer_relevancy": 0.85,
            "context_precision": 0.88,
            "context_recall": 0.75,  # Below threshold
            "answer_similarity": 0.80,
        }

        result = evaluator.evaluate(scores, strict_mode=True)

        assert result["overall_passed"] is False
        assert result["overall_score"] < 1.0
        assert len(result["passed_gates"]) == 3
        assert len(result["failed_gates"]) == 2
        assert result["summary"]["pass_rate"] == 0.6

    def test_evaluate_strict_mode(self):
        """Test evaluation in strict mode."""
        evaluator = QualityGateEvaluator()

        scores = {
            "faithfulness": 0.90,
            "answer_relevancy": 0.85,
            "context_precision": 0.88,
            "context_recall": 0.75,  # Below threshold
            "answer_similarity": 0.80,
        }

        # Non-strict mode should pass based on weighted score
        result = evaluator.evaluate(scores, strict_mode=False)
        assert result["overall_passed"] is True

        # Strict mode should fail
        result = evaluator.evaluate(scores, strict_mode=True)
        assert result["overall_passed"] is False

    def test_get_improvement_suggestions(self):
        """Test improvement suggestions for failed gates."""
        evaluator = QualityGateEvaluator()

        scores = {
            "faithfulness": 0.70,  # Well below threshold
            "answer_relevancy": 0.78,  # Slightly below threshold
        }

        result = evaluator.evaluate(scores)
        suggestions = evaluator.get_improvement_suggestions(result)

        assert len(suggestions) > 0
        assert any("faithfulness" in s.lower() for s in suggestions)
        assert any("relevancy" in s.lower() for s in suggestions)

    def test_generate_report(self):
        """Test quality gate report generation."""
        evaluator = QualityGateEvaluator()

        scores = {
            "faithfulness": 0.90,
            "answer_relevancy": 0.78,  # Failed
            "context_precision": 0.88,
        }

        result = evaluator.evaluate(scores)
        report = evaluator.generate_report(result)

        assert "QUALITY GATE EVALUATION REPORT" in report
        assert "PASSED GATES:" in report
        assert "FAILED GATES:" in report
        assert "IMPROVEMENT SUGGESTIONS:" in report
        assert "answer_relevancy" in report


class TestMahabharataEvaluator:
    """Test Mahabharata evaluator functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def golden_dataset_path(self, temp_dir):
        """Create a test golden dataset."""
        dataset_path = temp_dir / "test_golden.jsonl"

        test_data = [
            {
                "question": "Who was Arjuna?",
                "answer": "Arjuna was the third Pandava brother and a master archer.",
                "contexts": [
                    "Arjuna, the third Pandava, was a master archer.",
                    "He wielded the divine bow Gandiva.",
                ],
                "metadata": {"parva": "Adi Parva", "difficulty": "easy"},
            },
            {
                "question": "What is dharma?",
                "answer": "Dharma represents moral duty and righteousness.",
                "contexts": [
                    "Dharma represents moral duty and righteousness in the Mahabharata.",
                    "It guides the actions of all characters.",
                ],
                "metadata": {"parva": "Bhishma Parva", "difficulty": "medium"},
            },
        ]

        with open(dataset_path, "w", encoding="utf-8") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")

        return dataset_path

    @pytest.fixture
    def evaluator(self, golden_dataset_path, temp_dir):
        """Create a test evaluator."""
        output_dir = temp_dir / "evaluation"
        return MahabharataEvaluator(
            golden_dataset_path=golden_dataset_path,
            output_dir=output_dir,
            metrics=["faithfulness", "answer_relevancy"],
        )

    def test_load_golden_dataset(self, evaluator):
        """Test loading golden dataset."""
        assert len(evaluator.golden_dataset) == 2
        assert evaluator.golden_dataset[0]["question"] == "Who was Arjuna?"
        assert evaluator.golden_dataset[1]["question"] == "What is dharma?"

    def test_prepare_ragas_dataset(self, evaluator):
        """Test preparing dataset for Ragas evaluation."""
        rag_results = [
            {
                "question": "Who was Arjuna?",
                "answer": "Arjuna was a great warrior.",
                "retrieved_contexts": ["Arjuna was the third Pandava."],
            },
            {
                "question": "What is dharma?",
                "answer": "Dharma is duty.",
                "retrieved_contexts": ["Dharma represents moral duty."],
            },
        ]

        dataset = evaluator.prepare_ragas_dataset(rag_results)

        assert len(dataset) == 2
        assert dataset[0]["question"] == "Who was Arjuna?"
        assert (
            dataset[0]["ground_truth"]
            == "Arjuna was the third Pandava brother and a master archer."
        )
        assert dataset[0]["contexts"] == ["Arjuna was the third Pandava."]

    @patch("src.evaluation.evaluator.evaluate")
    def test_evaluate_with_mock(self, mock_ragas_evaluate, evaluator):
        """Test evaluation with mocked Ragas."""
        # Mock Ragas evaluation result
        mock_df = Mock()
        mock_df.to_dict.return_value = [
            {"faithfulness": 0.9, "answer_relevancy": 0.85},
            {"faithfulness": 0.8, "answer_relevancy": 0.75},
        ]
        mock_result = Mock()
        mock_result.to_pandas.return_value = mock_df
        mock_ragas_evaluate.return_value = mock_result

        # Mock the metrics in the result
        mock_result.__contains__ = Mock(return_value=True)
        mock_result.__getitem__ = Mock(
            side_effect=lambda x: {
                "faithfulness": mock_df,
                "answer_relevancy": mock_df,
            }.get(x)
        )

        # Create mock RAG results
        rag_results = [
            {
                "question": "Who was Arjuna?",
                "answer": "Arjuna was a great warrior.",
                "retrieved_contexts": ["Arjuna was the third Pandava."],
            },
            {
                "question": "What is dharma?",
                "answer": "Dharma is duty.",
                "retrieved_contexts": ["Dharma represents moral duty."],
            },
        ]

        # Run evaluation
        results = evaluator.evaluate(rag_results, run_name="test_run")

        # Check results
        assert results["run_name"] == "test_run"
        assert results["num_samples"] == 2
        assert "faithfulness" in results["aggregate_scores"]
        assert "answer_relevancy" in results["aggregate_scores"]
        assert "quality_gates" in results
        assert "quality_gate_report" in results

    def test_compare_evaluations(self, evaluator, temp_dir):
        """Test comparing multiple evaluation runs."""
        # Create mock evaluation files
        eval1 = {
            "run_name": "run1",
            "timestamp": "2024-01-01",
            "num_samples": 10,
            "passed_all_gates": True,
            "aggregate_scores": {
                "faithfulness": {"mean": 0.9, "std": 0.05},
                "answer_relevancy": {"mean": 0.85, "std": 0.03},
            },
            "quality_gates": {
                "overall_passed": True,
                "all_results": [
                    {
                        "name": "faithfulness",
                        "passed": True,
                        "threshold": 0.85,
                        "value": 0.9,
                    },
                    {
                        "name": "answer_relevancy",
                        "passed": True,
                        "threshold": 0.8,
                        "value": 0.85,
                    },
                ],
            },
        }

        eval2 = {
            "run_name": "run2",
            "timestamp": "2024-01-02",
            "num_samples": 10,
            "passed_all_gates": False,
            "aggregate_scores": {
                "faithfulness": {"mean": 0.85, "std": 0.04},
                "answer_relevancy": {"mean": 0.78, "std": 0.05},
            },
            "quality_gates": {
                "overall_passed": False,
                "all_results": [
                    {
                        "name": "faithfulness",
                        "passed": True,
                        "threshold": 0.85,
                        "value": 0.85,
                    },
                    {
                        "name": "answer_relevancy",
                        "passed": False,
                        "threshold": 0.8,
                        "value": 0.78,
                    },
                ],
            },
        }

        # Save evaluation files
        file1 = evaluator.output_dir / "eval1.json"
        file2 = evaluator.output_dir / "eval2.json"

        with open(file1, "w") as f:
            json.dump(eval1, f)
        with open(file2, "w") as f:
            json.dump(eval2, f)

        # Compare evaluations
        comparison = evaluator.compare_evaluations(["eval1.json", "eval2.json"])

        assert comparison["comparison_name"] == "comparison_2_runs"
        assert len(comparison["runs"]) == 2
        assert "faithfulness" in comparison["metric_comparison"]
        assert comparison["metric_comparison"]["faithfulness"]["best_run"] == "run1"
        assert comparison["metric_comparison"]["faithfulness"]["worst_run"] == "run2"


if __name__ == "__main__":
    pytest.main([__file__])
