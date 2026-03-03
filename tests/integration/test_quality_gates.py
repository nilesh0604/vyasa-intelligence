"""Integration tests for quality gates evaluation."""

import json
import tempfile
from pathlib import Path

import pytest

from src.evaluation.quality_gates import QualityGate, QualityGateEvaluator


class TestQualityGate:
    """Test individual quality gate functionality."""

    def test_quality_gate_gte(self):
        """Test quality gate with greater-than-or-equal comparison."""
        gate = QualityGate(name="test_metric", threshold=0.8, comparison="gte")

        # Test passing case
        result = gate.evaluate(0.9)
        assert result["name"] == "test_metric"
        assert result["threshold"] == 0.8
        assert result["value"] == 0.9
        assert result["passed"] is True
        assert abs(result["delta"] - 0.1) < 1e-6

        # Test failing case
        result = gate.evaluate(0.7)
        assert result["passed"] is False
        assert abs(result["delta"] - (-0.1)) < 1e-6

        # Test edge case
        result = gate.evaluate(0.8)
        assert result["passed"] is True
        assert result["delta"] == 0.0

    def test_quality_gate_lte(self):
        """Test quality gate with less-than-or-equal comparison."""
        gate = QualityGate(name="error_rate", threshold=0.1, comparison="lte")

        # Test passing case
        result = gate.evaluate(0.05)
        assert result["passed"] is True
        assert result["delta"] == 0.05

        # Test failing case
        result = gate.evaluate(0.15)
        assert result["passed"] is False
        assert abs(result["delta"] - (-0.05)) < 1e-6

    def test_quality_gate_eq(self):
        """Test quality gate with equality comparison."""
        gate = QualityGate(name="exact_metric", threshold=1.0, comparison="eq")

        # Test passing case
        result = gate.evaluate(1.0)
        assert result["passed"] is True

        # Test failing case
        result = gate.evaluate(0.999)
        assert result["passed"] is False

    def test_quality_gate_invalid_comparison(self):
        """Test quality gate with invalid comparison operator."""
        gate = QualityGate(name="test", threshold=0.5, comparison="invalid")

        with pytest.raises(ValueError, match="Unknown comparison operator"):
            gate.evaluate(0.6)


class TestQualityGateEvaluator:
    """Test quality gate evaluator functionality."""

    def test_default_gates(self):
        """Test default quality gates for Mahabharata RAG system."""
        evaluator = QualityGateEvaluator()

        assert len(evaluator.gates) == 5

        # Check specific gates
        gate_names = [gate.name for gate in evaluator.gates]
        assert "faithfulness" in gate_names
        assert "answer_relevancy" in gate_names
        assert "context_precision" in gate_names
        assert "context_recall" in gate_names
        assert "answer_similarity" in gate_names

        # Check thresholds
        faithfulness_gate = next(g for g in evaluator.gates if g.name == "faithfulness")
        assert faithfulness_gate.threshold == 0.85
        assert faithfulness_gate.weight == 0.3

    def test_evaluate_all_passed(self):
        """Test evaluation when all gates pass."""
        evaluator = QualityGateEvaluator()

        scores = {
            "faithfulness": 0.9,
            "answer_relevancy": 0.85,
            "context_precision": 0.9,
            "context_recall": 0.85,
            "answer_similarity": 0.8,
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
            "faithfulness": 0.9,  # Pass
            "answer_relevancy": 0.75,  # Fail (threshold 0.8)
            "context_precision": 0.9,  # Pass (changed from 0.8 to 0.9)
            "context_recall": 0.7,  # Fail (threshold 0.8)
            "answer_similarity": 0.8,  # Pass
        }

        result = evaluator.evaluate(scores)

        # Overall score should be above 0.8 threshold
        assert result["overall_passed"] is True
        assert result["overall_score"] >= 0.8
        assert len(result["passed_gates"]) == 3
        assert len(result["failed_gates"]) == 2
        assert result["summary"]["pass_rate"] == 0.6

    def test_evaluate_strict_mode(self):
        """Test evaluation in strict mode (all gates must pass)."""
        evaluator = QualityGateEvaluator()

        scores = {
            "faithfulness": 0.9,
            "answer_relevancy": 0.75,  # Fail
            "context_precision": 0.9,
            "context_recall": 0.85,
            "answer_similarity": 0.8,
        }

        # Normal mode should pass (weighted score >= 0.8)
        result = evaluator.evaluate(scores, strict_mode=False)
        assert result["overall_passed"] is True

        # Strict mode should fail (one gate failed)
        result = evaluator.evaluate(scores, strict_mode=True)
        assert result["overall_passed"] is False

    def test_evaluate_missing_metrics(self):
        """Test evaluation with missing metrics."""
        evaluator = QualityGateEvaluator()

        scores = {
            "faithfulness": 0.9,
            "answer_relevancy": 0.85,
            # Missing context_precision, context_recall, answer_similarity
        }

        result = evaluator.evaluate(scores)

        assert result["overall_passed"] is False
        assert len(result["passed_gates"]) == 2
        assert len(result["failed_gates"]) == 3  # Missing metrics count as failed

    def test_custom_gates(self):
        """Test evaluator with custom quality gates."""
        custom_gates = [
            QualityGate(name="custom_metric1", threshold=0.9, weight=0.6),
            QualityGate(name="custom_metric2", threshold=0.7, weight=0.4),
        ]

        evaluator = QualityGateEvaluator(gates=custom_gates)

        scores = {
            "custom_metric1": 0.95,
            "custom_metric2": 0.65,
        }

        result = evaluator.evaluate(scores)

        assert len(result["passed_gates"]) == 1
        assert len(result["failed_gates"]) == 1
        # Calculate expected score: 0.6 * 1.0 + 0.4 * (0.65 / 0.7)
        expected_score = 0.6 * 1.0 + 0.4 * (0.65 / 0.7)
        assert abs(result["overall_score"] - expected_score) < 1e-6

    def test_improvement_suggestions(self):
        """Test improvement suggestions for failed gates."""
        evaluator = QualityGateEvaluator()

        # Create a result with failed gates
        result = {
            "passed_gates": [],
            "failed_gates": [
                {
                    "name": "faithfulness",
                    "threshold": 0.85,
                    "value": 0.7,
                    "delta": -0.15,
                },
                {
                    "name": "answer_relevancy",
                    "threshold": 0.8,
                    "value": 0.75,
                    "delta": -0.05,
                },
            ],
        }

        suggestions = evaluator.get_improvement_suggestions(result)

        assert len(suggestions) > 0
        assert any("faithfulness" in s.lower() for s in suggestions)
        assert any("relevancy" in s.lower() for s in suggestions)

    def test_generate_report(self):
        """Test report generation."""
        evaluator = QualityGateEvaluator()

        # Create a mock evaluation result
        result = {
            "overall_passed": False,
            "overall_score": 0.65,
            "passed_gates": [
                {
                    "name": "faithfulness",
                    "value": 0.9,
                    "threshold": 0.85,
                }
            ],
            "failed_gates": [
                {
                    "name": "answer_relevancy",
                    "value": 0.75,
                    "threshold": 0.8,
                    "delta": -0.05,
                }
            ],
            "summary": {
                "pass_rate": 0.5,
            },
        }

        report = evaluator.generate_report(result)

        assert "QUALITY GATE EVALUATION REPORT" in report
        assert "✗ FAILED" in report
        assert "Overall Score: 65.0%" in report
        assert "Pass Rate: 50.0%" in report
        assert "✓ faithfulness: 0.900" in report
        assert "✗ answer_relevancy: 0.750" in report


class TestQualityGateIntegration:
    """Integration tests for quality gates with evaluation pipeline."""

    def test_end_to_end_evaluation(self):
        """Test complete quality gate evaluation workflow."""
        # Create evaluator with default gates
        evaluator = QualityGateEvaluator()

        # Simulate RAG evaluation scores
        rag_scores = {
            "faithfulness": 0.87,
            "answer_relevancy": 0.82,
            "context_precision": 0.86,
            "context_recall": 0.81,
            "answer_similarity": 0.78,
        }

        # Evaluate quality gates
        result = evaluator.evaluate(rag_scores)

        # Verify results
        assert result["overall_passed"] is True
        assert result["overall_score"] >= 0.8

        # Generate report
        report = evaluator.generate_report(result)
        assert "✓ PASSED" in report

        # Get improvement suggestions (should be empty for passing result)
        suggestions = evaluator.get_improvement_suggestions(result)
        assert len(suggestions) == 0

    def test_evaluation_with_persistence(self):
        """Test saving and loading evaluation results."""
        evaluator = QualityGateEvaluator()

        scores = {
            "faithfulness": 0.9,
            "answer_relevancy": 0.85,
            "context_precision": 0.88,
            "context_recall": 0.82,
            "answer_similarity": 0.8,
        }

        result = evaluator.evaluate(scores)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(result, f)
            temp_path = Path(f.name)

        try:
            # Load and verify
            with open(temp_path, "r") as f:
                loaded_result = json.load(f)

            assert loaded_result["overall_passed"] is True
            assert loaded_result["overall_score"] == result["overall_score"]
            assert len(loaded_result["passed_gates"]) == 5

        finally:
            # Clean up
            temp_path.unlink()

    def test_batch_evaluation(self):
        """Test evaluating multiple score sets."""
        evaluator = QualityGateEvaluator()

        batch_scores = [
            {
                "faithfulness": 0.9,
                "answer_relevancy": 0.85,
                "context_precision": 0.88,
                "context_recall": 0.82,
                "answer_similarity": 0.8,
            },
            {
                "faithfulness": 0.75,
                "answer_relevancy": 0.7,
                "context_precision": 0.8,
                "context_recall": 0.75,
                "answer_similarity": 0.65,
            },
        ]

        results = []
        for scores in batch_scores:
            result = evaluator.evaluate(scores)
            results.append(result)

        # Verify first passed, second passed (both have high enough scores)
        assert results[0]["overall_passed"] is True
        assert results[1]["overall_passed"] is True

        # Check overall statistics
        passed_count = sum(1 for r in results if r["overall_passed"])
        assert passed_count == 2
        assert passed_count / len(results) == 1.0
