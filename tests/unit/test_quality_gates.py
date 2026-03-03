"""Unit tests for quality gates module."""

import pytest

from src.evaluation.quality_gates import QualityGate, QualityGateEvaluator


class TestQualityGateUnit:
    """Unit tests for QualityGate class."""

    def test_quality_gate_initialization(self):
        """Test QualityGate initialization with different parameters."""
        # Basic gate
        gate = QualityGate(name="test", threshold=0.5)
        assert gate.name == "test"
        assert gate.threshold == 0.5
        assert gate.comparison == "gte"
        assert gate.weight == 1.0
        assert gate.description == "test gte 0.5"

        # Full parameters
        gate = QualityGate(
            name="complex",
            threshold=0.8,
            comparison="lte",
            description="Custom description",
            weight=0.5,
        )
        assert gate.name == "complex"
        assert gate.threshold == 0.8
        assert gate.comparison == "lte"
        assert gate.weight == 0.5
        assert gate.description == "Custom description"

    def test_quality_gate_invalid_comparison(self):
        """Test QualityGate with invalid comparison operator."""
        gate = QualityGate(name="test", threshold=0.5, comparison="invalid")

        with pytest.raises(ValueError, match="Unknown comparison operator"):
            gate.evaluate(0.6)

    def test_quality_gate_edge_cases(self):
        """Test QualityGate with edge cases."""
        gate = QualityGate(name="test", threshold=0.0)

        # Test with zero threshold
        result = gate.evaluate(0.0)
        assert result["passed"] is True
        assert result["delta"] == 0.0

        # Test with negative values
        result = gate.evaluate(-0.1)
        assert result["passed"] is False

        # Test with very large values
        result = gate.evaluate(1000.0)
        assert result["passed"] is True
        assert result["delta"] == 1000.0


class TestQualityGateEvaluatorUnit:
    """Unit tests for QualityGateEvaluator class."""

    def test_evaluator_with_custom_gates(self):
        """Test evaluator with custom quality gates."""
        custom_gates = [
            QualityGate(name="metric1", threshold=0.5, weight=0.6),
            QualityGate(name="metric2", threshold=0.8, weight=0.4),
        ]

        evaluator = QualityGateEvaluator(gates=custom_gates)

        assert len(evaluator.gates) == 2
        assert evaluator.gates[0].weight == 0.6
        assert evaluator.gates[1].weight == 0.4

    def test_evaluate_missing_metrics(self):
        """Test evaluation with missing metrics."""
        evaluator = QualityGateEvaluator()

        # Only provide one metric
        scores = {"faithfulness": 0.9}

        result = evaluator.evaluate(scores)

        # Should have one passed gate and four failed (missing) gates
        assert len(result["passed_gates"]) == 1
        assert len(result["failed_gates"]) == 4
        assert result["overall_passed"] is False
        assert result["overall_score"] < 1.0

        # Check that missing gates have error information
        missing_gates = [g for g in result["failed_gates"] if "error" in g]
        assert len(missing_gates) == 4

    def test_evaluate_zero_weight_total(self):
        """Test evaluation when total weight is zero."""
        # Create evaluator with zero-weight gates
        zero_weight_gates = [
            QualityGate(name="metric1", threshold=0.5, weight=0.0),
            QualityGate(name="metric2", threshold=0.5, weight=0.0),
        ]

        evaluator = QualityGateEvaluator(gates=zero_weight_gates)
        scores = {"metric1": 0.6, "metric2": 0.4}

        result = evaluator.evaluate(scores)

        # Should handle zero weight gracefully
        assert result["overall_score"] == 0
        assert result["total_weight"] == 0

    def test_evaluate_all_negative_deltas(self):
        """Test evaluation when all metrics fail significantly."""
        evaluator = QualityGateEvaluator()

        scores = {
            "faithfulness": 0.5,  # 0.35 below threshold
            "answer_relevancy": 0.5,  # 0.3 below threshold
            "context_precision": 0.5,  # 0.35 below threshold
            "context_recall": 0.5,  # 0.3 below threshold
            "answer_similarity": 0.5,  # 0.25 below threshold
        }

        result = evaluator.evaluate(scores, strict_mode=False)

        # Should fail overall but score might be above 0.5 due to weights
        assert result["overall_passed"] is False
        # The exact score depends on weights, just check it's not 1.0
        assert result["overall_score"] < 1.0

    def test_improvement_suggestions_edge_cases(self):
        """Test improvement suggestions for various edge cases."""
        evaluator = QualityGateEvaluator()

        # Test with no failed gates
        result = {"failed_gates": []}
        suggestions = evaluator.get_improvement_suggestions(result)
        assert len(suggestions) == 0

        # Test with unknown metric name
        result = {"failed_gates": [{"name": "unknown_metric", "delta": -0.1}]}
        suggestions = evaluator.get_improvement_suggestions(result)
        # Should not crash, just return empty or general suggestions
        assert isinstance(suggestions, list)

    def test_report_formatting(self):
        """Test that the generated report has proper formatting."""
        evaluator = QualityGateEvaluator()

        # Create a result with various scenarios
        result = {
            "overall_passed": False,
            "overall_score": 0.75,
            "summary": {
                "pass_rate": 0.6,
                "total_gates": 5,
                "passed_count": 3,
                "failed_count": 2,
            },
            "passed_gates": [{"name": "faithfulness", "value": 0.9, "threshold": 0.85}],
            "failed_gates": [
                {
                    "name": "answer_relevancy",
                    "value": 0.75,
                    "threshold": 0.8,
                    "delta": -0.05,
                },
                {
                    "name": "context_recall",
                    "value": 0.7,
                    "threshold": 0.8,
                    "delta": -0.1,
                },
            ],
        }

        report = evaluator.generate_report(result)

        # Check report structure
        lines = report.split("\n")
        assert any("QUALITY GATE EVALUATION REPORT" in line for line in lines)
        assert any("Overall Status: ✗ FAILED" in line for line in lines)
        assert any("Overall Score: 75.0%" in line for line in lines)
        assert any("Pass Rate: 60.0%" in line for line in lines)
        assert any("PASSED GATES:" in line for line in lines)
        assert any("FAILED GATES:" in line for line in lines)
        assert any("IMPROVEMENT SUGGESTIONS:" in line for line in lines)

    def test_weighted_score_calculation(self):
        """Test weighted score calculation accuracy."""
        # Custom gates with different weights
        gates = [
            QualityGate(name="high_weight", threshold=0.5, weight=0.7),
            QualityGate(name="low_weight", threshold=0.5, weight=0.3),
        ]

        evaluator = QualityGateEvaluator(gates)

        # First metric passes, second fails at 50% of threshold
        scores = {
            "high_weight": 0.6,  # Pass - full weight
            "low_weight": 0.25,  # Fail - 50% of threshold
        }

        result = evaluator.evaluate(scores, strict_mode=False)

        # Expected: 0.7 * 1.0 + 0.3 * 0.5 = 0.85
        expected_score = 0.85
        assert abs(result["overall_score"] - expected_score) < 0.01


if __name__ == "__main__":
    pytest.main([__file__])
