"""Quality gates for RAG system evaluation.

This module implements quality gates with specific thresholds for different metrics.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class QualityGate:
    """Represents a single quality gate with threshold and evaluation logic."""

    def __init__(
        self,
        name: str,
        threshold: float,
        comparison: str = "gte",
        description: Optional[str] = None,
        weight: float = 1.0,
    ):
        """Initialize a quality gate.

        Args:
            name: Name of the metric
            threshold: Threshold value
            comparison: Comparison operator ('gte', 'lte', 'eq')
            description: Optional description
            weight: Weight in overall score (0-1)
        """
        self.name = name
        self.threshold = threshold
        self.comparison = comparison
        self.description = description or f"{name} {comparison} {threshold}"
        self.weight = weight

    def evaluate(self, value: float) -> Dict[str, Any]:
        """Evaluate if value passes the quality gate.

        Args:
            value: Metric value to evaluate

        Returns:
            Dictionary with evaluation result
        """
        if self.comparison == "gte":
            passed = value >= self.threshold
            delta = value - self.threshold
        elif self.comparison == "lte":
            passed = value <= self.threshold
            delta = self.threshold - value
        elif self.comparison == "eq":
            passed = abs(value - self.threshold) < 1e-6
            delta = value - self.threshold
        else:
            raise ValueError(f"Unknown comparison operator: {self.comparison}")

        return {
            "name": self.name,
            "threshold": self.threshold,
            "value": value,
            "passed": passed,
            "delta": delta,
            "weight": self.weight,
            "description": self.description,
        }


class QualityGateEvaluator:
    """Evaluates multiple quality gates and provides overall assessment."""

    def __init__(self, gates: Optional[List[QualityGate]] = None):
        """Initialize quality gate evaluator.

        Args:
            gates: List of quality gates. Defaults to Mahabharata-specific gates.
        """
        self.gates = gates or self._get_default_gates()

    def _get_default_gates(self) -> List[QualityGate]:
        """Get default quality gates for Mahabharata RAG system.

        Returns:
            List of default quality gates
        """
        return [
            QualityGate(
                name="faithfulness",
                threshold=0.85,
                comparison="gte",
                description="Answer must be faithful to retrieved contexts",
                weight=0.3,
            ),
            QualityGate(
                name="answer_relevancy",
                threshold=0.80,
                comparison="gte",
                description="Answer must be relevant to the question",
                weight=0.25,
            ),
            QualityGate(
                name="context_precision",
                threshold=0.85,
                comparison="gte",
                description="Retrieved contexts must be precise and relevant",
                weight=0.2,
            ),
            QualityGate(
                name="context_recall",
                threshold=0.80,
                comparison="gte",
                description="Retrieved contexts must cover all relevant information",
                weight=0.15,
            ),
            QualityGate(
                name="answer_similarity",
                threshold=0.75,
                comparison="gte",
                description="Answer should be similar to expected answer",
                weight=0.1,
            ),
        ]

    def evaluate(
        self, scores: Dict[str, float], strict_mode: bool = False
    ) -> Dict[str, Any]:
        """Evaluate all quality gates against provided scores.

        Args:
            scores: Dictionary of metric scores
            strict_mode: If True, all gates must pass. If False, weighted score is used.

        Returns:
            Dictionary with evaluation results
        """
        results = []
        passed_gates = []
        failed_gates = []
        total_weight = 0
        weighted_score = 0

        for gate in self.gates:
            if gate.name in scores:
                result = gate.evaluate(scores[gate.name])
                results.append(result)

                total_weight += gate.weight
                if result["passed"]:
                    passed_gates.append(result)
                    # Full weight for passed gates
                    weighted_score += gate.weight
                else:
                    failed_gates.append(result)
                    # Proportional weight based on how close to threshold
                    if not strict_mode:
                        proportion = max(0, 1 + result["delta"] / gate.threshold)
                        weighted_score += gate.weight * proportion
            else:
                # Missing metric - treat as failed
                logger.warning(f"Missing score for metric: {gate.name}")
                result = {
                    "name": gate.name,
                    "threshold": gate.threshold,
                    "value": None,
                    "passed": False,
                    "delta": None,
                    "weight": gate.weight,
                    "description": gate.description,
                    "error": "Metric not evaluated",
                }
                results.append(result)
                failed_gates.append(result)
                total_weight += gate.weight

        # Calculate overall score
        overall_score = weighted_score / total_weight if total_weight > 0 else 0

        # Determine overall pass/fail
        if strict_mode:
            overall_passed = len(failed_gates) == 0
        else:
            overall_passed = overall_score >= 0.8  # 80% of total weight

        return {
            "overall_passed": overall_passed,
            "overall_score": overall_score,
            "total_weight": total_weight,
            "passed_gates": passed_gates,
            "failed_gates": failed_gates,
            "all_results": results,
            "summary": {
                "total_gates": len(self.gates),
                "passed_count": len(passed_gates),
                "failed_count": len(failed_gates),
                "pass_rate": len(passed_gates) / len(self.gates) if self.gates else 0,
            },
        }

    def get_improvement_suggestions(
        self, evaluation_result: Dict[str, Any]
    ) -> List[str]:
        """Get suggestions for improving failed quality gates.

        Args:
            evaluation_result: Result from evaluate() method

        Returns:
            List of improvement suggestions
        """
        suggestions = []

        for failed_gate in evaluation_result["failed_gates"]:
            name = failed_gate["name"]
            delta = failed_gate.get("delta", 0)

            # Skip if delta is None (missing metric)
            if delta is None:
                continue

            if name == "faithfulness":
                if delta < -0.1:
                    suggestions.append(
                        "Faithfulness is significantly below threshold. "
                        "Consider improving answer generation to stick closer to retrieved contexts."
                    )
                else:
                    suggestions.append(
                        "Slightly improve faithfulness by reducing hallucinations "
                        "and ensuring all claims are supported by contexts."
                    )

            elif name == "answer_relevancy":
                if delta < -0.1:
                    suggestions.append(
                        "Answer relevancy is low. Focus on directly addressing the question "
                        "and avoiding irrelevant information."
                    )
                else:
                    suggestions.append(
                        "Improve answer relevancy by ensuring answers directly address "
                        "the user's question."
                    )

            elif name == "context_precision":
                if delta < -0.1:
                    suggestions.append(
                        "Context precision needs improvement. Review retrieval strategy "
                        "to return more relevant documents."
                    )
                else:
                    suggestions.append(
                        "Fine-tune retrieval parameters to improve context precision."
                    )

            elif name == "context_recall":
                if delta < -0.1:
                    suggestions.append(
                        "Context recall is low. Consider increasing top_k or improving "
                        "query understanding."
                    )
                else:
                    suggestions.append(
                        "Slightly increase retrieval coverage to improve context recall."
                    )

            elif name == "answer_similarity":
                if delta < -0.1:
                    suggestions.append(
                        "Answer similarity is low. Review prompt templates and "
                        "answer generation approach."
                    )
                else:
                    suggestions.append(
                        "Minor adjustments to answer generation may improve similarity."
                    )

        # General suggestions if multiple gates failed
        failed_count = len(evaluation_result["failed_gates"])
        if failed_count > 2:
            suggestions.append(
                "Multiple quality gates failed. Consider reviewing the entire RAG pipeline, "
                "including retrieval quality, prompt engineering, and LLM configuration."
            )

        return suggestions

    def generate_report(self, evaluation_result: Dict[str, Any]) -> str:
        """Generate a human-readable quality gate report.

        Args:
            evaluation_result: Result from evaluate() method

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("QUALITY GATE EVALUATION REPORT")
        lines.append("=" * 60)

        # Overall status
        status = "✓ PASSED" if evaluation_result["overall_passed"] else "✗ FAILED"
        lines.append(f"Overall Status: {status}")
        lines.append(f"Overall Score: {evaluation_result['overall_score']:.1%}")

        # Add summary if available
        if "summary" in evaluation_result:
            lines.append(f"Pass Rate: {evaluation_result['summary']['pass_rate']:.1%}")
        else:
            # Calculate pass rate from gates
            total_gates = len(evaluation_result.get("all_results", []))
            passed_count = len(evaluation_result.get("passed_gates", []))
            pass_rate = passed_count / total_gates if total_gates > 0 else 0
            lines.append(f"Pass Rate: {pass_rate:.1%}")
        lines.append("")

        # Passed gates
        if evaluation_result["passed_gates"]:
            lines.append("PASSED GATES:")
            for gate in evaluation_result["passed_gates"]:
                lines.append(
                    f"  ✓ {gate['name']}: {gate['value']:.3f} "
                    f"(threshold: {gate['threshold']:.2f})"
                )
            lines.append("")

        # Failed gates
        if evaluation_result["failed_gates"]:
            lines.append("FAILED GATES:")
            for gate in evaluation_result["failed_gates"]:
                if "error" in gate:
                    lines.append(f"  ✗ {gate['name']}: ERROR - {gate['error']}")
                else:
                    lines.append(
                        f"  ✗ {gate['name']}: {gate['value']:.3f} "
                        f"(threshold: {gate['threshold']:.2f}, "
                        f"delta: {gate['delta']:.3f})"
                    )
            lines.append("")

        # Improvement suggestions
        suggestions = self.get_improvement_suggestions(evaluation_result)
        if suggestions:
            lines.append("IMPROVEMENT SUGGESTIONS:")
            for i, suggestion in enumerate(suggestions, 1):
                lines.append(f"  {i}. {suggestion}")
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)
