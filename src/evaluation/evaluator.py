"""Evaluation module for Vyasa Intelligence using Ragas."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (answer_relevancy, answer_similarity,
                           context_precision, context_recall, faithfulness)

from .quality_gates import QualityGateEvaluator

logger = logging.getLogger(__name__)


class MahabharataEvaluator:
    """Evaluates RAG system performance using Ragas metrics."""

    def __init__(
        self,
        golden_dataset_path: Path,
        output_dir: Path,
        metrics: Optional[List[str]] = None,
    ):
        """Initialize the evaluator.

        Args:
            golden_dataset_path: Path to golden dataset JSONL file
            output_dir: Directory to save evaluation results
            metrics: List of metrics to compute. Defaults to all available metrics.
        """
        self.golden_dataset_path = golden_dataset_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Default metrics for evaluation
        self.metrics = metrics or [
            "faithfulness",
            "answer_relevancy",
            "context_recall",
            "context_precision",
            "answer_similarity",
        ]

        # Initialize metric objects
        self.metric_objects = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_recall": context_recall,
            "context_precision": context_precision,
            "answer_similarity": answer_similarity,
        }

        # Quality gate thresholds
        self.quality_gates = {
            "faithfulness": 0.85,
            "answer_relevancy": 0.80,
            "context_recall": 0.80,
            "context_precision": 0.85,
            "answer_similarity": 0.75,
        }

        # Initialize quality gate evaluator
        self.quality_gate_evaluator = QualityGateEvaluator()

        # Load golden dataset
        self.golden_dataset = self._load_golden_dataset()

    def _load_golden_dataset(self) -> List[Dict[str, Any]]:
        """Load the golden dataset from JSONL file.

        Returns:
            List of Q&A pairs with contexts
        """
        dataset = []

        try:
            with open(self.golden_dataset_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        item = json.loads(line.strip())
                        dataset.append(item)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON on line {line_num}: {e}")

            logger.info(f"Loaded {len(dataset)} Q&A pairs from golden dataset")
            return dataset

        except FileNotFoundError:
            logger.error(f"Golden dataset not found: {self.golden_dataset_path}")
            raise

    def prepare_ragas_dataset(self, rag_results: List[Dict[str, Any]]) -> Dataset:
        """Prepare dataset for Ragas evaluation.

        Args:
            rag_results: List of RAG system results with 'question', 'answer',
                        and 'retrieved_contexts'

        Returns:
            HuggingFace Dataset for Ragas
        """
        # Map golden dataset to RAG results
        data_points = []

        for golden_item, rag_result in zip(self.golden_dataset, rag_results):
            # Extract ground truth contexts
            ground_truth_contexts = golden_item.get("contexts", [])

            # Prepare data point
            data_point = {
                "question": golden_item["question"],
                "answer": rag_result.get("answer", ""),
                "contexts": rag_result.get("retrieved_contexts", []),
                "ground_truth": golden_item["answer"],
                "ground_truth_contexts": ground_truth_contexts,
            }

            data_points.append(data_point)

        return Dataset.from_list(data_points)

    def evaluate(
        self,
        rag_results: List[Dict[str, Any]],
        run_name: Optional[str] = None,
        mock: bool = False,
        judge_llm: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run Ragas evaluation on RAG system results.

        Args:
            rag_results: List of RAG system results
            run_name: Optional name for this evaluation run
            mock: If True, use mock evaluation instead of Ragas
            judge_llm: Optional custom LLM for Ragas judge

        Returns:
            Dictionary with evaluation results
        """
        if mock:
            logger.info("Running mock evaluation...")
            return self._mock_evaluate(rag_results, run_name)

        logger.info("Starting Ragas evaluation...")

        # Prepare dataset
        dataset = self.prepare_ragas_dataset(rag_results)

        # Select metrics to compute
        selected_metrics = [
            self.metric_objects[metric]
            for metric in self.metrics
            if metric in self.metric_objects
        ]

        # Run evaluation
        result = evaluate(
            dataset=dataset,
            metrics=selected_metrics,
            raise_exceptions=False,
            llm=judge_llm,
        )

        # Convert to dictionary
        scores = result.to_pandas().to_dict("records")

        # Calculate aggregate statistics
        aggregate_scores = {}
        for metric in self.metrics:
            if metric in result:
                values = [
                    score[metric] for score in scores if not np.isnan(score[metric])
                ]
                if values:
                    aggregate_scores[metric] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "median": np.median(values),
                    }

        # Check quality gates using QualityGateEvaluator
        mean_scores = {
            metric: scores["mean"] for metric, scores in aggregate_scores.items()
        }
        quality_gate_results = self.quality_gate_evaluator.evaluate(mean_scores)

        # Generate quality gate report
        quality_gate_report = self.quality_gate_evaluator.generate_report(
            quality_gate_results
        )

        # Prepare final results
        evaluation_results = {
            "run_name": run_name or f"evaluation_{len(scores)}_samples",
            "timestamp": str(np.datetime64("now")),
            "num_samples": len(scores),
            "metrics": self.metrics,
            "aggregate_scores": aggregate_scores,
            "quality_gates": quality_gate_results,
            "quality_gate_report": quality_gate_report,
            "detailed_scores": scores,
            "passed_all_gates": quality_gate_results["overall_passed"],
        }

        # Save results
        self._save_results(evaluation_results)

        # Log summary
        self._log_summary(evaluation_results)

        return evaluation_results

    def _mock_evaluate(
        self, rag_results: List[Dict[str, Any]], run_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run mock evaluation without requiring API calls.

        Args:
            rag_results: List of RAG system results
            run_name: Optional name for this evaluation run

        Returns:
            Dictionary with mock evaluation results
        """
        import random

        # Generate mock scores for each metric
        mock_scores = []
        for i, rag_result in enumerate(rag_results):
            score = {}
            for metric in self.metrics:
                # Generate realistic mock scores
                if metric == "faithfulness":
                    score[metric] = random.uniform(0.82, 0.95)  # nosec B311
                elif metric == "answer_relevancy":
                    score[metric] = random.uniform(0.78, 0.92)  # nosec B311
                elif metric == "context_recall":
                    score[metric] = random.uniform(0.75, 0.90)  # nosec B311
                elif metric == "context_precision":
                    score[metric] = random.uniform(0.80, 0.95)  # nosec B311
                elif metric == "answer_similarity":
                    score[metric] = random.uniform(0.70, 0.88)  # nosec B311
            mock_scores.append(score)

        # Calculate aggregate statistics
        aggregate_scores = {}
        for metric in self.metrics:
            values = [score[metric] for score in mock_scores]
            aggregate_scores[metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }

        # Evaluate quality gates
        # Extract mean values for quality gate evaluation
        mean_scores = {
            metric: stats["mean"] for metric, stats in aggregate_scores.items()
        }
        quality_gate_results = self.quality_gate_evaluator.evaluate(mean_scores)
        quality_gate_report = self.quality_gate_evaluator.generate_report(
            quality_gate_results
        )

        # Prepare results
        run_name = run_name or f"mock_evaluation_{int(time.time())}"
        evaluation_results = {
            "run_name": run_name,
            "timestamp": time.time(),
            "num_questions": len(rag_results),
            "num_samples": len(rag_results),  # Add this for compatibility
            "metrics": self.metrics,
            "aggregate_scores": aggregate_scores,
            "quality_gate_report": quality_gate_report,
            "quality_gates": quality_gate_results,  # Add this for compatibility
            "detailed_scores": mock_scores,
            "passed_all_gates": quality_gate_results["overall_passed"],
        }

        # Save results
        self._save_results(evaluation_results)

        # Log summary
        self._log_summary(evaluation_results)

        return evaluation_results

    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results to file.

        Args:
            results: Evaluation results dictionary
        """
        # Save detailed results
        output_path = self.output_dir / f"{results['run_name']}_results.json"

        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if hasattr(obj, "item"):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj

        serializable_results = convert_numpy(results)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Evaluation results saved to {output_path}")

        # Save quality gate report
        report_path = self.output_dir / f"{results['run_name']}_quality_gates.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(results["quality_gate_report"])
        logger.info(f"Quality gate report saved to {report_path}")

        # Save summary CSV
        summary_path = self.output_dir / f"{results['run_name']}_summary.csv"
        import pandas as pd

        # Create summary dataframe
        summary_data = []
        for metric, scores in results["aggregate_scores"].items():
            # Find corresponding gate result
            gate_result = None
            for gate in results["quality_gates"]["all_results"]:
                if gate["name"] == metric:
                    gate_result = gate
                    break

            if gate_result:
                summary_data.append(
                    {
                        "metric": metric,
                        "mean": scores["mean"],
                        "std": scores["std"],
                        "min": scores["min"],
                        "max": scores["max"],
                        "threshold": gate_result["threshold"],
                        "passed": gate_result["passed"],
                    }
                )

        df = pd.DataFrame(summary_data)
        df.to_csv(summary_path, index=False)
        logger.info(f"Evaluation summary saved to {summary_path}")

    def _log_summary(self, results: Dict[str, Any]):
        """Log evaluation summary.

        Args:
            results: Evaluation results dictionary
        """
        logger.info("=" * 60)
        logger.info("RAGAS EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Run: {results['run_name']}")
        logger.info(f"Samples: {results['num_samples']}")
        logger.info(f"Timestamp: {results['timestamp']}")
        logger.info("-" * 60)

        for metric, scores in results["aggregate_scores"].items():
            # Find corresponding gate result
            gate_result = None
            for gate in results["quality_gates"]["all_results"]:
                if gate["name"] == metric:
                    gate_result = gate
                    break

            if gate_result:
                status = "✓ PASS" if gate_result["passed"] else "✗ FAIL"
                logger.info(
                    f"{metric:20s}: {scores['mean']:.3f} ± {scores['std']:.3f} "
                    f"(threshold: {gate_result['threshold']:.2f}) {status}"
                )

        logger.info("-" * 60)
        overall_status = "✓ PASSED" if results["passed_all_gates"] else "✗ FAILED"
        logger.info(f"Overall Quality Gates: {overall_status}")
        logger.info(f"Overall Score: {results['quality_gates']['overall_score']:.2%}")
        logger.info("=" * 60)

    def compare_evaluations(
        self, evaluation_files: List[str], comparison_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compare multiple evaluation runs.

        Args:
            evaluation_files: List of evaluation result filenames
            comparison_name: Optional name for comparison

        Returns:
            Dictionary with comparison results
        """
        logger.info("Comparing evaluation runs...")

        # Load evaluation results
        results = []
        for filename in evaluation_files:
            filepath = self.output_dir / filename
            if filepath.exists():
                with open(filepath, "r", encoding="utf-8") as f:
                    results.append(json.load(f))
            else:
                logger.warning(f"Evaluation file not found: {filepath}")

        if len(results) < 2:
            logger.error("Need at least 2 evaluation runs for comparison")
            return {}

        # Prepare comparison data
        comparison = {
            "comparison_name": comparison_name or f"comparison_{len(results)}_runs",
            "timestamp": str(np.datetime64("now")),
            "runs": [],
            "metric_comparison": {},
        }

        # Collect run data
        for result in results:
            run_info = {
                "name": result["run_name"],
                "timestamp": result["timestamp"],
                "num_samples": result["num_samples"],
                "passed_gates": result["passed_all_gates"],
            }

            # Add metric scores
            for metric, scores in result["aggregate_scores"].items():
                run_info[f"{metric}_mean"] = scores["mean"]
                run_info[f"{metric}_std"] = scores["std"]

            comparison["runs"].append(run_info)

        # Compare metrics across runs
        all_metrics = set()
        for result in results:
            all_metrics.update(result["aggregate_scores"].keys())

        for metric in all_metrics:
            comparison["metric_comparison"][metric] = {
                "runs": [],
                "best_run": None,
                "best_score": -float("inf"),
                "worst_run": None,
                "worst_score": float("inf"),
                "mean_score": 0,
                "std_score": 0,
            }

            scores = []
            for result in results:
                if metric in result["aggregate_scores"]:
                    score = result["aggregate_scores"][metric]["mean"]
                    scores.append(score)
                    # Find passed gate status
                    passed_gate = None
                    for gate in result["quality_gates"]["all_results"]:
                        if gate["name"] == metric:
                            passed_gate = gate
                            break

                    passed_status = passed_gate["passed"] if passed_gate else False
                    comparison["metric_comparison"][metric]["runs"].append(
                        {
                            "run": result["run_name"],
                            "score": score,
                            "passed_gate": passed_status,
                        }
                    )

                    # Track best/worst
                    if score > comparison["metric_comparison"][metric]["best_score"]:
                        comparison["metric_comparison"][metric]["best_score"] = score
                        comparison["metric_comparison"][metric]["best_run"] = result[
                            "run_name"
                        ]

                    if score < comparison["metric_comparison"][metric]["worst_score"]:
                        comparison["metric_comparison"][metric]["worst_score"] = score
                        comparison["metric_comparison"][metric]["worst_run"] = result[
                            "run_name"
                        ]

            # Calculate statistics
            if scores:
                comparison["metric_comparison"][metric]["mean_score"] = np.mean(scores)
                comparison["metric_comparison"][metric]["std_score"] = np.std(scores)

        # Save comparison
        comparison_path = self.output_dir / f"{comparison['comparison_name']}.json"
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)

        logger.info(f"Comparison saved to {comparison_path}")

        # Log comparison summary
        self._log_comparison_summary(comparison)

        return comparison

    def _log_comparison_summary(self, comparison: Dict[str, Any]):
        """Log comparison summary.

        Args:
            comparison: Comparison results dictionary
        """
        logger.info("=" * 60)
        logger.info("EVALUATION COMPARISON SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Comparison: {comparison['comparison_name']}")
        logger.info(f"Runs: {len(comparison['runs'])}")
        logger.info("-" * 60)

        for metric, data in comparison["metric_comparison"].items():
            logger.info(f"\n{metric.upper()}:")
            logger.info(f"  Best: {data['best_run']} ({data['best_score']:.3f})")
            logger.info(f"  Worst: {data['worst_run']} ({data['worst_score']:.3f})")
            logger.info(f"  Mean: {data['mean_score']:.3f} ± {data['std_score']:.3f}")

        logger.info("=" * 60)
