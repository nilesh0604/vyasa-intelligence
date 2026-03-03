"""Run evaluation with Groq as LLM provider."""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv(project_root / ".env")

# Set environment variables for Groq
os.environ["LLM_PROVIDER"] = "groq"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from src.evaluation.evaluator import MahabharataEvaluator
from src.pipeline import RAGPipeline


def main():
    """Run evaluation with Groq."""
    # Verify Groq API key is set
    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY environment variable is required")
        sys.exit(1)

    # Initialize pipeline with Groq
    pipeline = RAGPipeline()

    # Load golden dataset
    golden_dataset_path = Path("data/processed/golden_dataset.jsonl")
    if not golden_dataset_path.exists():
        print(f"ERROR: Golden dataset not found at {golden_dataset_path}")
        print("Please create the golden dataset first")
        sys.exit(1)

    # Load questions from golden dataset

    questions = []
    with open(golden_dataset_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            questions.append(item["question"])

    print(f"Running {len(questions)} questions through pipeline with Groq...")

    # Run pipeline
    results = []
    for i, question in enumerate(questions, 1):
        print(f"Processing question {i}/{len(questions)}: {question[:50]}...")
        result = pipeline.query(question, user_role="public")

        # Extract contexts for evaluation
        retrieved_contexts = [
            doc["page_content"] for doc in result.get("retrieved_docs", [])
        ]

        results.append(
            {
                "question": question,
                "answer": result.get("answer", ""),
                "retrieved_contexts": retrieved_contexts,
            }
        )

    # Initialize evaluator
    output_dir = Path("data/evaluation")
    evaluator = MahabharataEvaluator(
        golden_dataset_path=golden_dataset_path,
        output_dir=output_dir,
    )

    # Run evaluation with Groq as judge
    print("\nRunning Ragas evaluation with Groq as judge...")
    evaluation_results = evaluator.evaluate(
        rag_results=results,
        run_name="m7_groq_evaluation",
    )

    # Print summary
    print("\n" + "=" * 60)
    print("M7 GROQ EVALUATION SUMMARY")
    print("=" * 60)

    for metric, scores in evaluation_results["aggregate_scores"].items():
        # Find corresponding gate result
        gate_result = None
        for gate in evaluation_results["quality_gates"]["all_results"]:
            if gate["name"] == metric:
                gate_result = gate
                break

        if gate_result:
            status = "✓ PASS" if gate_result["passed"] else "✗ FAIL"
            print(
                f"{metric:20s}: {scores['mean']:.3f} ± {scores['std']:.3f} "
                f"(threshold: {gate_result['threshold']:.2f}) {status}"
            )

    print("-" * 60)
    overall_status = (
        "✓ PASSED" if evaluation_results["passed_all_gates"] else "✗ FAILED"
    )
    print(f"Overall Quality Gates: {overall_status}")
    print(f"Overall Score: {evaluation_results['quality_gates']['overall_score']:.2%}")
    print("=" * 60)

    # Save results
    print(f"\nResults saved to: {output_dir}")
    print("- Detailed results: m7_groq_evaluation_results.json")
    print("- Quality gates: m7_groq_evaluation_quality_gates.txt")
    print("- Summary: m7_groq_evaluation_summary.csv")


if __name__ == "__main__":
    main()
