"""Script to evaluate RAG system performance using the golden dataset."""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.evaluator import MahabharataEvaluator
from generation.answer_generator import AnswerGenerator
from retrieval.pipeline import RetrievalPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_mock_rag_results(golden_dataset: list) -> list:
    """Create mock RAG results for testing evaluation pipeline.

    Args:
        golden_dataset: List of golden dataset items

    Returns:
        List of mock RAG results
    """
    results = []

    for item in golden_dataset:
        # Create a mock answer based on the question
        question = item["question"]
        contexts = item["contexts"]

        # Simple mock answer generation
        if "Arjuna" in question:
            mock_answer = "Arjuna was one of the Pandava brothers and a skilled archer."
        elif "dharma" in question.lower():
            mock_answer = (
                "Dharma is a concept of duty and righteousness in Hindu philosophy."
            )
        elif "war" in question.lower():
            mock_answer = (
                "The Kurukshetra war was a great battle fought for eighteen days."
            )
        else:
            # Use first context as mock answer
            mock_answer = contexts[0] if contexts else "This is a mock answer."

        results.append(
            {
                "question": question,
                "answer": mock_answer,
                "retrieved_contexts": contexts,
                "sources": [f"Source {i+1}" for i in range(len(contexts))],
            }
        )

    return results


def run_real_rag_pipeline(
    golden_dataset: list,
    chroma_dir: Path,
    bm25_path: Path,
    top_k: int = 5,
) -> list:
    """Run actual RAG pipeline on golden dataset questions.

    Args:
        golden_dataset: List of golden dataset items
        chroma_dir: Path to ChromaDB index
        bm25_path: Path to BM25 index
        top_k: Number of documents to retrieve

    Returns:
        List of RAG results
    """
    logger.info("Initializing RAG pipeline...")

    # Initialize retrieval pipeline
    retrieval_pipeline = RetrievalPipeline(
        chroma_dir=chroma_dir,
        bm25_path=bm25_path,
        enable_reranking=True,
        enable_query_classification=True,
    )

    # Initialize answer generator
    answer_generator = AnswerGenerator(
        model_name="BAAI/bge-base-en-v1.5",  # This would be an LLM in production
    )

    results = []

    for i, item in enumerate(golden_dataset, 1):
        logger.info(f"Processing question {i}/{len(golden_dataset)}")

        question = item["question"]

        # Retrieve relevant documents
        retrieval_result = retrieval_pipeline.retrieve(question, top_k=top_k)

        # Extract contexts
        contexts = [doc["content"] for doc in retrieval_result["results"]]

        # Generate answer (mock for now)
        # In production, this would use an LLM
        answer = answer_generator.generate_answer(
            question=question,
            contexts=contexts,
        )

        results.append(
            {
                "question": question,
                "answer": answer,
                "retrieved_contexts": contexts,
                "sources": [
                    doc.get("metadata", {}).get("source", f"Doc {i}")
                    for i, doc in enumerate(retrieval_result["results"])
                ],
            }
        )

    return results


def main():
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate RAG system performance")
    parser.add_argument(
        "--golden-dataset",
        type=Path,
        default=Path("data/processed/golden_dataset.jsonl"),
        help="Path to golden dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/evaluation"),
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=Path("data/chroma"),
        help="ChromaDB index directory",
    )
    parser.add_argument(
        "--bm25-path",
        type=Path,
        default=Path("data/bm25_index.pkl"),
        help="BM25 index path",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock RAG results instead of running pipeline",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Name for this evaluation run",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "faithfulness",
            "answer_relevancy",
            "context_recall",
            "context_precision",
            "answer_similarity",
        ],
        help="Metrics to evaluate",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize evaluator
    evaluator = MahabharataEvaluator(
        golden_dataset_path=args.golden_dataset,
        output_dir=args.output_dir,
        metrics=args.metrics,
    )

    # Load golden dataset
    golden_dataset = evaluator.golden_dataset
    logger.info(f"Loaded {len(golden_dataset)} questions from golden dataset")

    # Get RAG results
    if args.mock:
        logger.info("Using mock RAG results...")
        rag_results = create_mock_rag_results(golden_dataset)
    else:
        logger.info("Running RAG pipeline...")
        rag_results = run_real_rag_pipeline(
            golden_dataset=golden_dataset,
            chroma_dir=args.chroma_dir,
            bm25_path=args.bm25_path,
            top_k=args.top_k,
        )

    # Run evaluation
    evaluation_results = evaluator.evaluate(
        rag_results=rag_results,
        run_name=args.run_name,
        mock=args.mock,
    )

    # Print final status
    if evaluation_results["passed_all_gates"]:
        logger.info("✓ All quality gates passed!")
        sys.exit(0)
    else:
        logger.error("✗ Some quality gates failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
