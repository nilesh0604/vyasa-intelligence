"""Simple evaluation script to test Groq performance."""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv(project_root / ".env")

# Set environment variables for Groq early
os.environ["LLM_PROVIDER"] = "groq"

# Disable LangSmith tracing to avoid API issues
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from src.pipeline import RAGPipeline


def main():
    """Run simple evaluation with Groq."""

    # Verify Groq API key is set
    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY environment variable is required")
        sys.exit(1)

    # Initialize pipeline with Groq
    print("Initializing pipeline with Groq...")
    pipeline = RAGPipeline(enable_cache=False, enable_tracing=False)

    # Test questions
    test_questions = [
        "Who is Arjuna?",
        "What is the Bhagavad Gita?",
        "Who fought in the Kurukshetra war?",
        "What are the Pandavas' names?",
        "Who is Drona's most famous student?",
    ]

    print(f"\nRunning {len(test_questions)} test questions with Groq...\n")

    results = []
    total_time = 0

    for i, question in enumerate(test_questions, 1):
        print(f"Question {i}: {question}")

        start_time = time.time()
        result = pipeline.query(question, user_role="public")
        end_time = time.time()

        answer_time = end_time - start_time
        total_time += answer_time

        print(f"Answer: {result['answer'][:200]}...")
        print(f"Citations: {len(result.get('citations', []))}")
        print(f"Generation time: {answer_time:.2f}s")
        print("-" * 60)

        results.append(
            {
                "question": question,
                "answer": result["answer"],
                "citations": result.get("citations", []),
                "sources": result.get("sources", []),
                "time": answer_time,
            }
        )

    # Summary
    print("\n" + "=" * 60)
    print("M7 GROQ TEST SUMMARY")
    print("=" * 60)
    print(f"Total questions: {len(test_questions)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per question: {total_time/len(test_questions):.2f}s")
    print(
        f"Average citations per answer: {sum(len(r['citations']) for r in results)/len(results):.1f}"
    )
    print("=" * 60)

    # Save results
    output_dir = Path("data/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    import json

    output_file = output_dir / "m7_groq_test_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
