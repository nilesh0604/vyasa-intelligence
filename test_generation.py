#!/usr/bin/env python3
"""Test script for the generation pipeline.

This script tests the end-to-end RAG pipeline with local Ollama.
"""

import os
import sys

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.pipeline import RAGPipeline

# Load environment variables
load_dotenv()


def test_pipeline():
    """Test the RAG pipeline with sample questions."""

    print("🚀 Initializing Vyasa Intelligence RAG Pipeline...")
    print("=" * 60)

    # Initialize pipeline
    pipeline = RAGPipeline(
        llm_provider="ollama",
        llm_model="llama3.2",
        enable_cache=True,
        enable_guardrails=True,
    )

    # Test questions
    test_questions = [
        {
            "question": "Who is Arjuna and what is his role in the Mahabharata?",
            "role": "public",
        },
        {
            "question": "Explain the concept of dharma as presented in the Bhagavad Gita",
            "role": "scholar",
        },
        {
            "question": "What weapons did Arjuna receive from the gods?",
            "role": "public",
        },
        {
            "question": "Describe the Kurukshetra war and its significance",
            "role": "scholar",
        },
    ]

    print("✓ Pipeline initialized successfully")
    print(f"✓ Retrieval index: {pipeline.retriever.get_statistics()}")
    print("=" * 60)

    # Process each question
    for i, test in enumerate(test_questions, 1):
        print(f"\n📝 Question {i}: {test['question']}")
        print(f"   Role: {test['role']}")
        print("-" * 40)

        # Query the pipeline
        result = pipeline.query(
            question=test["question"],
            user_role=test["role"],
            top_k=5,
        )

        # Display results
        print(f"\n💡 Answer ({result['total_time']:.2f}s):")
        print(f"   {result['answer']}")

        if result["citations"]:
            print("\n📚 Citations:")
            for citation in result["citations"]:
                print(f"   {citation}")

        if result["sources"]:
            print("\n📖 Sources:")
            for source in result["sources"]:
                print(f"   {source}")

        # Display metadata
        metadata = result["metadata"]
        print("\n📊 Metadata:")
        print(f"   Context used: {result['context_used']}")
        print(f"   Retrieval time: {result['retrieval_time']:.2f}s")
        print(f"   Generation time: {result['generation_time']:.2f}s")
        print(f"   Cache hit: {result['cache_hit']}")
        print(f"   Guardrails passed: {result['guardrails_passed']}")

        if "tokens_generated" in metadata:
            print(f"   Tokens generated: {metadata['tokens_generated']}")

        if "warnings" in metadata:
            print(f"   Warnings: {metadata['warnings']}")

        print("\n" + "=" * 60)

    # Display pipeline statistics
    print("\n📈 Pipeline Statistics:")
    stats = pipeline.get_statistics()
    print(f"   Retrieval index: {stats['retrieval']}")
    if stats["cache"]:
        print(f"   Cache: {stats['cache']}")
    print(f"   Guardrails: {stats['guardrails']}")


def test_llm_only():
    """Test LLM connection without retrieval."""
    print("\n🔧 Testing LLM connection only...")

    from src.generation.llm_factory import get_llm

    try:
        llm = get_llm(provider="ollama", model="llama3.2")
        response = llm.invoke("What is the Mahabharata?")
        print(f"✓ LLM response: {response[:100]}...")
    except Exception as e:
        print(f"✗ LLM test failed: {e}")
        return False

    return True


if __name__ == "__main__":
    # Check if Ollama is running
    print("🔍 Checking Ollama connection...")
    if not test_llm_only():
        print("\n❌ Please ensure Ollama is running:")
        print("   ollama serve")
        print("   ollama pull llama3.2")
        sys.exit(1)

    # Run full pipeline test
    try:
        test_pipeline()
        print("\n✅ All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
