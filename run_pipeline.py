#!/usr/bin/env python3
"""Command-line interface for the Vyasa Intelligence RAG pipeline."""

import argparse
import os
import sys

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.pipeline import RAGPipeline


def main():
    """Run the RAG pipeline from command line."""
    parser = argparse.ArgumentParser(
        description="Vyasa Intelligence - Mahabharata RAG System"
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Question about the Mahabharata",
    )
    parser.add_argument(
        "--role",
        choices=["public", "scholar", "admin"],
        default="public",
        help="User role for response customization",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve",
    )
    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=0.5,
        help="Weight for BM25 retrieval (0.0 to 1.0)",
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=0.5,
        help="Weight for dense retrieval (0.0 to 1.0)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable response caching",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show pipeline statistics",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Initialize pipeline
    print("🚀 Initializing Vyasa Intelligence...")
    pipeline = RAGPipeline(
        enable_cache=not args.no_cache,
    )

    if args.stats:
        # Show statistics and exit
        stats = pipeline.get_statistics()
        print("\n📊 Pipeline Statistics:")
        print(f"   Retrieval: {stats['retrieval']}")
        if stats["cache"]:
            print(f"   Cache: {stats['cache']}")
        print(f"   Guardrails: {stats['guardrails']}")
        return

    if args.interactive:
        # Interactive mode
        print("\n💬 Interactive mode - Type 'quit' to exit")
        print("=" * 60)

        while True:
            try:
                question = input("\n❓ Ask a question: ").strip()
                if question.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break

                if not question:
                    continue

                # Process question
                result = pipeline.query(
                    question=question,
                    user_role=args.role,
                    top_k=args.top_k,
                    bm25_weight=args.bm25_weight,
                    dense_weight=args.dense_weight,
                )

                # Display answer
                print(f"\n💡 Answer ({result['total_time']:.2f}s):")
                print(f"   {result['answer']}")

                if result["sources"]:
                    print("\n📚 Sources:")
                    for source in result["sources"][:3]:  # Show top 3
                        print(f"   {source}")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

    else:
        # Single question mode
        if not args.question:
            print("❌ Please provide a question or use --interactive mode")
            sys.exit(1)

        print(f"\n❓ Question: {args.question}")
        print(f"   Role: {args.role}")
        print("-" * 40)

        # Process question
        result = pipeline.query(
            question=args.question,
            user_role=args.role,
            top_k=args.top_k,
            bm25_weight=args.bm25_weight,
            dense_weight=args.dense_weight,
        )

        # Display answer
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

        # Show brief metadata
        print(
            f"\n📊 Context: {result['context_used']}, "
            f"Retrieval: {result['retrieval_time']:.2f}s, "
            f"Generation: {result['generation_time']:.2f}s"
        )

        if result["cache_hit"]:
            print("   ⚡ Cache hit!")


if __name__ == "__main__":
    main()
