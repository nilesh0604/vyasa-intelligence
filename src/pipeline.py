"""End-to-end RAG pipeline for Vyasa Intelligence.

This module integrates retrieval and generation to provide complete
question-answering functionality for the Mahabharata knowledge base.
"""

import logging
import os
import time
from typing import Dict, List, Optional

from src.generation.answer_generator import AnswerGenerator
from src.generation.cache import RedisCache, ResponseCache
from src.generation.guardrails import ContentGuardrails
from src.retrieval.hybrid_search import HybridSearcher

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Complete RAG pipeline for Mahabharata Q&A."""

    def __init__(
        self,
        chroma_dir: Optional[str] = None,
        bm25_path: Optional[str] = None,
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        enable_cache: bool = True,
        cache_type: str = "memory",
        redis_url: Optional[str] = None,
        enable_guardrails: bool = True,
        enable_tracing: bool = True,
    ):
        """Initialize the RAG pipeline.

        Args:
            chroma_dir: Directory containing ChromaDB index
            bm25_path: Path to BM25 index
            embedding_model: Name of embedding model
            llm_provider: LLM provider ('ollama' or 'groq')
            llm_model: LLM model name
            enable_cache: Whether to enable response caching
            cache_type: Type of cache ('memory' or 'redis')
            redis_url: Redis URL for distributed cache
            enable_guardrails: Whether to enable content guardrails
            enable_tracing: Whether to enable LangSmith tracing
        """
        # Set default paths
        self.chroma_dir = chroma_dir or os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
        self.bm25_path = bm25_path or os.getenv(
            "BM25_INDEX_PATH", "./data/bm25_index.pkl"
        )

        # Initialize retrieval
        logger.info("Initializing retrieval components...")
        self.retriever = HybridSearcher(
            chroma_dir=self.chroma_dir,
            bm25_path=self.bm25_path,
            embedding_model=embedding_model,
        )

        # Initialize generation
        logger.info("Initializing generation components...")
        self.generator = AnswerGenerator(
            llm_provider=llm_provider,
            llm_model=llm_model,
            enable_tracing=enable_tracing,
        )

        # Initialize cache
        self.cache = None
        if enable_cache:
            logger.info(f"Initializing {cache_type} cache...")
            if cache_type == "redis" and redis_url:
                self.cache = RedisCache(redis_url=redis_url)
            else:
                self.cache = ResponseCache(enable_persistence=True)

        # Initialize guardrails
        self.guardrails = ContentGuardrails() if enable_guardrails else None

        logger.info("✓ RAG pipeline initialized successfully")

    def query(
        self,
        question: str,
        user_role: str = "public",
        top_k: int = 5,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        use_cache: Optional[bool] = None,
        conversation_history: Optional[List[Dict]] = None,
    ) -> Dict[str, any]:
        """Process a query through the complete RAG pipeline.

        Args:
            question: User's question
            user_role: User's role (public, scholar, admin)
            top_k: Number of documents to retrieve
            bm25_weight: Weight for BM25 retrieval
            dense_weight: Weight for dense retrieval
            use_cache: Whether to use cache (overrides instance setting)
            conversation_history: Previous Q&A pairs

        Returns:
            Complete response with answer, sources, and metadata
        """
        start_time = time.time()

        # Initialize response structure
        response = {
            "question": question,
            "answer": "",
            "sources": [],
            "citations": [],
            "context_used": False,
            "retrieval_time": 0,
            "generation_time": 0,
            "total_time": 0,
            "cache_hit": False,
            "guardrails_passed": True,
            "metadata": {},
        }

        try:
            # Check guardrails for input
            if self.guardrails:
                input_check = self.guardrails.check_input(question, user_role)
                if not input_check["allowed"]:
                    response["answer"] = (
                        "I cannot process this question as it violates content guidelines."
                    )
                    response["guardrails_passed"] = False
                    response["metadata"]["blocked_categories"] = input_check[
                        "blocked_categories"
                    ]
                    return response

            # Check cache
            context_docs = None  # Initialize context_docs
            if use_cache is not False and self.cache:
                # First, perform retrieval to get context hash
                retrieval_start = time.time()
                context_docs = self._retrieve_context(
                    question, top_k, bm25_weight, dense_weight
                )
                response["retrieval_time"] = time.time() - retrieval_start

                if context_docs:
                    context_hash = self.cache.generate_context_hash(context_docs)
                    cached_response = self.cache.get(question, context_hash, user_role)

                    if cached_response:
                        response.update(cached_response)
                        response["cache_hit"] = True
                        response["total_time"] = time.time() - start_time
                        logger.info(f"Cache hit for query: {question[:50]}...")
                        return response

            # Perform retrieval
            if not context_docs:
                retrieval_start = time.time()
                context_docs = self._retrieve_context(
                    question, top_k, bm25_weight, dense_weight
                )
                response["retrieval_time"] = time.time() - retrieval_start

            response["context_used"] = len(context_docs) > 0

            # Generate answer
            generation_start = time.time()
            generation_result = self.generator.generate_answer(
                question=question,
                context_docs=context_docs,
                user_role=user_role,
                conversation_history=conversation_history,
            )
            response["generation_time"] = time.time() - generation_start

            # Update response with generation results
            response.update(generation_result)

            # Check guardrails for output
            if self.guardrails:
                output_check = self.guardrails.check_output(
                    response["answer"], response["context_used"]
                )
                if not output_check["allowed"]:
                    response["answer"] = (
                        "The generated response violates content guidelines and cannot be displayed."
                    )
                    response["guardrails_passed"] = False
                    response["metadata"]["output_violations"] = output_check[
                        "violations"
                    ]
                else:
                    response["metadata"]["output_warnings"] = output_check["warnings"]

            # Cache the response
            if use_cache is not False and self.cache and context_docs:
                context_hash = self.cache.generate_context_hash(context_docs)
                cache_data = {
                    "answer": response["answer"],
                    "sources": response["sources"],
                    "citations": response["citations"],
                    "context_used": response["context_used"],
                    "metadata": response["metadata"],
                }
                self.cache.set(question, context_hash, cache_data, user_role)

            # Add retrieval metadata
            response["metadata"]["retrieval_count"] = len(context_docs)
            response["metadata"]["retrieval_scores"] = [
                doc.get("relevance_score", 0) for doc in context_docs
            ]

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            response["answer"] = (
                f"An error occurred while processing your question: {str(e)}"
            )
            response["metadata"]["error"] = str(e)

        finally:
            response["total_time"] = time.time() - start_time

        return response

    def _retrieve_context(
        self,
        question: str,
        top_k: int,
        bm25_weight: float,
        dense_weight: float,
    ) -> List[Dict]:
        """Retrieve relevant context for a question.

        Args:
            question: User's question
            top_k: Number of documents to retrieve
            bm25_weight: Weight for BM25 retrieval
            dense_weight: Weight for dense retrieval

        Returns:
            List of retrieved documents
        """
        # Classify query type for optimized retrieval
        query_type = self._classify_query(question)

        # Adjust retrieval strategy based on query type
        if query_type == "entity":
            # Entity queries favor BM25
            docs = self.retriever.search(
                query=question,
                top_k=top_k,
                bm25_weight=0.7,
                dense_weight=0.3,
                include_scores=True,
            )
        elif query_type == "philosophy":
            # Philosophical queries favor dense search
            docs = self.retriever.search(
                query=question,
                top_k=top_k,
                bm25_weight=0.3,
                dense_weight=0.7,
                include_scores=True,
            )
        else:
            # Balanced hybrid search
            docs = self.retriever.search(
                query=question,
                top_k=top_k,
                bm25_weight=bm25_weight,
                dense_weight=dense_weight,
                include_scores=True,
            )

        return docs

    def _classify_query(self, question: str) -> str:
        """Classify query type for optimized retrieval.

        Args:
            question: User's question

        Returns:
            Query type: 'entity', 'philosophy', or 'general'
        """
        question_lower = question.lower()

        # Entity query indicators
        entity_keywords = [
            "who",
            "what",
            "where",
            "when",
            "character",
            "person",
            "place",
            "weapon",
            "king",
            "prince",
            "goddess",
            "god",
            "demon",
            "rishi",
        ]
        if any(keyword in question_lower for keyword in entity_keywords):
            return "entity"

        # Philosophy query indicators
        philosophy_keywords = [
            "why",
            "how",
            "dharma",
            "karma",
            "moral",
            "ethic",
            "meaning",
            "philosophy",
            "teaching",
            "lesson",
            "wisdom",
            "truth",
            "duty",
        ]
        if any(keyword in question_lower for keyword in philosophy_keywords):
            return "philosophy"

        return "general"

    def get_statistics(self) -> Dict:
        """Get pipeline statistics.

        Returns:
            Dictionary with pipeline statistics
        """
        stats = {
            "retrieval": self.retriever.get_statistics(),
            "cache": self.cache.get_statistics() if self.cache else None,
            "guardrails": {
                "enabled": self.guardrails is not None,
                "blocked_categories": (
                    list(self.guardrails.blocked_categories.keys())
                    if self.guardrails
                    else []
                ),
            },
        }

        return stats

    def clear_cache(self):
        """Clear the response cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Response cache cleared")

    def update_llm_config(self, **kwargs):
        """Update LLM configuration.

        Args:
            **kwargs: LLM configuration parameters
        """
        self.generator.update_llm_config(**kwargs)
        logger.info("LLM configuration updated")
