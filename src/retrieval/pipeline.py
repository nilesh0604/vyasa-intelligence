"""Main retrieval pipeline orchestration for the Mahabharata RAG system.

This module orchestrates the entire retrieval process including query classification,
hybrid search, rank fusion, and reranking.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from .hybrid_search import HybridSearcher
from .query_classifier import QueryClassifier, QueryType
from .rank_fusion import RankFusion
from .reranker import Reranker

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """Main retrieval pipeline that orchestrates all retrieval components."""
    
    def __init__(
        self,
        chroma_dir: Union[str, Path],
        bm25_path: Union[str, Path],
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        reranker_model: str = "BAAI/bge-reranker-base",
        device: str = "cpu",
        enable_reranking: bool = True,
        enable_query_classification: bool = True,
    ):
        """Initialize the retrieval pipeline.
        
        Args:
            chroma_dir: Directory containing ChromaDB index
            bm25_path: Path to BM25 index pickle file
            embedding_model: Name of the embedding model
            reranker_model: Name of the reranker model
            device: Device to run on ('cpu' or 'cuda')
            enable_reranking: Whether to enable reranking
            enable_query_classification: Whether to enable query classification
        """
        self.chroma_dir = Path(chroma_dir)
        self.bm25_path = Path(bm25_path)
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.device = device
        self.enable_reranking = enable_reranking
        self.enable_query_classification = enable_query_classification
        
        # Initialize components
        self.query_classifier = None
        self.hybrid_searcher = None
        self.rank_fusion = None
        self.reranker = None
        
        # Initialize pipeline
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize all pipeline components."""
        try:
            # Initialize query classifier
            if self.enable_query_classification:
                logger.info("Initializing query classifier...")
                self.query_classifier = QueryClassifier(
                    embedding_model_name=self.embedding_model
                )
            
            # Initialize hybrid searcher
            logger.info("Initializing hybrid searcher...")
            self.hybrid_searcher = HybridSearcher(
                chroma_dir=self.chroma_dir,
                bm25_path=self.bm25_path,
                embedding_model=self.embedding_model,
                device=self.device,
            )
            
            # Initialize rank fusion
            logger.info("Initializing rank fusion...")
            self.rank_fusion = RankFusion(k=60)
            
            # Initialize reranker
            if self.enable_reranking:
                logger.info("Initializing reranker...")
                self.reranker = Reranker(
                    model_name=self.reranker_model,
                    device=self.device,
                )
            
            logger.info("✓ Retrieval pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        strategy: Optional[Dict] = None,
        conversation_history: Optional[List[str]] = None,
        filters: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, any]:
        """Perform end-to-end retrieval.
        
        Args:
            query: The search query
            top_k: Number of results to return
            strategy: Optional retrieval strategy (auto-determined if not provided)
            conversation_history: Optional conversation history for contextual retrieval
            filters: Optional metadata filters
            
        Returns:
            Dictionary with retrieval results and metadata
        """
        # Step 1: Query classification (if enabled)
        if self.enable_query_classification and strategy is None:
            strategy = self.query_classifier.get_retrieval_strategy(query)
        elif strategy is None:
            # Default strategy
            strategy = {
                "query_type": "unknown",
                "confidence": 0.5,
                "use_bm25": True,
                "use_dense": True,
                "bm25_weight": 0.5,
                "dense_weight": 0.5,
                "rerank": self.enable_reranking,
                "top_k": top_k * 2 if self.enable_reranking else top_k,
                "rerank_top_k": top_k,
                "query_expansion": False,
                "hyde": False,
            }
        
        # Step 2: Query expansion (if enabled)
        expanded_query = query
        if strategy.get("query_expansion", False):
            expanded_query = self._expand_query(query)
        
        # Step 3: HyDE query transformation (if enabled)
        if strategy.get("hyde", False):
            hyde_query = self._generate_hyde_query(expanded_query)
            search_query = hyde_query
        else:
            search_query = expanded_query
        
        # Step 4: Hybrid search
        search_results = self._perform_hybrid_search(
            search_query,
            strategy,
            strategy["top_k"],
        )
        
        # Step 5: Apply filters (if provided)
        if filters:
            search_results = self._apply_filters(search_results, filters)
        
        # Step 6: Reranking (if enabled)
        if strategy.get("rerank", False) and self.reranker:
            if conversation_history:
                # Contextual reranking
                final_results = self.reranker.contextual_rerank(
                    query=search_query,
                    documents=search_results,
                    conversation_history=conversation_history,
                    top_k=strategy.get("rerank_top_k", top_k),
                )
            else:
                # Standard reranking
                final_results = self.reranker.rerank(
                    query=search_query,
                    documents=search_results,
                    top_k=strategy.get("rerank_top_k", top_k),
                )
        else:
            # No reranking, just truncate
            final_results = search_results[:top_k]
        
        # Step 7: Prepare response
        response = {
            "query": query,
            "results": final_results,
            "strategy": strategy,
            "num_retrieved": len(search_results),
            "num_returned": len(final_results),
        }
        
        # Add debug information
        if strategy.get("query_expansion", False):
            response["expanded_query"] = expanded_query
        if strategy.get("hyde", False):
            response["hyde_query"] = search_query
        
        return response
    
    def _perform_hybrid_search(
        self,
        query: str,
        strategy: Dict,
        top_k: int,
    ) -> List[Dict]:
        """Perform hybrid search based on strategy.
        
        Args:
            query: The search query
            strategy: Retrieval strategy dictionary
            top_k: Number of results to retrieve
            
        Returns:
            List of retrieved documents
        """
        # Determine which retrieval methods to use
        use_bm25 = strategy.get("use_bm25", True)
        use_dense = strategy.get("use_dense", True)
        
        if use_bm25 and use_dense:
            # Hybrid search with RRF fusion
            bm25_results = self.hybrid_searcher._bm25_search(query, top_k)
            dense_results = self.hybrid_searcher._dense_search(query, top_k)
            
            # Convert to (doc_id, score) format for fusion
            bm25_pairs = [(doc_id, score) for doc_id, score in bm25_results]
            dense_pairs = [(doc_id, score) for doc_id, score in dense_results]
            
            # Apply RRF fusion
            fused_results = self.rank_fusion.reciprocal_rank_fusion(
                [bm25_pairs, dense_pairs],
                top_k=top_k,
                weights=[strategy.get("bm25_weight", 0.5), strategy.get("dense_weight", 0.5)],
            )
            
            # Get full documents
            documents = {}
            for doc_id, _ in bm25_results + dense_results:
                doc_data = self.hybrid_searcher._get_document_data(doc_id)
                if doc_data:
                    documents[doc_id] = {
                        "chunk_id": doc_id,
                        "content": doc_data["content"],
                        "metadata": doc_data["metadata"],
                    }
            
            # Merge with metadata
            final_results = self.rank_fusion.merge_with_metadata(
                fused_results, documents, include_individual_scores=True
            )
            
        elif use_bm25:
            # BM25 only
            final_results = self.hybrid_searcher.search_bm25_only(query, top_k)
            
        elif use_dense:
            # Dense only
            final_results = self.hybrid_searcher.search_dense_only(query, top_k)
            
        else:
            # No retrieval method specified
            logger.warning("No retrieval method specified, using hybrid search")
            final_results = self.hybrid_searcher.search(
                query, top_k, include_scores=True
            )
        
        return final_results
    
    def _expand_query(self, query: str) -> str:
        """Expand query with related terms.
        
        Args:
            query: Original query
            
        Returns:
            Expanded query
        """
        # Simple query expansion with Mahabharata-specific terms
        expansions = {
            "krishna": ["lord krishna", "vasudeva", "gopal"],
            "arjuna": ["partha", "dhananjaya", "phalguna"],
            "war": ["battle", "conflict", "mahabharata war"],
            "dharma": ["righteousness", "duty", "moral law"],
            "karna": ["radheya", "suryaputra", "angaraja"],
        }
        
        expanded_terms = [query.lower()]
        query_lower = query.lower()
        
        for term, synonyms in expansions.items():
            if term in query_lower:
                expanded_terms.extend(synonyms)
        
        return " ".join(expanded_terms)
    
    def _generate_hyde_query(self, query: str) -> str:
        """Generate a hypothetical document for HyDE.
        
        Args:
            query: Original query
            
        Returns:
            Hypothetical document query
        """
        # For now, return a simple expansion
        # In a full implementation, this would use an LLM
        return f"According to the Mahabharata, {query}. This passage describes the details and significance of this aspect in the epic."
    
    def _apply_filters(
        self,
        documents: List[Dict],
        filters: Dict[str, List[str]],
    ) -> List[Dict]:
        """Apply metadata filters to documents.
        
        Args:
            documents: List of document dictionaries
            filters: Dictionary of field -> allowed values
            
        Returns:
            Filtered list of documents
        """
        if not filters:
            return documents
        
        filtered = []
        
        for doc in documents:
            metadata = doc.get("metadata", {})
            include = True
            
            for field, allowed_values in filters.items():
                if field in metadata:
                    value = metadata[field]
                    if isinstance(value, list):
                        # Check if any value matches
                        if not any(v.lower() in [av.lower() for av in allowed_values] for v in value):
                            include = False
                            break
                    else:
                        if value.lower() not in [av.lower() for av in allowed_values]:
                            include = False
                            break
                else:
                    # Field not present, exclude
                    include = False
                    break
            
            if include:
                filtered.append(doc)
        
        return filtered
    
    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 5,
        strategy: Optional[Dict] = None,
    ) -> List[Dict]:
        """Perform batch retrieval for multiple queries.
        
        Args:
            queries: List of search queries
            top_k: Number of results per query
            strategy: Optional retrieval strategy
            
        Returns:
            List of retrieval results
        """
        results = []
        
        for query in queries:
            try:
                result = self.retrieve(query, top_k, strategy)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to retrieve for query '{query}': {e}")
                results.append({
                    "query": query,
                    "results": [],
                    "error": str(e),
                })
        
        return results
    
    def get_pipeline_statistics(self) -> Dict:
        """Get statistics about the retrieval pipeline.
        
        Returns:
            Dictionary with pipeline statistics
        """
        stats = {
            "components": {
                "query_classifier": self.query_classifier is not None,
                "hybrid_searcher": self.hybrid_searcher is not None,
                "rank_fusion": self.rank_fusion is not None,
                "reranker": self.reranker is not None,
            },
            "models": {
                "embedding_model": self.embedding_model,
                "reranker_model": self.reranker_model if self.enable_reranking else None,
                "device": self.device,
            },
        }
        
        # Add search statistics
        if self.hybrid_searcher:
            stats["search_stats"] = self.hybrid_searcher.get_statistics()
        
        # Add reranker statistics
        if self.reranker:
            stats["reranker_stats"] = self.reranker.get_reranking_statistics()
        
        return stats
    
    def health_check(self) -> Dict:
        """Check the health of all pipeline components.
        
        Returns:
            Dictionary with health check results
        """
        health = {
            "overall": "healthy",
            "components": {},
        }
        
        # Check query classifier
        if self.query_classifier:
            try:
                self.query_classifier.classify_query("test query")
                health["components"]["query_classifier"] = "healthy"
            except Exception as e:
                health["components"]["query_classifier"] = f"unhealthy: {e}"
                health["overall"] = "degraded"
        else:
            health["components"]["query_classifier"] = "disabled"
        
        # Check hybrid searcher
        if self.hybrid_searcher:
            try:
                stats = self.hybrid_searcher.get_statistics()
                if stats["chroma_db_count"] > 0 and stats["bm25_doc_count"] > 0:
                    health["components"]["hybrid_searcher"] = "healthy"
                else:
                    health["components"]["hybrid_searcher"] = "warning: empty indices"
                    health["overall"] = "degraded"
            except Exception as e:
                health["components"]["hybrid_searcher"] = f"unhealthy: {e}"
                health["overall"] = "unhealthy"
        else:
            health["components"]["hybrid_searcher"] = "disabled"
        
        # Check reranker
        if self.reranker:
            try:
                reranker_stats = self.reranker.get_reranking_statistics()
                if reranker_stats["model_loaded"]:
                    health["components"]["reranker"] = "healthy"
                else:
                    health["components"]["reranker"] = "warning: model not loaded"
                    health["overall"] = "degraded"
            except Exception as e:
                health["components"]["reranker"] = f"unhealthy: {e}"
                health["overall"] = "degraded"
        else:
            health["components"]["reranker"] = "disabled"
        
        # Rank fusion is always healthy (no external dependencies)
        health["components"]["rank_fusion"] = "healthy"
        
        return health
