"""Hybrid search implementation combining BM25 and dense retrieval.

This module implements both sparse (BM25) and dense (vector) retrieval
methods for the Mahabharata RAG system.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class HybridSearcher:
    """Hybrid search implementation combining BM25 and dense retrieval."""
    
    def __init__(
        self,
        chroma_dir: Union[str, Path],
        bm25_path: Union[str, Path],
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        device: str = "cpu",
    ):
        """Initialize the hybrid searcher.
        
        Args:
            chroma_dir: Directory containing ChromaDB index
            bm25_path: Path to BM25 index pickle file
            embedding_model: Name of the embedding model
            device: Device to run embeddings on ('cpu' or 'cuda')
        """
        self.chroma_dir = Path(chroma_dir)
        self.bm25_path = Path(bm25_path)
        self.embedding_model_name = embedding_model
        self.device = device
        
        # Initialize components
        self.embedding_model = None
        self.chroma_client = None
        self.chroma_collection = None
        self.bm25_index = None
        self.bm25_data = None
        
        # Load indices
        self._load_indices()
    
    def _load_indices(self):
        """Load ChromaDB and BM25 indices."""
        try:
            # Load embedding model
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            # Load ChromaDB
            logger.info(f"Loading ChromaDB from: {self.chroma_dir}")
            self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_dir))
            self.chroma_collection = self.chroma_client.get_collection("mahabharata")
            
            # Load BM25 index
            logger.info(f"Loading BM25 index from: {self.bm25_path}")
            if self.bm25_path.exists():
                with open(self.bm25_path, "rb") as f:
                    # Loading trusted internal data
                    self.bm25_data = pickle.load(f)  # nosec B301
                    self.bm25_index = self.bm25_data["bm25_index"]
            else:
                logger.error(f"BM25 index not found at: {self.bm25_path}")
                raise FileNotFoundError(f"BM25 index not found: {self.bm25_path}")
            
            logger.info("✓ All indices loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load indices: {e}")
            raise
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        include_scores: bool = True,
    ) -> List[Dict]:
        """Perform hybrid search combining BM25 and dense retrieval.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            bm25_weight: Weight for BM25 scores (0.0 to 1.0)
            dense_weight: Weight for dense scores (0.0 to 1.0)
            include_scores: Whether to include relevance scores
            
        Returns:
            List of retrieved documents with metadata
        """
        # Normalize weights
        total_weight = bm25_weight + dense_weight
        if total_weight > 0:
            bm25_weight = bm25_weight / total_weight
            dense_weight = dense_weight / total_weight
        
        # Perform BM25 search
        bm25_results = self._bm25_search(query, top_k * 2)
        
        # Perform dense search
        dense_results = self._dense_search(query, top_k * 2)
        
        # Combine results
        combined_results = self._combine_results(
            bm25_results,
            dense_results,
            bm25_weight,
            dense_weight,
            top_k,
            include_scores,
        )
        
        return combined_results
    
    def _bm25_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Perform BM25 search.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of (chunk_id, score) tuples
        """
        # Tokenize query
        tokens = query.lower().split()
        
        # Get BM25 scores
        doc_scores = self.bm25_index.get_scores(tokens)
        
        # Get top-k results
        top_indices = np.argsort(doc_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if doc_scores[idx] > 0:  # Only include results with positive scores
                chunk_id = self.bm25_data["chunk_ids"][idx]
                score = float(doc_scores[idx])
                results.append((chunk_id, score))
        
        return results
    
    def _dense_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Perform dense vector search.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of (chunk_id, score) tuples
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            query, convert_to_numpy=True, normalize_embeddings=True
        )
        
        # Search ChromaDB
        results = self.chroma_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["metadatas", "distances", "documents"]
        )
        
        # Convert distances to similarities (higher is better)
        chroma_results = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                # Convert cosine distance to similarity
                distance = results["distances"][0][i]
                similarity = 1.0 - distance
                chroma_results.append((chunk_id, similarity))
        
        return chroma_results
    
    def _combine_results(
        self,
        bm25_results: List[Tuple[str, float]],
        dense_results: List[Tuple[str, float]],
        bm25_weight: float,
        dense_weight: float,
        top_k: int,
        include_scores: bool,
    ) -> List[Dict]:
        """Combine BM25 and dense search results.
        
        Args:
            bm25_results: List of (chunk_id, bm25_score) tuples
            dense_results: List of (chunk_id, dense_score) tuples
            bm25_weight: Weight for BM25 scores
            dense_weight: Weight for dense scores
            top_k: Number of final results to return
            include_scores: Whether to include relevance scores
            
        Returns:
            List of combined result documents
        """
        # Create dictionaries for quick lookup
        bm25_dict = {chunk_id: score for chunk_id, score in bm25_results}
        dense_dict = {chunk_id: score for chunk_id, score in dense_results}
        
        # Normalize scores
        bm25_dict = self._normalize_scores(bm25_dict)
        dense_dict = self._normalize_scores(dense_dict)
        
        # Combine scores
        combined_scores = {}
        all_chunk_ids = set(bm25_dict.keys()).union(set(dense_dict.keys()))
        
        for chunk_id in all_chunk_ids:
            bm25_score = bm25_dict.get(chunk_id, 0.0)
            dense_score = dense_dict.get(chunk_id, 0.0)
            combined_score = bm25_weight * bm25_score + dense_weight * dense_score
            combined_scores[chunk_id] = combined_score
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Build result documents
        final_results = []
        for chunk_id, combined_score in sorted_results:
            # Get document data
            doc_data = self._get_document_data(chunk_id)
            if doc_data:
                result = {
                    "chunk_id": chunk_id,
                    "content": doc_data["content"],
                    "metadata": doc_data["metadata"],
                }
                
                if include_scores:
                    result["relevance_score"] = combined_score
                    result["bm25_score"] = bm25_dict.get(chunk_id, 0.0)
                    result["dense_score"] = dense_dict.get(chunk_id, 0.0)
                
                final_results.append(result)
        
        return final_results
    
    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores to [0, 1] range.
        
        Args:
            scores: Dictionary of chunk_id -> score
            
        Returns:
            Normalized scores dictionary
        """
        if not scores:
            return scores
        
        max_score = max(scores.values())
        min_score = min(scores.values())
        
        if max_score == min_score:
            # All scores are the same
            return {k: 1.0 for k in scores}
        
        # Min-max normalization
        normalized = {}
        for chunk_id, score in scores.items():
            normalized[chunk_id] = (score - min_score) / (max_score - min_score)
        
        return normalized
    
    def _get_document_data(self, chunk_id: str) -> Optional[Dict]:
        """Get document data for a chunk ID.
        
        Args:
            chunk_id: The chunk ID to retrieve
            
        Returns:
            Dictionary with content and metadata or None if not found
        """
        # Try BM25 data first
        if self.bm25_data and chunk_id in self.bm25_data["chunk_ids"]:
            idx = self.bm25_data["chunk_ids"].index(chunk_id)
            return {
                "content": self.bm25_data["documents"][idx],
                "metadata": self.bm25_data["metadatas"][idx],
            }
        
        # Fallback to ChromaDB
        try:
            results = self.chroma_collection.get(
                ids=[chunk_id],
                include=["metadatas", "documents"]
            )
            if results["ids"] and results["documents"]:
                return {
                    "content": results["documents"][0],
                    "metadata": results["metadatas"][0],
                }
        except Exception as e:
            logger.warning(f"Failed to get document {chunk_id} from ChromaDB: {e}")
        
        return None
    
    def search_bm25_only(self, query: str, top_k: int = 10) -> List[Dict]:
        """Perform BM25-only search.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of retrieved documents
        """
        results = self._bm25_search(query, top_k)
        
        documents = []
        for chunk_id, score in results:
            doc_data = self._get_document_data(chunk_id)
            if doc_data:
                documents.append({
                    "chunk_id": chunk_id,
                    "content": doc_data["content"],
                    "metadata": doc_data["metadata"],
                    "relevance_score": score,
                })
        
        return documents
    
    def search_dense_only(self, query: str, top_k: int = 10) -> List[Dict]:
        """Perform dense-only search.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of retrieved documents
        """
        results = self._dense_search(query, top_k)
        
        documents = []
        for chunk_id, score in results:
            doc_data = self._get_document_data(chunk_id)
            if doc_data:
                documents.append({
                    "chunk_id": chunk_id,
                    "content": doc_data["content"],
                    "metadata": doc_data["metadata"],
                    "relevance_score": score,
                })
        
        return documents
    
    def get_statistics(self) -> Dict:
        """Get search index statistics.
        
        Returns:
            Dictionary with index statistics
        """
        stats = {
            "chroma_db_count": 0,
            "bm25_doc_count": 0,
            "embedding_model": self.embedding_model_name,
            "device": self.device,
        }
        
        try:
            # ChromaDB count
            if self.chroma_collection:
                stats["chroma_db_count"] = self.chroma_collection.count()
            
            # BM25 count
            if self.bm25_data:
                stats["bm25_doc_count"] = len(self.bm25_data["chunk_ids"])
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
        
        return stats
