"""Reranking module using cross-encoders for improved relevance.

This module implements various reranking strategies including cross-encoders
and other advanced reranking techniques.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sentence_transformers import CrossEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

# Import CrossEncoder for type checking
try:
    from sentence_transformers import CrossEncoder as STCrossEncoder
except ImportError:
    STCrossEncoder = None


class Reranker:
    """Reranks retrieved documents using cross-encoders and other techniques."""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 32,
    ):
        """Initialize the reranker.
        
        Args:
            model_name: Name of the cross-encoder model
            device: Device to run on ('cpu' or 'cuda')
            max_length: Maximum sequence length for the model
            batch_size: Batch size for inference
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Initialize model
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the cross-encoder model."""
        try:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            
            # Try loading as a CrossEncoder first
            if "reranker" in self.model_name.lower() or "cross" in self.model_name.lower():
                self.model = CrossEncoder(
                    self.model_name,
                    device=self.device,
                    max_length=self.max_length
                )
            else:
                # Load as a regular sequence classification model
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name
                ).to(self.device)
            
            logger.info("✓ Cross-encoder model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            logger.warning("Falling back to simple score-based reranking")
            self.model = None
    
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5,
        return_scores: bool = True,
    ) -> List[Dict]:
        """Rerank documents based on query relevance.
        
        Args:
            query: The search query
            documents: List of document dictionaries with 'content' and 'metadata'
            top_k: Number of top documents to return
            return_scores: Whether to include reranking scores
            
        Returns:
            Reranked list of documents
        """
        if not documents:
            return []
        
        if self.model is None:
            # Fallback to simple keyword-based reranking
            return self._fallback_rerank(query, documents, top_k, return_scores)
        
        # Prepare query-document pairs
        pairs = [(query, doc["content"]) for doc in documents]
        
        # Get relevance scores
        if hasattr(self.model, 'predict'):  # Check if it's a CrossEncoder
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        else:
            scores = self._predict_with_transformer(pairs)
        
        # Sort by scores
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k documents
        reranked = []
        for doc, score in scored_docs[:top_k]:
            doc_copy = doc.copy()
            if return_scores:
                doc_copy["rerank_score"] = float(score)
            reranked.append(doc_copy)
        
        return reranked
    
    def _predict_with_transformer(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Predict scores using a transformer model."""
        scores = []
        
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i:i + self.batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                [f"Query: {q} Document: {d}" for q, d in batch_pairs],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_scores = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
                scores.extend(batch_scores.tolist())
        
        return scores
    
    def _fallback_rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int,
        return_scores: bool,
    ) -> List[Dict]:
        """Fallback reranking using simple keyword matching."""
        query_terms = set(query.lower().split())
        
        scored_docs = []
        for doc in documents:
            content = doc["content"].lower()
            content_terms = set(content.split())
            
            # Simple overlap score
            overlap = len(query_terms.intersection(content_terms))
            coverage = overlap / len(query_terms) if query_terms else 0
            
            # TF-IDF-like scoring (simplified)
            term_freq = sum(content.count(term) for term in query_terms)
            doc_length = len(content.split())
            normalized_tf = term_freq / doc_length if doc_length > 0 else 0
            
            # Combined score
            score = 0.7 * coverage + 0.3 * normalized_tf
            
            scored_docs.append((doc, score))
        
        # Sort and return
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        result = []
        for doc, score in scored_docs[:top_k]:
            doc_copy = doc.copy()
            if return_scores:
                doc_copy["rerank_score"] = score
            result.append(doc_copy)
        
        return result
    
    def multi_stage_rerank(
        self,
        query: str,
        documents: List[Dict],
        stage1_top_k: int = 20,
        stage2_top_k: int = 5,
        filters: Optional[Dict[str, List[str]]] = None,
    ) -> List[Dict]:
        """Perform multi-stage reranking with optional filtering.
        
        Args:
            query: The search query
            documents: List of document dictionaries
            stage1_top_k: Number of documents after first stage
            stage2_top_k: Number of documents after second stage
            filters: Optional metadata filters to apply
            
        Returns:
            Reranked list of documents
        """
        # Apply filters if provided
        if filters:
            documents = self._apply_filters(documents, filters)
        
        if len(documents) <= stage2_top_k:
            return self.rerank(query, documents, stage2_top_k)
        
        # Stage 1: Quick reranking on all documents
        stage1_results = self.rerank(query, documents, stage1_top_k)
        
        # Stage 2: More careful reranking on top candidates
        stage2_results = self.rerank(query, stage1_results, stage2_top_k)
        
        return stage2_results
    
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
        filtered = []
        
        for doc in documents:
            metadata = doc.get("metadata", {})
            include = True
            
            for field, allowed_values in filters.items():
                if field in metadata:
                    value = metadata[field]
                    if isinstance(value, list):
                        # Check if any value matches
                        if not any(v in allowed_values for v in value):
                            include = False
                            break
                    else:
                        if value not in allowed_values:
                            include = False
                            break
                else:
                    # Field not present, exclude
                    include = False
                    break
            
            if include:
                filtered.append(doc)
        
        return filtered
    
    def diversity_rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5,
        diversity_lambda: float = 0.3,
        similarity_threshold: float = 0.8,
    ) -> List[Dict]:
        """Rerank with diversity consideration.
        
        Args:
            query: The search query
            documents: List of document dictionaries
            top_k: Number of documents to return
            diversity_lambda: Weight for diversity (0 = pure relevance, 1 = pure diversity)
            similarity_threshold: Threshold for document similarity
            
        Returns:
            Diversified reranked documents
        """
        if not documents:
            return []
        
        # First get relevance scores
        relevance_scores = {}
        if self.model:
            pairs = [(query, doc["content"]) for doc in documents]
            if hasattr(self.model, 'predict'):  # Check if it's a CrossEncoder
                scores = self.model.predict(pairs, batch_size=self.batch_size)
            else:
                scores = self._predict_with_transformer(pairs)
            
            for doc, score in zip(documents, scores):
                relevance_scores[doc["chunk_id"]] = float(score)
        else:
            # Fallback scoring
            query_terms = set(query.lower().split())
            for doc in documents:
                content = doc["content"].lower()
                overlap = len(query_terms.intersection(set(content.split())))
                relevance_scores[doc["chunk_id"]] = overlap / len(query_terms)
        
        # Greedy selection with diversity
        selected = []
        remaining = documents.copy()
        
        while remaining and len(selected) < top_k:
            # Find best document considering relevance and diversity
            best_doc = None
            best_score = -float("inf")
            
            for doc in remaining:
                relevance = relevance_scores[doc["chunk_id"]]
                
                # Calculate diversity penalty
                diversity_penalty = 0
                if selected:
                    max_similarity = self._max_similarity(doc, selected)
                    if max_similarity > similarity_threshold:
                        diversity_penalty = max_similarity - similarity_threshold
                
                # Combined score
                combined_score = (1 - diversity_lambda) * relevance - diversity_lambda * diversity_penalty
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_doc = doc
            
            if best_doc:
                selected.append(best_doc)
                remaining.remove(best_doc)
            else:
                break
        
        # Add scores to selected documents
        for doc in selected:
            doc["rerank_score"] = relevance_scores[doc["chunk_id"]]
        
        return selected
    
    def _max_similarity(self, doc: Dict, selected_docs: List[Dict]) -> float:
        """Calculate maximum similarity between doc and selected documents."""
        # Simple content overlap similarity (can be enhanced with embeddings)
        doc_terms = set(doc["content"].lower().split())
        max_sim = 0.0
        
        for selected in selected_docs:
            selected_terms = set(selected["content"].lower().split())
            intersection = len(doc_terms.intersection(selected_terms))
            union = len(doc_terms.union(selected_terms))
            
            if union > 0:
                similarity = intersection / union
                max_sim = max(max_sim, similarity)
        
        return max_sim
    
    def contextual_rerank(
        self,
        query: str,
        documents: List[Dict],
        conversation_history: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[Dict]:
        """Rerank considering conversation context.
        
        Args:
            query: The current search query
            documents: List of document dictionaries
            conversation_history: Previous conversation turns
            top_k: Number of documents to return
            
        Returns:
            Contextually reranked documents
        """
        # Build contextual query
        contextual_query = query
        if conversation_history:
            # Include recent context
            recent_context = " ".join(conversation_history[-3:])  # Last 3 turns
            contextual_query = f"Context: {recent_context} Query: {query}"
        
        # Rerank with contextual query
        return self.rerank(contextual_query, documents, top_k)
    
    def get_reranking_statistics(self) -> Dict:
        """Get statistics about the reranker.
        
        Returns:
            Dictionary with reranker statistics
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "model_loaded": self.model is not None,
        }
