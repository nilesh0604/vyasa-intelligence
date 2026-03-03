"""Reciprocal Rank Fusion (RRF) implementation for result merging.

This module implements RRF and other fusion techniques to combine
results from multiple retrieval systems.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class RankFusion:
    """Implements various rank fusion techniques for combining search results."""
    
    def __init__(self, k: int = 60):
        """Initialize the rank fusion module.
        
        Args:
            k: RRF parameter (typically 60 for web search, smaller values work well for RAG)
        """
        self.k = k
    
    def reciprocal_rank_fusion(
        self,
        result_lists: List[List[Tuple[str, float]]],
        top_k: int = 10,
        weights: Optional[List[float]] = None,
    ) -> List[Tuple[str, float]]:
        """Perform Reciprocal Rank Fusion (RRF) on multiple result lists.
        
        RRF score = sum(w_i / (k + rank_i)) for each result in each list
        
        Args:
            result_lists: List of result lists, each containing (doc_id, score) tuples
            top_k: Number of top results to return
            weights: Optional weights for each result list (default: equal weights)
            
        Returns:
            Fused results as (doc_id, rrf_score) tuples
        """
        if not result_lists:
            return []
        
        # Set default weights if not provided
        if weights is None:
            weights = [1.0] * len(result_lists)
        elif len(weights) != len(result_lists):
            raise ValueError("Number of weights must match number of result lists")
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Calculate RRF scores
        rrf_scores = {}
        
        for i, (results, weight) in enumerate(zip(result_lists, weights)):
            # Sort by original score if not already sorted
            sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
            
            for rank, (doc_id, original_score) in enumerate(sorted_results, 1):
                # RRF formula: w / (k + rank)
                rrf_score = weight / (self.k + rank)
                
                if doc_id in rrf_scores:
                    rrf_scores[doc_id] += rrf_score
                else:
                    rrf_scores[doc_id] = rrf_score
        
        # Sort by RRF score and return top-k
        fused_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return fused_results[:top_k]
    
    def weighted_score_fusion(
        self,
        result_lists: List[List[Tuple[str, float]]],
        top_k: int = 10,
        weights: Optional[List[float]] = None,
        normalize: bool = True,
    ) -> List[Tuple[str, float]]:
        """Perform weighted score fusion on multiple result lists.
        
        Args:
            result_lists: List of result lists, each containing (doc_id, score) tuples
            top_k: Number of top results to return
            weights: Optional weights for each result list (default: equal weights)
            normalize: Whether to normalize scores before fusion
            
        Returns:
            Fused results as (doc_id, fused_score) tuples
        """
        if not result_lists:
            return []
        
        # Set default weights if not provided
        if weights is None:
            weights = [1.0] * len(result_lists)
        elif len(weights) != len(result_lists):
            raise ValueError("Number of weights must match number of result lists")
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Calculate fused scores
        fused_scores = {}
        
        for i, (results, weight) in enumerate(zip(result_lists, weights)):
            # Normalize scores if requested
            if normalize and results:
                scores = [score for _, score in results]
                max_score = max(scores)
                min_score = min(scores)
                
                if max_score != min_score:
                    normalized_results = [
                        (doc_id, (score - min_score) / (max_score - min_score))
                        for doc_id, score in results
                    ]
                else:
                    normalized_results = [(doc_id, 1.0) for doc_id, _ in results]
            else:
                normalized_results = results
            
            # Add weighted scores
            for doc_id, score in normalized_results:
                weighted_score = weight * score
                
                if doc_id in fused_scores:
                    fused_scores[doc_id] += weighted_score
                else:
                    fused_scores[doc_id] = weighted_score
        
        # Sort by fused score and return top-k
        fused_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return fused_results[:top_k]
    
    def condorcet_fusion(
        self,
        result_lists: List[List[Tuple[str, float]]],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Perform Condorcet fusion based on pairwise preferences.
        
        This method counts how many times each document is ranked higher
        than other documents across all result lists.
        
        Args:
            result_lists: List of result lists, each containing (doc_id, score) tuples
            top_k: Number of top results to return
            
        Returns:
            Fused results as (doc_id, condorcet_score) tuples
        """
        if not result_lists:
            return []
        
        # Collect all unique documents
        all_docs = set()
        for results in result_lists:
            all_docs.update(doc_id for doc_id, _ in results)
        
        # Calculate pairwise wins
        wins = {doc_id: 0 for doc_id in all_docs}
        
        for results in result_lists:
            # Sort by score
            sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
            
            # Count wins for each document
            for i, (doc_i, _) in enumerate(sorted_results):
                for j, (doc_j, _) in enumerate(sorted_results):
                    if i < j:  # doc_i is ranked higher than doc_j
                        wins[doc_i] += 1
        
        # Sort by wins and return top-k
        fused_results = sorted(wins.items(), key=lambda x: x[1], reverse=True)
        return fused_results[:top_k]
    
    def borda_count_fusion(
        self,
        result_lists: List[List[Tuple[str, float]]],
        top_k: int = 10,
        weights: Optional[List[float]] = None,
    ) -> List[Tuple[str, float]]:
        """Perform Borda count fusion.
        
        Each document gets points based on its rank position.
        Higher ranked documents get more points.
        
        Args:
            result_lists: List of result lists, each containing (doc_id, score) tuples
            top_k: Number of top results to return
            weights: Optional weights for each result list (default: equal weights)
            
        Returns:
            Fused results as (doc_id, borda_score) tuples
        """
        if not result_lists:
            return []
        
        # Set default weights if not provided
        if weights is None:
            weights = [1.0] * len(result_lists)
        elif len(weights) != len(result_lists):
            raise ValueError("Number of weights must match number of result lists")
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Calculate Borda scores
        borda_scores = {}
        
        for results, weight in zip(result_lists, weights):
            # Sort by score
            sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
            n = len(sorted_results)
            
            # Assign Borda points (n - rank)
            for rank, (doc_id, _) in enumerate(sorted_results):
                points = (n - rank) * weight
                
                if doc_id in borda_scores:
                    borda_scores[doc_id] += points
                else:
                    borda_scores[doc_id] = points
        
        # Sort by Borda score and return top-k
        fused_results = sorted(borda_scores.items(), key=lambda x: x[1], reverse=True)
        return fused_results[:top_k]
    
    def adaptive_fusion(
        self,
        result_lists: List[List[Tuple[str, float]]],
        query_type: str = "general",
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Adaptive fusion that selects the best method based on query type.
        
        Args:
            result_lists: List of result lists, each containing (doc_id, score) tuples
            query_type: Type of query (entity, philosophical, narrative, etc.)
            top_k: Number of top results to return
            
        Returns:
            Fused results as (doc_id, score) tuples
        """
        if not result_lists:
            return []
        
        # Choose fusion method based on query type
        if query_type == "entity":
            # Entity queries benefit from RRF
            return self.reciprocal_rank_fusion(result_lists, top_k)
        
        elif query_type == "philosophical":
            # Philosophical queries benefit from weighted score fusion
            weights = [0.3, 0.7] if len(result_lists) == 2 else None  # Favor dense retrieval
            return self.weighted_score_fusion(result_lists, top_k, weights)
        
        elif query_type == "narrative":
            # Narrative queries benefit from Borda count
            return self.borda_count_fusion(result_lists, top_k)
        
        elif query_type == "comparative":
            # Comparative queries benefit from Condorcet
            return self.condorcet_fusion(result_lists, top_k)
        
        else:
            # Default to RRF for general queries
            return self.reciprocal_rank_fusion(result_lists, top_k)
    
    def merge_with_metadata(
        self,
        fused_results: List[Tuple[str, float]],
        documents: Dict[str, Dict],
        include_individual_scores: bool = False,
        individual_results: Optional[Dict[str, List[Tuple[str, float]]]] = None,
    ) -> List[Dict]:
        """Merge fused results with document metadata.
        
        Args:
            fused_results: List of (doc_id, fused_score) tuples
            documents: Dictionary mapping doc_id to document data
            include_individual_scores: Whether to include individual retrieval scores
            individual_results: Individual results from each retriever
            
        Returns:
            List of documents with metadata and scores
        """
        merged_results = []
        
        for doc_id, fused_score in fused_results:
            if doc_id in documents:
                doc_data = documents[doc_id].copy()
                doc_data["fused_score"] = fused_score
                
                # Add individual scores if requested
                if include_individual_scores and individual_results:
                    doc_data["individual_scores"] = {}
                    for retriever_name, results in individual_results.items():
                        for result_doc_id, score in results:
                            if result_doc_id == doc_id:
                                doc_data["individual_scores"][retriever_name] = score
                                break
                
                merged_results.append(doc_data)
        
        return merged_results
    
    def evaluate_fusion(
        self,
        fused_results: List[Tuple[str, float]],
        ground_truth: List[str],
        k_values: List[int] = [1, 3, 5, 10],
    ) -> Dict[str, float]:
        """Evaluate fusion results using precision@k.
        
        Args:
            fused_results: List of (doc_id, fused_score) tuples
            ground_truth: List of relevant document IDs
            k_values: Values of k for precision@k calculation
            
        Returns:
            Dictionary with precision@k scores
        """
        # Get ranked document IDs
        ranked_docs = [doc_id for doc_id, _ in fused_results]
        relevant_set = set(ground_truth)
        
        # Calculate precision@k
        precision_scores = {}
        for k in k_values:
            if k <= len(ranked_docs):
                top_k_docs = ranked_docs[:k]
                relevant_in_top_k = len(set(top_k_docs).intersection(relevant_set))
                precision_scores[f"precision@{k}"] = relevant_in_top_k / k
            else:
                precision_scores[f"precision@{k}"] = 0.0
        
        # Calculate MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for i, doc_id in enumerate(ranked_docs):
            if doc_id in relevant_set:
                mrr = 1.0 / (i + 1)
                break
        precision_scores["mrr"] = mrr
        
        return precision_scores
