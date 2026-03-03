"""Query classification for the Mahabharata RAG system.

This module implements query classification to determine the optimal
retrieval strategy based on query type and characteristics.
"""

import logging
import re
from enum import Enum
from typing import Dict, List, Optional, Tuple

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries in the Mahabharata context."""
    ENTITY = "entity"  # Queries about specific characters, places, weapons
    PHILOSOPHICAL = "philosophical"  # Questions about dharma, karma, philosophy
    NARRATIVE = "narrative"  # Questions about story events, sequences
    CONCEPTUAL = "conceptual"  # Abstract concepts, relationships
    TEMPORAL = "temporal"  # Questions about time, sequence, duration
    COMPARATIVE = "comparative"  # Comparisons between entities/events
    UNKNOWN = "unknown"  # Cannot determine type


class QueryClassifier:
    """Classifies queries to determine optimal retrieval strategy."""
    
    def __init__(
        self,
        embedding_model: Optional[SentenceTransformer] = None,
        entity_keywords: Optional[Dict[str, List[str]]] = None,
    ):
        """Initialize the query classifier.
        
        Args:
            embedding_model: Sentence transformer for semantic similarity
            entity_keywords: Dictionary of entity types and their keywords
        """
        self.embedding_model = embedding_model or SentenceTransformer(
            "BAAI/bge-base-en-v1.5"
        )
        
        # Mahabharata-specific entity keywords
        self.entity_keywords = entity_keywords or {
            "characters": [
                "arjuna", "krishna", "yudhishthira", "bhima", "nakula", "sahadeva",
                "duryodhana", "dushasana", "karna", "bhishma", "drona", "kripa",
                "ashwatthama", "shakuni", "gandhari", "dhritarashtra", "kunti",
                "draupadi", "subhadra", "abhimanyu", "ghatotkacha", "vrikodara",
                "pandu", "madri", "satyavati", "vyasa", "sanjaya", "vidura"
            ],
            "places": [
                "hastinapura", "indraprastha", "kurukshetra", "dwarka", "mathura",
                "varanavata", "panchala", "karnapravarohana", "vrikotprastha",
                "ganga", "yamuna", "saraswati", "himalayas", "vardhamana"
            ],
            "weapons": [
                "gandiva", "brahmastra", "pashupatastra", "narayanastra", "vaishnavastra",
                "brahmashirsha", "vajra", "shakti", "chakra", "mace", "gada",
                "bow", "arrow", "sword", "khadga", "parashu"
            ],
            "concepts": [
                "dharma", "karma", "moksha", "yoga", "vedanta", "bhagavad gita",
                "duty", "righteousness", "fate", "destiny", "war", "peace",
                "justice", "truth", "non-violence", "sacrifice", "devotion"
            ]
        }
        
        # Query type patterns
        self.patterns = {
            QueryType.ENTITY: [
                r"who (?:is|was) (\w+)",
                r"tell me about (\w+)",
                r"what (?:is|was) (\w+)'s",
                r"describe (\w+)",
                r"information about (\w+)",
                r"(\w+)'s (?:role|character|story)"
            ],
            QueryType.PHILOSOPHICAL: [
                r"what (?:is|was) (?:the )?(?:concept of )?(dharma|karma|moksha|yoga)",
                r"explain (?:the )?(?:concept of )?(dharma|karma|moksha|yoga)",
                r"meaning of (?:dharma|karma|moksha|yoga)",
                r"philosophy (?:of|in|about)",
                r"teachings (?:of|about)",
                r"principles (?:of|for)"
            ],
            QueryType.NARRATIVE: [
                r"what (?:happened|occurred)",
                r"how (?:did|do|does)",
                r"when (?:did|do|does)",
                r"where (?:did|do|does)",
                r"story (?:of|about)",
                r"narrative (?:of|about)",
                r"events (?:of|leading to|during)",
                r"battle (?:of|at|in)"
            ],
            QueryType.TEMPORAL: [
                r"when (?:did|was)",
                r"how long",
                r"duration (?:of|for)",
                r"time (?:period|frame)",
                r"before (?:the|after)",
                r"after (?:the|before)",
                r"during (?:the)",
                r"sequence (?:of|in)"
            ],
            QueryType.COMPARATIVE: [
                r"compare (\w+) and (\w+)",
                r"difference (?:between|of)",
                r"better (?:than|between)",
                r"versus|vs",
                r"which (?:is|was) (?:better|worse|stronger|weaker)",
                r"similarities (?:between|of)"
            ]
        }
        
        # Embeddings for semantic query type classification
        self._initialize_type_embeddings()
    
    def _initialize_type_embeddings(self):
        """Initialize embeddings for semantic query type classification."""
        type_examples = {
            QueryType.ENTITY: [
                "Who is Arjuna?",
                "Tell me about Krishna",
                "What was Karna's role?",
                "Describe Bhishma",
                "Information about Draupadi"
            ],
            QueryType.PHILOSOPHICAL: [
                "What is dharma?",
                "Explain the concept of karma",
                "Philosophy in Bhagavad Gita",
                "Teachings about duty",
                "Principles of righteousness"
            ],
            QueryType.NARRATIVE: [
                "What happened in the Kurukshetra war?",
                "How did the dice game occur?",
                "Story of the Pandavas",
                "Events leading to the war",
                "Battle at Kurukshetra"
            ],
            QueryType.CONCEPTUAL: [
                "Relationship between dharma and karma",
                "Concept of duty in Mahabharata",
                "Abstract principles",
                "Theoretical framework",
                "Philosophical underpinnings"
            ],
            QueryType.TEMPORAL: [
                "When did the war happen?",
                "How long did the exile last?",
                "Time period of Mahabharata",
                "Sequence of events",
                "Duration of the battle"
            ],
            QueryType.COMPARATIVE: [
                "Compare Arjuna and Karna",
                "Difference between Pandavas and Kauravas",
                "Arjuna vs Karna",
                "Who was stronger: Bhima or Duryodhana?",
                "Similarities between Krishna and Balarama"
            ]
        }
        
        # Compute embeddings for each query type
        self.type_embeddings = {}
        for query_type, examples in type_examples.items():
            embeddings = self.embedding_model.encode(examples)
            self.type_embeddings[query_type] = np.mean(embeddings, axis=0)
    
    def classify_query(self, query: str) -> Tuple[QueryType, Dict[str, float]]:
        """Classify a query into one of the predefined types.
        
        Args:
            query: The input query string
            
        Returns:
            Tuple of (QueryType, confidence_scores)
        """
        query = query.lower().strip()
        confidence_scores = {}
        
        # Pattern-based classification
        pattern_scores = self._pattern_based_classification(query)
        
        # Keyword-based classification
        keyword_scores = self._keyword_based_classification(query)
        
        # Semantic similarity-based classification
        semantic_scores = self._semantic_classification(query)
        
        # Combine scores
        for query_type in QueryType:
            if query_type == QueryType.UNKNOWN:
                continue
            
            confidence_scores[query_type.value] = (
                0.4 * pattern_scores.get(query_type, 0.0) +
                0.3 * keyword_scores.get(query_type, 0.0) +
                0.3 * semantic_scores.get(query_type, 0.0)
            )
        
        # Determine the most likely type
        if not confidence_scores or max(confidence_scores.values()) < 0.3:
            return QueryType.UNKNOWN, confidence_scores
        
        best_type = max(confidence_scores.items(), key=lambda x: x[1])
        return QueryType(best_type[0]), confidence_scores
    
    def _pattern_based_classification(self, query: str) -> Dict[QueryType, float]:
        """Classify query based on regex patterns."""
        scores = {}
        
        for query_type, patterns in self.patterns.items():
            score = 0.0
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    score += 1.0
            scores[query_type] = min(score / len(patterns), 1.0)
        
        return scores
    
    def _keyword_based_classification(self, query: str) -> Dict[QueryType, float]:
        """Classify query based on keyword matching."""
        scores = {}
        query_tokens = set(query.split())
        
        # Entity queries
        entity_score = 0.0
        for entity_type, keywords in self.entity_keywords.items():
            matches = len(query_tokens.intersection(set(keywords)))
            if matches > 0:
                entity_score = max(entity_score, matches / len(keywords))
        scores[QueryType.ENTITY] = entity_score
        
        # Philosophical queries
        phil_keywords = self.entity_keywords["concepts"]
        phil_matches = len(query_tokens.intersection(set(phil_keywords)))
        scores[QueryType.PHILOSOPHICAL] = phil_matches / len(phil_keywords)
        
        # Narrative queries
        narrative_keywords = ["story", "happened", "occurred", "event", "battle", "war"]
        narrative_matches = len(query_tokens.intersection(set(narrative_keywords)))
        scores[QueryType.NARRATIVE] = narrative_matches / len(narrative_keywords)
        
        # Temporal queries
        temporal_keywords = ["when", "how long", "duration", "time", "before", "after", "during"]
        temporal_matches = len(query_tokens.intersection(set(temporal_keywords)))
        scores[QueryType.TEMPORAL] = temporal_matches / len(temporal_keywords)
        
        # Comparative queries
        comparative_keywords = ["compare", "difference", "versus", "vs", "better", "similarities"]
        comparative_matches = len(query_tokens.intersection(set(comparative_keywords)))
        scores[QueryType.COMPARATIVE] = comparative_matches / len(comparative_keywords)
        
        # Conceptual queries (abstract)
        conceptual_keywords = ["concept", "relationship", "principle", "theory", "framework", "meaning"]
        conceptual_matches = len(query_tokens.intersection(set(conceptual_keywords)))
        scores[QueryType.CONCEPTUAL] = conceptual_matches / len(conceptual_keywords)
        
        return scores
    
    def _semantic_classification(self, query: str) -> Dict[QueryType, float]:
        """Classify query using semantic similarity."""
        query_embedding = self.embedding_model.encode([query])[0]
        scores = {}
        
        for query_type, type_embedding in self.type_embeddings.items():
            # Ensure both are numpy arrays
            if isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding)
            if isinstance(type_embedding, list):
                type_embedding = np.array(type_embedding)
            
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                type_embedding.reshape(1, -1)
            )[0][0]
            scores[query_type] = max(similarity, 0.0)
        
        return scores
    
    def get_retrieval_strategy(self, query: str) -> Dict[str, any]:
        """Get the optimal retrieval strategy for a query.
        
        Args:
            query: The input query string
            
        Returns:
            Dictionary containing retrieval strategy parameters
        """
        query_type, confidence_scores = self.classify_query(query)
        
        # Default strategy
        strategy = {
            "query_type": query_type.value,
            "confidence": confidence_scores.get(query_type.value, 0.0),
            "use_bm25": True,
            "use_dense": True,
            "bm25_weight": 0.5,
            "dense_weight": 0.5,
            "rerank": True,
            "top_k": 10,
            "rerank_top_k": 5,
            "query_expansion": False,
            "hyde": False
        }
        
        # Adjust strategy based on query type
        if query_type == QueryType.ENTITY:
            # Entity queries benefit more from BM25
            strategy["bm25_weight"] = 0.7
            strategy["dense_weight"] = 0.3
            strategy["query_expansion"] = True
            
        elif query_type == QueryType.PHILOSOPHICAL:
            # Philosophical queries benefit from semantic search and HyDE
            strategy["bm25_weight"] = 0.3
            strategy["dense_weight"] = 0.7
            strategy["hyde"] = True
            strategy["top_k"] = 15
            
        elif query_type == QueryType.NARRATIVE:
            # Narrative queries need balanced approach
            strategy["bm25_weight"] = 0.5
            strategy["dense_weight"] = 0.5
            strategy["rerank_top_k"] = 7
            
        elif query_type == QueryType.CONCEPTUAL:
            # Conceptual queries rely heavily on semantic understanding
            strategy["bm25_weight"] = 0.2
            strategy["dense_weight"] = 0.8
            strategy["hyde"] = True
            strategy["top_k"] = 20
            
        elif query_type == QueryType.TEMPORAL:
            # Temporal queries need precise matching
            strategy["bm25_weight"] = 0.6
            strategy["dense_weight"] = 0.4
            
        elif query_type == QueryType.COMPARATIVE:
            # Comparative queries need comprehensive retrieval
            strategy["top_k"] = 15
            strategy["rerank_top_k"] = 8
            strategy["query_expansion"] = True
        
        return strategy
