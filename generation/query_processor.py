from typing import List, Dict, Any
import logging
import re

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Process and normalize queries before retrieval"""

    def __init__(self, embedding_manager=None):
        self.embedding_manager = embedding_manager

    def preprocess(self, query: str) -> str:
        """Clean and normalize query for retrieval"""
        query = query.strip()
        query = re.sub(r"\s+", " ", query)
        query = re.sub(r"[^\w\s\?\!\.\,\-]", "", query)
        return query

    def expand(self, query: str) -> List[str]:
        """Expand query with simple variations"""
        variations = [query]

        if "?" in query:
            base = query.replace("?", "").strip()
            variations.append(base)

        if not query[0].isupper():
            variations.append(query.capitalize())

        return list(set(variations))

    def extract_keywords(self, query: str) -> List[str]:
        """Extract key terms from query"""
        stopwords = {
            "what", "is", "the", "a", "an", "how", "does", "do", "are",
            "can", "could", "would", "should", "will", "and", "or", "of",
            "in", "on", "at", "to", "for", "with", "by", "about", "from",
        }
        words = re.findall(r"\b[a-zA-Z]{2,}\b", query.lower())
        return [w for w in words if w not in stopwords]

    def enhance(self, query: str) -> Dict[str, Any]:
        """Full query enhancement"""
        enhanced = {
            "original": query,
            "processed": self.preprocess(query),
            "variations": self.expand(query),
            "keywords": self.extract_keywords(query),
        }

        if self.embedding_manager:
            try:
                enhanced["embedding"] = self.embedding_manager.embed_query(query)
            except Exception as e:
                logger.warning(f"Could not generate embedding: {e}")

        return enhanced
