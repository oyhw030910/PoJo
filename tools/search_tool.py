"""Search Tool Module.

Provides search and information retrieval tools for LLM agents.
"""

import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import re


@dataclass
class SearchResult:
    """A search result item.

    Attributes:
        title: Result title
        content: Result content/snippet
        url: Source URL
        score: Relevance score
    """
    title: str
    content: str
    url: Optional[str] = None
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeBase:
    """Simple in-memory knowledge base for search."""

    def __init__(self):
        """Initialize knowledge base."""
        self._documents: List[Dict[str, Any]] = []
        self._index: Dict[str, List[int]] = {}

    def add_document(
        self,
        content: str,
        title: str = "",
        url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Add a document to the knowledge base.

        Args:
            content: Document content
            title: Document title
            url: Optional URL
            metadata: Optional metadata

        Returns:
            Document ID
        """
        doc_id = len(self._documents)

        self._documents.append({
            "id": doc_id,
            "title": title,
            "content": content,
            "url": url,
            "metadata": metadata or {},
        })

        # Index words
        words = re.findall(r'\w+', content.lower())
        for word in set(words):
            if word not in self._index:
                self._index[word] = []
            self._index[word].append(doc_id)

        return doc_id

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[SearchResult]:
        """Search the knowledge base.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of search results
        """
        query_words = set(re.findall(r'\w+', query.lower()))

        # Score documents
        doc_scores: Dict[int, float] = {}

        for word in query_words:
            if word in self._index:
                for doc_id in self._index[word]:
                    if doc_id not in doc_scores:
                        doc_scores[doc_id] = 0
                    doc_scores[doc_id] += 1

        # Sort by score
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Build results
        results = []
        for doc_id, score in sorted_docs[:top_k]:
            doc = self._documents[doc_id]
            results.append(SearchResult(
                title=doc["title"],
                content=doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                url=doc["url"],
                score=float(score),
                metadata=doc["metadata"],
            ))

        return results

    def clear(self) -> None:
        """Clear the knowledge base."""
        self._documents = []
        self._index = {}


class WebSearchSimulator:
    """Simulates web search for environments without internet access."""

    def __init__(self):
        """Initialize web search simulator."""
        self._kb = KnowledgeBase()
        self._setup_default_knowledge()

    def _setup_default_knowledge(self) -> None:
        """Setup default knowledge for simulation."""
        # Add some general knowledge
        self._kb.add_document(
            content="""Python is a high-level programming language known for its simplicity
            and readability. It supports multiple programming paradigms including procedural,
            object-oriented, and functional programming.""",
            title="Python Programming Language",
            metadata={"category": "programming"}
        )

        self._kb.add_document(
            content="""Reinforcement Learning is a type of machine learning where an agent
            learns to make decisions by interacting with an environment. The agent receives
            rewards or penalties based on its actions and learns to maximize cumulative reward.""",
            title="Reinforcement Learning",
            metadata={"category": "machine learning"}
        )

        self._kb.add_document(
            content="""Large Language Models (LLMs) are AI models trained on vast amounts of
            text data. They can generate human-like text, answer questions, translate languages,
            and perform various natural language processing tasks.""",
            title="Large Language Models",
            metadata={"category": "artificial intelligence"}
        )

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search simulated web.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of search results
        """
        return self._kb.search(query, top_k)

    def add_knowledge(
        self,
        content: str,
        title: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add knowledge to the simulator.

        Args:
            content: Knowledge content
            title: Title
            metadata: Optional metadata
        """
        self._kb.add_document(content, title, metadata=metadata)


# Global search simulator
_search_simulator = WebSearchSimulator()


def register_tool(name=None, description=None, category="general", examples=None):
    """Placeholder decorator for tool registration."""
    def decorator(func):
        return func
    return decorator


@register_tool(
    name="search",
    description="Search for information on a topic",
    category="search",
    examples=["search(query='Python programming language')"]
)
def search(
    query: str,
    top_k: int = 5,
    use_web: bool = False
) -> Dict[str, Any]:
    """Search for information.

    Args:
        query: Search query
        top_k: Number of results to return
        use_web: Whether to use web search (if available)

    Returns:
        Dictionary with search results
    """
    results = _search_simulator.search(query, top_k)

    return {
        "query": query,
        "results": [
            {
                "title": r.title,
                "content": r.content,
                "score": r.score,
            }
            for r in results
        ],
        "count": len(results),
    }


@register_tool(
    name="lookup",
    description="Look up a specific term or concept",
    category="search",
    examples=["lookup(term='reinforcement learning')"]
)
def lookup(term: str) -> Dict[str, Any]:
    """Look up a term.

    Args:
        term: Term to look up

    Returns:
        Dictionary with definition/information
    """
    results = _search_simulator.search(f"what is {term}", top_k=1)

    if results:
        return {
            "term": term,
            "definition": results[0].content,
            "found": True,
        }

    return {
        "term": term,
        "definition": None,
        "found": False,
        "message": f"No information found for '{term}'",
    }


@register_tool(
    name="add_knowledge",
    description="Add new knowledge to the search system",
    category="search",
    examples=["add_knowledge(content='...', title='Example')"]
)
def add_knowledge(
    content: str,
    title: str = "",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Add knowledge to the system.

    Args:
        content: Knowledge content
        title: Title
        metadata: Optional metadata

    Returns:
        Status dictionary
    """
    doc_id = _search_simulator._kb.add_document(content, title, metadata=metadata)

    return {
        "success": True,
        "document_id": doc_id,
        "message": f"Added document: {title or 'Untitled'}",
    }


@register_tool(
    name="calculate",
    description="Perform mathematical calculations",
    category="search",
    examples=["calculate(expression='2 + 2 * 3')"]
)
def calculate(expression: str) -> Dict[str, Any]:
    """Calculate a mathematical expression.

    Args:
        expression: Mathematical expression

    Returns:
        Dictionary with result
    """
    try:
        # Safe evaluation
        result = eval(expression, {"__builtins__": {}}, {"math": __import__("math")})
        return {
            "expression": expression,
            "result": result,
            "success": True,
        }
    except Exception as e:
        return {
            "expression": expression,
            "error": str(e),
            "success": False,
        }


@register_tool(
    name="fact_check",
    description="Verify a fact or statement",
    category="search",
    examples=["fact_check(statement='Python is a programming language')"]
)
def fact_check(statement: str) -> Dict[str, Any]:
    """Fact check a statement.

    Args:
        statement: Statement to verify

    Returns:
        Dictionary with verification results
    """
    # Extract key terms from statement
    words = re.findall(r'\w+', statement.lower())
    key_terms = [w for w in words if len(w) > 3][:3]

    if not key_terms:
        return {
            "statement": statement,
            "verdict": "unknown",
            "confidence": 0.0,
            "reason": "Could not extract key terms",
        }

    # Search for supporting evidence
    query = " ".join(key_terms)
    results = _search_simulator.search(query, top_k=3)

    if results:
        # Simple heuristic: if we found relevant results, likely true
        avg_score = sum(r.score for r in results) / len(results)
        confidence = min(1.0, avg_score / 3)

        return {
            "statement": statement,
            "verdict": "likely_true" if confidence > 0.5 else "uncertain",
            "confidence": confidence,
            "evidence": [r.content for r in results],
        }

    return {
        "statement": statement,
        "verdict": "unknown",
        "confidence": 0.0,
        "reason": "No supporting evidence found",
    }
