"""Memory Manager Module.

Implements memory management for LLM agents including short-term,
long-term, and working memory.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import deque
import json


@dataclass
class MemoryConfig:
    """Configuration for memory manager.

    Attributes:
        short_term_size: Maximum size of short-term memory
        long_term_enabled: Whether long-term memory is enabled
        long_term_top_k: Number of items to retrieve from long-term
        embedding_model: Model for embeddings
        vector_store: Vector store type ('chroma', 'faiss')
        working_memory_capacity: Working memory capacity
    """
    short_term_size: int = 10
    long_term_enabled: bool = True
    long_term_top_k: int = 5
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_store: str = "chroma"
    working_memory_capacity: int = 5


@dataclass
class MemoryItem:
    """A single memory item.

    Attributes:
        content: Content of the memory
        embedding: Optional embedding vector
        metadata: Additional metadata
        importance: Importance score
        timestamp: Creation timestamp
    """
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 1.0
    timestamp: int = 0


class ShortTermMemory:
    """Short-term memory using deque for recent items.

    Stores recent observations, actions, and rewards for context.
    """

    def __init__(self, max_size: int = 10):
        """Initialize short-term memory.

        Args:
            max_size: Maximum number of items to store
        """
        self.max_size = max_size
        self._memory: deque = deque(maxlen=max_size)

    def add(
        self,
        observation: Optional[str] = None,
        action: Optional[str] = None,
        reward: Optional[float] = None,
        **kwargs
    ) -> None:
        """Add an item to short-term memory.

        Args:
            observation: Environment observation
            action: Agent action
            reward: Reward received
            **kwargs: Additional data
        """
        item = {
            "observation": observation,
            "action": action,
            "reward": reward,
            **kwargs,
        }
        self._memory.append(item)

    def get_recent(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get recent items.

        Args:
            n: Number of items (None for all)

        Returns:
            List of memory items
        """
        if n is None:
            return list(self._memory)
        return list(self._memory)[-n:]

    def format_for_prompt(self) -> str:
        """Format memory as prompt text.

        Returns:
            Formatted string
        """
        if not self._memory:
            return "No recent history."

        lines = ["Recent history:"]
        for i, item in enumerate(self._memory):
            parts = []
            if item.get("observation"):
                parts.append(f"Obs: {item['observation'][:100]}...")
            if item.get("action"):
                parts.append(f"Act: {item['action'][:50]}...")
            if item.get("reward") is not None:
                parts.append(f"Reward: {item['reward']:.3f}")
            lines.append(f"  {i+1}. " + " | ".join(parts))

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all items."""
        self._memory.clear()

    def __len__(self) -> int:
        return len(self._memory)


class LongTermMemory:
    """Long-term memory with vector storage for retrieval.

    Stores important experiences for later retrieval.
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_store: str = "chroma",
        top_k: int = 5
    ):
        """Initialize long-term memory.

        Args:
            embedding_model: Embedding model name
            vector_store: Vector store type
            top_k: Default number of items to retrieve
        """
        self.embedding_model = embedding_model
        self.vector_store_type = vector_store
        self.top_k = top_k
        self._embedding_fn = None
        self._vector_store = None
        self._items: List[MemoryItem] = []
        self._initialized = False

    def _initialize(self) -> None:
        """Initialize embedding model and vector store."""
        if self._initialized:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_fn = SentenceTransformer(self.embedding_model)
            self._initialized = True
        except ImportError:
            print("sentence-transformers not available. Using keyword-based retrieval.")
            self._embedding_fn = None

        # Initialize vector store
        if self.vector_store_type == "chroma":
            try:
                import chromadb
                client = chromadb.Client()
                self._vector_store = client.create_collection("long_term_memory")
            except ImportError:
                self._vector_store = None
        elif self.vector_store_type == "faiss":
            try:
                import faiss
                self._vector_store = {"index": None, "items": []}
            except ImportError:
                self._vector_store = None

    def add(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 1.0
    ) -> None:
        """Add an item to long-term memory.

        Args:
            content: Content to store
            metadata: Optional metadata
            importance: Importance score
        """
        self._initialize()

        item = MemoryItem(
            content=content,
            metadata=metadata or {},
            importance=importance,
            timestamp=len(self._items),
        )

        # Compute embedding
        if self._embedding_fn is not None:
            item.embedding = self._embedding_fn.encode(content)

        # Store in vector store
        if self._vector_store is not None and item.embedding is not None:
            try:
                self._vector_store.add(
                    embeddings=[item.embedding.tolist()],
                    documents=[content],
                    metadatas=[{"importance": importance, "id": len(self._items)}],
                )
            except Exception:
                pass

        self._items.append(item)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[MemoryItem]:
        """Retrieve relevant memories.

        Args:
            query: Query string
            top_k: Number of items to retrieve

        Returns:
            List of relevant memory items
        """
        self._initialize()

        if top_k is None:
            top_k = self.top_k

        # Use vector store if available
        if self._vector_store is not None and self._embedding_fn is not None:
            query_embedding = self._embedding_fn.encode(query)
            try:
                results = self._vector_store.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=top_k,
                )
                # Map back to MemoryItems
                if results and results.get("documents"):
                    indices = results.get("metadatas", [{}])[0]
                    return [self._items[i.get("id", 0)] for i in indices if "id" in i]
            except Exception:
                pass

        # Fallback: keyword-based retrieval
        return self._keyword_search(query, top_k)

    def _keyword_search(self, query: str, top_k: int) -> List[MemoryItem]:
        """Simple keyword-based search.

        Args:
            query: Query string
            top_k: Number of items

        Returns:
            List of relevant items
        """
        query_words = set(query.lower().split())

        scored_items = []
        for item in self._items:
            content_words = set(item.content.lower().split())
            overlap = len(query_words & content_words)
            score = overlap * item.importance
            scored_items.append((score, item))

        scored_items.sort(reverse=True)
        return [item for _, item in scored_items[:top_k]]

    def get_all(self) -> List[MemoryItem]:
        """Get all memory items.

        Returns:
            List of all items
        """
        return self._items

    def clear(self) -> None:
        """Clear all memories."""
        self._items = []
        if self._vector_store is not None:
            try:
                import chromadb
                if isinstance(self._vector_store, chromadb.api.models.Collection.Collection):
                    self._vector_store = self._vector_store.collection.client.create_collection(
                        "long_term_memory"
                    )
            except Exception:
                self._vector_store = None


class WorkingMemory:
    """Working memory for current task context.

    Holds information relevant to the current task.
    """

    def __init__(self, capacity: int = 5):
        """Initialize working memory.

        Args:
            capacity: Maximum number of items
        """
        self.capacity = capacity
        self._context: Dict[str, Any] = {}
        self._items: List[Dict[str, Any]] = []

    def set_context(self, key: str, value: Any) -> None:
        """Set a context value.

        Args:
            key: Context key
            value: Context value
        """
        self._context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a context value.

        Args:
            key: Context key
            default: Default value

        Returns:
            Context value
        """
        return self._context.get(key, default)

    def add_item(self, item: Dict[str, Any]) -> None:
        """Add an item to working memory.

        Args:
            item: Item dictionary
        """
        self._items.append(item)
        if len(self._items) > self.capacity:
            self._items.pop(0)

    def get_items(self) -> List[Dict[str, Any]]:
        """Get all working memory items.

        Returns:
            List of items
        """
        return self._items

    def clear(self) -> None:
        """Clear working memory."""
        self._context = {}
        self._items = []

    def format_for_prompt(self) -> str:
        """Format working memory as prompt.

        Returns:
            Formatted string
        """
        lines = ["Current context:"]

        for key, value in self._context.items():
            lines.append(f"  {key}: {value}")

        if self._items:
            lines.append("Working items:")
            for item in self._items:
                lines.append(f"  - {json.dumps(item)}")

        return "\n".join(lines)


class MemoryManager:
    """Main memory manager that coordinates all memory types.

    Provides unified interface for memory operations.
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        """Initialize memory manager.

        Args:
            config: Memory configuration
        """
        self.config = config or MemoryConfig()

        # Initialize memory components
        self.short_term = ShortTermMemory(
            max_size=self.config.short_term_size
        )
        self.long_term = LongTermMemory(
            embedding_model=self.config.embedding_model,
            vector_store=self.config.vector_store,
            top_k=self.config.long_term_top_k,
        )
        self.working = WorkingMemory(
            capacity=self.config.working_memory_capacity,
        )

    def add_experience(
        self,
        observation: str,
        action: str,
        reward: float,
        store_long_term: bool = True,
        importance: Optional[float] = None
    ) -> None:
        """Add an experience to memory.

        Args:
            observation: Environment observation
            action: Agent action
            reward: Reward received
            store_long_term: Whether to store in long-term memory
            importance: Importance score for long-term storage
        """
        # Add to short-term memory
        self.short_term.add(
            observation=observation,
            action=action,
            reward=reward,
        )

        # Add to long-term memory if important
        if store_long_term and self.config.long_term_enabled:
            if importance is None:
                # Compute importance based on reward
                importance = abs(reward)

            if importance > 0.5:  # Threshold for storage
                content = f"Obs: {observation} | Act: {action} | Reward: {reward}"
                self.long_term.add(
                    content=content,
                    metadata={"action": action, "reward": reward},
                    importance=importance,
                )

    def get_context(
        self,
        query: Optional[str] = None,
        include_short_term: bool = True,
        include_long_term: bool = True,
        num_recent: Optional[int] = None
    ) -> str:
        """Get memory context for decision making.

        Args:
            query: Optional query for retrieval
            include_short_term: Whether to include short-term memory
            include_long_term: Whether to include long-term memory
            num_recent: Number of recent items to include

        Returns:
            Formatted context string
        """
        parts = []

        # Add short-term memory
        if include_short_term:
            if num_recent is not None:
                recent = self.short_term.get_recent(num_recent)
                if recent:
                    parts.append("Recent experiences:")
                    for item in recent:
                        part = []
                        if item.get("observation"):
                            part.append(f"O: {item['observation'][:50]}")
                        if item.get("action"):
                            part.append(f"A: {item['action'][:30]}")
                        if item.get("reward") is not None:
                            part.append(f"R: {item['reward']:.2f}")
                        parts.append("  " + " | ".join(part))
            else:
                st_formatted = self.short_term.format_for_prompt()
                if st_formatted:
                    parts.append(st_formatted)

        # Add long-term memory
        if include_long_term and self.config.long_term_enabled:
            if query:
                memories = self.long_term.retrieve(query, top_k=self.config.long_term_top_k)
            else:
                memories = self.long_term.get_all()[:self.config.long_term_top_k]

            if memories:
                parts.append("Relevant memories:")
                for mem in memories:
                    parts.append(f"  - {mem.content[:100]}... (importance: {mem.importance:.2f})")

        # Add working memory
        working_formatted = self.working.format_for_prompt()
        if working_formatted != "Current context:":
            parts.append(working_formatted)

        return "\n".join(parts) if parts else "No memory context available."

    def set_task_context(self, task_description: str, task_goal: str) -> None:
        """Set current task context.

        Args:
            task_description: Description of current task
            task_goal: Goal of current task
        """
        self.working.set_context("task_description", task_description)
        self.working.set_context("task_goal", task_goal)

    def add_note_to_context(self, note: str) -> None:
        """Add a note to working memory.

        Args:
            note: Note to add
        """
        self.working.add_item({"type": "note", "content": note})

    def get_relevant_memories(self, query: str) -> List[MemoryItem]:
        """Get relevant memories for a query.

        Args:
            query: Query string

        Returns:
            List of relevant memories
        """
        return self.long_term.retrieve(query)

    def clear(self) -> None:
        """Clear all memories."""
        self.short_term.clear()
        self.long_term.clear()
        self.working.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            "short_term_size": len(self.short_term),
            "long_term_size": len(self.long_term.get_all()),
            "working_items": len(self.working.get_items()),
        }
