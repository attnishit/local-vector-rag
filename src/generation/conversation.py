"""
Conversation History Management

Author: Nishit Attrey

This module manages multi-turn conversation history for chat-based RAG interactions.
It handles conversation context, automatic pruning based on token limits, and
session persistence.

Key Features:
- Multi-turn conversation tracking
- Automatic context window management
- Token-based pruning (configurable limits)
- Session persistence (save/load from JSON)
- Message role management (user/assistant)

Example:
    >>> from src.generation.conversation import ConversationHistory
    >>>
    >>> conv = ConversationHistory(max_tokens=4096, max_turns=10)
    >>> conv.add_turn("What is HNSW?", "HNSW is a graph-based algorithm...")
    >>> conv.add_turn("How does it work?", "It builds a hierarchical graph...")
    >>>
    >>> history = conv.get_history()
    >>> print(f"Total turns: {len(history)}")
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ConversationHistory:
    """
    Manages conversation history with automatic pruning and persistence.

    Tracks user queries and assistant responses in a multi-turn conversation.
    Automatically prunes old messages when exceeding token or turn limits.

    Example:
        >>> conv = ConversationHistory(max_tokens=4096, max_turns=10)
        >>> conv.add_turn("What is vector search?", "Vector search is...")
        >>> conv.add_turn("Tell me more", "It uses embeddings...")
        >>> print(conv.get_formatted_history())
    """

    def __init__(
        self,
        max_tokens: int = 4096,
        max_turns: int = 10,
        session_id: Optional[str] = None,
    ):
        """
        Initialize conversation history.

        Args:
            max_tokens: Maximum total tokens to keep (approximate)
            max_turns: Maximum number of turns to keep
            session_id: Unique session identifier for persistence
        """
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self.session_id = session_id or self._generate_session_id()

        self.history: List[Dict[str, str]] = []
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

        logger.debug(
            f"Initialized conversation: session_id={self.session_id}, "
            f"max_tokens={max_tokens}, max_turns={max_turns}"
        )

    def _generate_session_id(self) -> str:
        """Generate unique session ID based on timestamp."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def add_turn(self, user_query: str, assistant_response: str) -> None:
        """
        Add a conversation turn (user query + assistant response).

        Args:
            user_query: User's question or message
            assistant_response: Assistant's response

        Example:
            >>> conv.add_turn("What is HNSW?", "HNSW is a graph-based algorithm...")
        """
        # Add user message
        self.history.append({
            "role": "user",
            "content": user_query,
        })

        # Add assistant message
        self.history.append({
            "role": "assistant",
            "content": assistant_response,
        })

        self.updated_at = datetime.now()

        # Prune if necessary
        self._prune_if_needed()

        logger.debug(
            f"Added turn to conversation {self.session_id}. "
            f"Total messages: {len(self.history)}"
        )

    def get_history(self) -> List[Dict[str, str]]:
        """
        Get conversation history as list of message dicts.

        Returns:
            List of messages with 'role' and 'content' keys

        Example:
            >>> history = conv.get_history()
            >>> for msg in history:
            ...     print(f"{msg['role']}: {msg['content']}")
        """
        return self.history.copy()

    def get_formatted_history(self) -> str:
        """
        Get conversation history as formatted string.

        Returns:
            Human-readable conversation history

        Example:
            >>> print(conv.get_formatted_history())
            user: What is HNSW?
            assistant: HNSW is a graph-based algorithm...
            user: How does it work?
            assistant: It builds a hierarchical graph...
        """
        lines = []
        for msg in self.history:
            role = msg['role']
            content = msg['content']
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def clear(self) -> None:
        """
        Clear all conversation history.

        Example:
            >>> conv.clear()
            >>> assert len(conv.get_history()) == 0
        """
        self.history = []
        self.updated_at = datetime.now()
        logger.info(f"Cleared conversation history for session {self.session_id}")

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text (rough approximation).

        Uses simple heuristic: ~4 characters per token on average.
        This is approximate but works well enough for context management.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        # Rough approximation: 4 chars per token
        return len(text) // 4

    def _get_total_tokens(self) -> int:
        """
        Calculate total tokens in conversation history.

        Returns:
            Estimated total token count
        """
        total = 0
        for msg in self.history:
            total += self._estimate_tokens(msg['content'])
        return total

    def _prune_if_needed(self) -> None:
        """
        Prune old messages if exceeding limits.

        Removes oldest messages first (FIFO) until within limits.
        Always keeps pairs of messages (user + assistant) together.
        """
        # Check turn limit
        num_turns = len(self.history) // 2  # Each turn = 2 messages
        if num_turns > self.max_turns:
            # Remove oldest turn (2 messages)
            num_to_remove = (num_turns - self.max_turns) * 2
            self.history = self.history[num_to_remove:]
            logger.debug(
                f"Pruned {num_to_remove} messages due to turn limit. "
                f"Remaining: {len(self.history)}"
            )

        # Check token limit
        while self._get_total_tokens() > self.max_tokens and len(self.history) > 2:
            # Remove oldest turn (keep at least 1 turn)
            removed = self.history[:2]
            self.history = self.history[2:]
            logger.debug(
                f"Pruned 2 messages due to token limit. "
                f"Tokens: {self._get_total_tokens()}/{self.max_tokens}"
            )

    def save(self, save_dir: Path) -> Path:
        """
        Save conversation to JSON file.

        Args:
            save_dir: Directory to save session file

        Returns:
            Path to saved file

        Example:
            >>> save_path = conv.save(Path("data/sessions"))
            >>> print(f"Saved to {save_path}")
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / f"session_{self.session_id}.json"

        data = {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "max_tokens": self.max_tokens,
            "max_turns": self.max_turns,
            "history": self.history,
        }

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved conversation to {save_path}")
        return save_path

    @classmethod
    def load(cls, session_id: str, save_dir: Path) -> "ConversationHistory":
        """
        Load conversation from JSON file.

        Args:
            session_id: Session ID to load
            save_dir: Directory containing session files

        Returns:
            ConversationHistory instance

        Raises:
            FileNotFoundError: If session file doesn't exist

        Example:
            >>> conv = ConversationHistory.load("20231215_143022", Path("data/sessions"))
            >>> print(f"Loaded {len(conv.get_history())} messages")
        """
        save_dir = Path(save_dir)
        save_path = save_dir / f"session_{session_id}.json"

        if not save_path.exists():
            raise FileNotFoundError(f"Session file not found: {save_path}")

        with open(save_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        conv = cls(
            max_tokens=data['max_tokens'],
            max_turns=data['max_turns'],
            session_id=data['session_id'],
        )

        conv.history = data['history']
        conv.created_at = datetime.fromisoformat(data['created_at'])
        conv.updated_at = datetime.fromisoformat(data['updated_at'])

        logger.info(
            f"Loaded conversation {session_id} with {len(conv.history)} messages"
        )
        return conv

    def get_stats(self) -> Dict[str, Any]:
        """
        Get conversation statistics.

        Returns:
            Dictionary with conversation stats

        Example:
            >>> stats = conv.get_stats()
            >>> print(f"Turns: {stats['num_turns']}, Tokens: {stats['total_tokens']}")
        """
        return {
            "session_id": self.session_id,
            "num_messages": len(self.history),
            "num_turns": len(self.history) // 2,
            "total_tokens": self._get_total_tokens(),
            "max_tokens": self.max_tokens,
            "max_turns": self.max_turns,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"ConversationHistory(session_id='{stats['session_id']}', "
            f"turns={stats['num_turns']}, tokens={stats['total_tokens']})"
        )
