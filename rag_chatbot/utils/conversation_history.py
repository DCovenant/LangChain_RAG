"""Minimal conversation memory."""
from dataclasses import dataclass, field
from typing import List, Dict, Optional

MAX_MESSAGES = 10


@dataclass
class Message:
    role: str
    content: str
    sources: Optional[List[Dict]] = None


class ConversationHistory:
    def __init__(self):
        self.messages: List[Message] = []
        self.entities: List[str] = []

    def add_message(self, role: str, content: str, sources=None):
        self.messages.append(Message(role, content, sources))
        if len(self.messages) > MAX_MESSAGES:
            self.messages = self.messages[-MAX_MESSAGES:]

    def add_entities(self, entities: List[str]):
        self.entities = list(set(self.entities + entities))[-20:]

    def get_recent_context(self, n: int = 3) -> str:
        return "\n".join(
            f"{'User' if m.role == 'user' else 'Assistant'}: {m.content[:300]}"
            for m in self.messages[-n:]
        )

    def clear(self):
        self.messages = []
        self.entities = []