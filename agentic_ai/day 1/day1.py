"""
Day 1 — Python for AI Engineers
Topics: dataclasses, Pydantic, type hints, comprehensions, *args/**kwargs
"""

# ============================================================
# SECTION 1: dataclasses
# ============================================================
# A dataclass auto-generates __init__, __repr__, __eq__ for you.
# Think of it like a plain JS object with a fixed shape.

from dataclasses import dataclass, field


@dataclass
class Message:
    role: str
    content: str
    tokens: int = 0  # default value


@dataclass
class Conversation:
    messages: list[Message] = field(default_factory=list)  # mutable default needs field()

    def add(self, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content))


# Try it:
convo = Conversation()
convo.add("user", "Hello Claude")
convo.add("assistant", "Hello! How can I help?")
print(convo)


# ============================================================
# SECTION 2: Pydantic models
# ============================================================
# Pydantic = dataclass + runtime validation + JSON parsing.
# This is what you use when data comes from outside (API, user input).
# Think: Zod in TypeScript, but for Python.

from pydantic import BaseModel, Field
from typing import Optional


class LLMResponse(BaseModel):
    id: str
    model: str
    content: str
    input_tokens: int
    output_tokens: int
    stop_reason: Optional[str] = None  # optional field, defaults to None


# Pydantic validates on construction — this works:
response = LLMResponse(
    id="msg_01abc",
    model="claude-sonnet-4-6",
    content="Paris is the capital of France.",
    input_tokens=12,
    output_tokens=8,
)
print(response)
print(response.model_dump())  # → plain dict, like JSON.stringify in JS

# Pydantic rejects bad data — uncomment to see the error:
# bad = LLMResponse(id=123, model="claude", content="hi", input_tokens="oops", output_tokens=8)


# ============================================================
# SECTION 3: Type hints
# ============================================================
# Type hints don't enforce at runtime (unlike Pydantic), but they:
# - document intent
# - enable autocomplete in VS Code
# - catch bugs with mypy / pyright

from typing import Union


def chunk_text(text: str, size: int = 512, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def count_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def get_model(complexity: str) -> str:
    """Route to appropriate model based on complexity."""
    models: dict[str, str] = {
        "simple": "claude-haiku-4-5-20251001",
        "medium": "claude-sonnet-4-6",
        "complex": "claude-opus-4-7",
    }
    return models.get(complexity, "claude-sonnet-4-6")


# Union type — when a value can be one of several types
def parse_token_count(value: Union[int, str]) -> int:
    return int(value)


# ============================================================
# SECTION 4: List and dict comprehensions
# ============================================================

texts = [
    "What is RAG?",
    "Explain embeddings.",
    "How does attention work?",
    "What is a transformer?",
    "",
    "   ",
]

# List comprehension: [expression for item in iterable if condition]
# JS equivalent: texts.filter(t => t.trim()).map(t => t.strip())
clean_texts = [t.strip() for t in texts if t.strip()]
print(clean_texts)

# Dict comprehension: {key: value for item in iterable}
token_counts = {text: count_tokens(text) for text in clean_texts}
print(token_counts)

# Nested comprehension: flatten a list of lists
batches = [["doc1", "doc2"], ["doc3"], ["doc4", "doc5", "doc6"]]
all_docs = [doc for batch in batches for doc in batch]
print(all_docs)


# ============================================================
# SECTION 5: *args and **kwargs
# ============================================================
# *args  → variable positional arguments → tuple inside function
# **kwargs → variable keyword arguments → dict inside function
# Same concept as (...args) in JS spread syntax

def log_api_call(*args: str, **kwargs: int) -> None:
    """Log model and metadata for an API call."""
    print("Models called:", args)      # tuple: ("claude-sonnet-4-6", "claude-haiku-4-5-20251001")
    print("Metadata:", kwargs)         # dict: {"tokens": 100, "latency_ms": 240}


log_api_call("claude-sonnet-4-6", "claude-haiku-4-5-20251001", tokens=100, latency_ms=240)


# Real use case: pass arbitrary config to an API call
def build_request(prompt: str, **options) -> dict:
    return {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": options.get("max_tokens", 1024),
        "temperature": options.get("temperature", 1.0),
        "model": options.get("model", "claude-sonnet-4-6"),
    }


req = build_request("Explain RAG", max_tokens=512, temperature=0.7)
print(req)