"""Groq LLM adapter (LLMClient interface) for grounded explanation generation.

Swappable behind a stable interface (Adapter + Factory per the template).
Provides both a blocking generate() and a streaming generate_stream().
"""
from __future__ import annotations

from collections.abc import Iterator

from app.config import settings


class LLMClient:
    def generate(self, system: str, user: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def generate_stream(self, system: str, user: str) -> Iterator[str]:  # pragma: no cover
        raise NotImplementedError


class GroqClient(LLMClient):
    def __init__(self) -> None:
        self._client = None

    def _lazy(self):
        if self._client is None:
            if not settings.groq_api_key:
                raise RuntimeError("GROQ_API_KEY is not set.")
            from groq import Groq

            self._client = Groq(api_key=settings.groq_api_key)
        return self._client

    def generate(self, system: str, user: str) -> str:
        client = self._lazy()
        resp = client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""

    def generate_stream(self, system: str, user: str) -> Iterator[str]:
        client = self._lazy()
        stream = client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class EchoClient(LLMClient):
    """Offline fallback when no GROQ_API_KEY is configured. Deterministic, no network."""

    def generate(self, system: str, user: str) -> str:
        return (
            "[offline explanation — set GROQ_API_KEY for LLM-generated narratives]\n\n"
            + user
        )

    def generate_stream(self, system: str, user: str) -> Iterator[str]:
        for line in self.generate(system, user).splitlines(keepends=True):
            yield line


def get_llm_client() -> LLMClient:
    return GroqClient() if settings.groq_api_key else EchoClient()
