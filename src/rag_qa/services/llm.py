"""Groq LLM client for answer generation."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when answer generation fails (config, network, or upstream API)."""


_SYSTEM_PROMPT = """You are a helpful document assistant.
Answer the user's question using ONLY the context provided below.
If the context does not contain enough information, say so clearly.
Be concise and accurate. Cite the source document names when relevant."""


def _build_prompt(question: str, context_chunks: list[dict]) -> str:
    """Format retrieved chunks into a context block for the LLM."""
    context_parts: list[str] = []
    for i, chunk in enumerate(context_chunks, start=1):
        filename = chunk.get("filename", "unknown")
        text = chunk.get("text", "")
        context_parts.append(f"[{i}] (from {filename}):\n{text}")

    context_block = "\n\n".join(context_parts)
    return f"Context:\n{context_block}\n\nQuestion: {question}"


class GroqLLM:
    """Wrapper around the Groq SDK for chat completions."""

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    @property
    def client(self):
        """Lazy-init Groq client."""
        if self._client is None:
            from groq import Groq

            self._client = Groq(api_key=self.api_key)
        return self._client

    def generate(self, question: str, context_chunks: list[dict]) -> str:
        """Generate an answer grounded in *context_chunks*.

        Raises:
            LLMError: if the API key is missing, the Groq SDK cannot be
                imported, or the upstream completion call fails.
        """
        if not self.api_key:
            raise LLMError("GROQ_API_KEY is not set. Set it in .env or environment.")

        user_message = _build_prompt(question, context_chunks)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except ImportError as exc:
            raise LLMError(
                "The 'groq' package is not installed. Run: pip install groq"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - normalize SDK/network errors
            logger.exception("Groq completion failed for model %s", self.model)
            raise LLMError(f"Answer generation failed: {exc}") from exc

        try:
            answer = response.choices[0].message.content or ""
        except (AttributeError, IndexError, KeyError) as exc:
            raise LLMError("Groq returned an unexpected response shape.") from exc

        logger.debug("LLM generated %d chars for question: %.60s...", len(answer), question)
        return answer.strip()
