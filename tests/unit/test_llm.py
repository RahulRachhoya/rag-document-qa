"""Unit tests for the Groq LLM client: error normalization and config wiring.

These tests never hit the network. The lazy ``client`` property is patched or
the underlying ``groq.Groq`` import is monkeypatched so we exercise:

- LLMError on missing api_key (config error)
- LLMError on ImportError (groq package not installed)
- LLMError on a generic SDK/network exception
- LLMError on an unexpected response shape
- temperature / max_tokens are forwarded to the completion call
- happy-path returns a stripped answer string
"""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from rag_qa.config import Settings
from rag_qa.services.llm import GroqLLM, LLMError, _build_prompt

CONTEXT = [
    {"filename": "doc_a.txt", "text": "RAG fuses dense and sparse retrieval."},
    {"filename": "doc_b.txt", "text": "RRF combines ranked lists by reciprocal rank."},
]


def _completion_response(content: str | None) -> MagicMock:
    """Build a MagicMock shaped like a Groq chat-completion response."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


class TestPromptBuilding:
    def test_build_prompt_numbers_and_attributes_chunks(self):
        prompt = _build_prompt("What is RRF?", CONTEXT)
        assert "Question: What is RRF?" in prompt
        assert "[1] (from doc_a.txt):" in prompt
        assert "[2] (from doc_b.txt):" in prompt
        assert "RRF combines ranked lists" in prompt

    def test_build_prompt_tolerates_missing_keys(self):
        prompt = _build_prompt("Q", [{}])
        assert "[1] (from unknown):" in prompt


class TestConfigError:
    def test_missing_api_key_raises_llm_error(self):
        llm = GroqLLM(api_key="")
        with pytest.raises(LLMError, match="GROQ_API_KEY is not set"):
            llm.generate("Q", CONTEXT)

    def test_missing_api_key_does_not_touch_client(self):
        """The api_key guard must short-circuit before the lazy client is built."""
        llm = GroqLLM(api_key="")
        with patch.object(GroqLLM, "client", new_callable=PropertyMock) as mock_client:
            with pytest.raises(LLMError):
                llm.generate("Q", CONTEXT)
            mock_client.assert_not_called()


class TestImportError:
    def test_groq_not_installed_raises_llm_error(self):
        llm = GroqLLM(api_key="real-key")
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = ImportError("No module named 'groq'")
        with patch.object(GroqLLM, "client", new_callable=PropertyMock, return_value=mock_client):
            with pytest.raises(LLMError, match="groq.*not installed"):
                llm.generate("Q", CONTEXT)


class TestUpstreamFailure:
    def test_generic_sdk_exception_is_normalized(self):
        llm = GroqLLM(api_key="real-key")
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("503 upstream timeout")
        with patch.object(GroqLLM, "client", new_callable=PropertyMock, return_value=mock_client):
            with pytest.raises(LLMError, match="Answer generation failed"):
                llm.generate("Q", CONTEXT)

    def test_original_exception_is_chained(self):
        llm = GroqLLM(api_key="real-key")
        original = ValueError("rate limited")
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = original
        with patch.object(GroqLLM, "client", new_callable=PropertyMock, return_value=mock_client):
            with pytest.raises(LLMError) as excinfo:
                llm.generate("Q", CONTEXT)
            assert excinfo.value.__cause__ is original


class TestResponseShape:
    def test_unexpected_response_shape_raises_llm_error(self):
        llm = GroqLLM(api_key="real-key")
        mock_client = MagicMock()
        broken = MagicMock()
        broken.choices = []  # IndexError on choices[0]
        mock_client.chat.completions.create.return_value = broken
        with patch.object(GroqLLM, "client", new_callable=PropertyMock, return_value=mock_client):
            with pytest.raises(LLMError, match="unexpected response shape"):
                llm.generate("Q", CONTEXT)

    def test_none_content_returns_empty_string(self):
        """A null completion content is valid (not a shape error) -> empty answer."""
        llm = GroqLLM(api_key="real-key")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _completion_response(None)
        with patch.object(GroqLLM, "client", new_callable=PropertyMock, return_value=mock_client):
            assert llm.generate("Q", CONTEXT) == ""


class TestHappyPath:
    def test_returns_stripped_answer(self):
        llm = GroqLLM(api_key="real-key")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _completion_response(
            "  Dense and sparse are fused via RRF.  "
        )
        with patch.object(GroqLLM, "client", new_callable=PropertyMock, return_value=mock_client):
            answer = llm.generate("How does fusion work?", CONTEXT)
        assert answer == "Dense and sparse are fused via RRF."


class TestSamplingWiring:
    def test_temperature_and_max_tokens_forwarded_to_completion(self):
        llm = GroqLLM(api_key="real-key", temperature=0.7, max_tokens=512)
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _completion_response("ok")
        with patch.object(GroqLLM, "client", new_callable=PropertyMock, return_value=mock_client):
            llm.generate("Q", CONTEXT)

        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 512
        assert kwargs["model"] == "llama-3.3-70b-versatile"

    def test_defaults_match_settings_defaults(self):
        """GroqLLM defaults should mirror the Settings defaults so the wiring
        in pipeline.py (Settings -> GroqLLM) is consistent."""
        settings = Settings(groq_api_key="x", qdrant_url="")
        llm = GroqLLM(api_key="x")
        assert llm.temperature == settings.llm_temperature
        assert llm.max_tokens == settings.llm_max_tokens
