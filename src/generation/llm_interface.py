"""LLM interface with provider abstraction.

Supports three providers:

* ``local``     - HuggingFace transformers pipeline (default).
* ``openai``    - OpenAI Chat Completions API.
* ``anthropic`` - Anthropic Messages API.

The local backend is deliberately dependency-light so the project
runs out-of-the-box without API keys. It uses a small seq2seq
model (``google/flan-t5-base``) which fits comfortably on CPU.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_TEMPLATE = (
    "You are a helpful document assistant. Use ONLY the context below "
    "to answer the question. If the answer is not in the context, say "
    "\"I don't have enough information in the provided documents.\"\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


class LLMInterface:
    """Provider-agnostic LLM wrapper."""

    def __init__(
        self,
        provider: str = "local",
        model_name: str = "google/flan-t5-base",
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        device: str = "cpu",
        prompt_template: Optional[str] = None,
    ):
        self.provider = provider.lower()
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE

        if self.provider == "local":
            self._init_local()
        elif self.provider == "openai":
            self._init_openai()
        elif self.provider == "anthropic":
            self._init_anthropic()
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self, question: str, context_chunks: List[str]) -> str:
        """Generate an answer given a question + retrieved context chunks."""
        context = self._format_context(context_chunks)
        prompt = self.prompt_template.format(context=context, question=question)

        if self.provider == "local":
            return self._generate_local(prompt)
        if self.provider == "openai":
            return self._generate_openai(prompt)
        if self.provider == "anthropic":
            return self._generate_anthropic(prompt)
        return ""

    def generate_with_citations(
        self,
        question: str,
        context_chunks: List[Dict],
    ) -> Dict:
        """Generate an answer and return per-chunk citation info."""
        texts = [c["text"] if isinstance(c, dict) else str(c) for c in context_chunks]
        answer = self.generate(question, texts)
        return {
            "answer": answer,
            "citations": [
                {
                    "chunk_id": c.get("chunk_id") if isinstance(c, dict) else None,
                    "source": (c.get("metadata") or {}).get("source")
                    if isinstance(c, dict)
                    else None,
                    "score": c.get("score") if isinstance(c, dict) else None,
                }
                for c in context_chunks
            ],
        }

    # ------------------------------------------------------------------
    # Context formatting
    # ------------------------------------------------------------------
    @staticmethod
    def _format_context(chunks: List[str], max_chars: int = 3000) -> str:
        """Join chunks into a single context string, capped to avoid OOM."""
        buf: List[str] = []
        total = 0
        for i, c in enumerate(chunks):
            snippet = c.strip()
            block = f"[{i + 1}] {snippet}"
            if total + len(block) > max_chars:
                break
            buf.append(block)
            total += len(block)
        return "\n\n".join(buf)

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------
    def _init_local(self) -> None:
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "transformers not installed. Run `pip install transformers`."
            ) from exc

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

        logger.info("Loading local LLM %s on %s ...", self.model_name, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        self._pipeline = pipeline(
            "text2text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            device=0 if self.device in ("cuda", "mps") and self._model.device.type != "cpu" else -1,
            max_new_tokens=self.max_new_tokens,
        )
        logger.info("Local LLM ready")

    def _generate_local(self, prompt: str) -> str:
        out = self._pipeline(
            prompt,
            do_sample=self.temperature > 0,
            temperature=max(self.temperature, 1e-5),
            num_return_sequences=1,
        )
        return out[0]["generated_text"].strip()

    def _init_openai(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set")
        try:
            from openai import OpenAI  # noqa: F401
        except ImportError as exc:
            raise ImportError("openai not installed. Run `pip install openai`.") from exc
        from openai import OpenAI

        self._openai_client = OpenAI(api_key=api_key)
        # If model_name wasn't provided as an OpenAI one, default to gpt-3.5-turbo
        if not self.model_name.startswith(("gpt-", "o")):
            self.model_name = "gpt-3.5-turbo"

    def _generate_openai(self, prompt: str) -> str:
        resp = self._openai_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return resp.choices[0].message.content.strip()

    def _init_anthropic(self) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set")
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:
            raise ImportError("anthropic not installed. Run `pip install anthropic`.") from exc
        import anthropic

        self._anthropic_client = anthropic.Anthropic(api_key=api_key)

    def _generate_anthropic(self, prompt: str) -> str:
        resp = self._anthropic_client.messages.create(
            model=self.model_name or "claude-3-haiku-20240307",
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        # Anthropic returns a list of content blocks
        return "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        ).strip()