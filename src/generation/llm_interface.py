"""LLM interface with provider abstraction.

Supports four providers:

* ``local``     - HuggingFace transformers pipeline (default).
* ``openai``    - OpenAI Chat Completions API.
* ``anthropic`` - Anthropic Messages API.
* ``gemini``    - Google Gemini API.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Optional

import torch

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
        # Auto-detect model architecture
        self._is_causal = self._detect_causal(model_name)
        # Use lower precision for causal LMs to fit in memory
        self._torch_dtype = torch.bfloat16 if self._is_causal and device == "cpu" else torch.float32

        if self.provider == "local":
            self._init_local()
        elif self.provider == "openai":
            self._init_openai()
        elif self.provider == "anthropic":
            self._init_anthropic()
        elif self.provider == "gemini":
            self._init_gemini()
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    @staticmethod
    def _detect_causal(model_name: str) -> bool:
        """Heuristic: models with -Instruct, Qwen, Phi, Llama, Mistral are causal."""
        causal_patterns = (
            r"Instruct", r"Qwen", r"Phi[-.]", r"Llama", r"Mistral",
            r"Mixtral", r"gemma", r"GPT[12]", r"pythia", r"opt-",
        )
        return any(re.search(p, model_name, re.IGNORECASE) for p in causal_patterns)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self, question: str, context_chunks: List[str]) -> str:
        """Generate an answer given a question + retrieved context chunks."""
        context = self._format_context(context_chunks)

        # Add system identity preamble for meta-questions about this RAG system.
        system_preamble = (
            "About this RAG system: you are a Multi-Modal RAG Assistant. "
            "Your embedding model is all-MiniLM-L6-v2, your vector store is ChromaDB, "
            f"and your generation model is {self.model_name}. "
            "You answer questions based strictly on the provided document context below."
        )

        if self.provider == "local":
            return self._generate_local(question, context, system_preamble)

        context = system_preamble + "\n\n---\n\n" + context
        prompt = self.prompt_template.format(context=context, question=question)
        if self.provider == "openai":
            return self._generate_openai(prompt)
        if self.provider == "anthropic":
            return self._generate_anthropic(prompt)
        if self.provider == "gemini":
            return self._generate_gemini(prompt)
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
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
        )

        logger.info("Loading local LLM %s on %s ...", self.model_name, self.device)

        if self._is_causal:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=self._torch_dtype
            )
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)

        logger.info("Local LLM ready")

    def _generate_local(self, question: str, context: str, system_preamble: str = "") -> str:
        if self._is_causal:
            return self._generate_local_causal(question, context, system_preamble)
        return self._generate_local_seq2seq(question, context)

    def _generate_local_seq2seq(self, question: str, context: str) -> str:
        """T5 / Flan-T5 generation using seq2seq pipeline."""
        from transformers import pipeline

        t5_prompt = f"question: {question}\ncontext: {context}"
        gen = pipeline(
            "text2text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            max_new_tokens=self.max_new_tokens,
        )
        out = gen(
            t5_prompt,
            do_sample=self.temperature > 0,
            temperature=max(self.temperature, 1e-5),
            num_return_sequences=1,
        )
        return out[0]["generated_text"].strip()

    def _generate_local_causal(self, question: str, context: str, system_preamble: str = "") -> str:
        """Instruct-model generation via chat template + batch decoding."""
        sys_content = (
            "You are a helpful document assistant. Use ONLY the provided "
            "context to answer the question. If the answer is not in the context, "
            "say \"I don't have enough information in the provided documents.\"\n\n"
            f"{system_preamble}"
        )
        messages = [
            {"role": "system", "content": sys_content.strip()},
            {
                "role": "user",
                "content": f"Document context:\n{context}\n\nQuestion: {question}",
            },
        ]

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the new tokens (skip the input prompt)
        input_len = inputs["input_ids"].shape[1]
        answer = self._tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        ).strip()
        return answer

    def _init_openai(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set")
        try:
            # Ensure openai is importable
            import openai as _openai_mod  # noqa: F401
        except ImportError as exc:
            raise ImportError("openai not installed. Run `pip install openai`.") from exc
        from openai import OpenAI

        self._openai_client = OpenAI(api_key=api_key)
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
            raise ImportError(
                "anthropic not installed. Run `pip install anthropic`."
            ) from exc
        import anthropic

        self._anthropic_client = anthropic.Anthropic(api_key=api_key)

    def _generate_anthropic(self, prompt: str) -> str:
        resp = self._anthropic_client.messages.create(
            model=self.model_name or "claude-3-haiku-20240307",
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text
            for block in resp.content
            if getattr(block, "type", "") == "text"
        ).strip()

    # ------------------------------------------------------------------
    # Gemini backend
    # ------------------------------------------------------------------
    def _init_gemini(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY not set. Add it to your .env file or "
                "export it in your shell."
            )
        try:
            from google import genai as _genai_mod  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "google-genai not installed. "
                "Run `pip install google-genai`."
            ) from exc

        if not self.model_name.startswith(("gemini-",)):
            self.model_name = "gemini-flash-lite-latest"

        from google import genai

        self._gemini_client = genai.Client(api_key=api_key)
        logger.info("Gemini client ready (model=%s)", self.model_name)

    def _generate_gemini(self, prompt: str) -> str:
        from google.genai import types

        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
        generate_config = types.GenerateContentConfig(
            max_output_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )

        resp = self._gemini_client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=generate_config,
        )
        return (resp.text or "").strip()
