"""Streamlit web UI for the Multi-Modal RAG Assistant.

Run with:
    streamlit run src/app.py

Features:
    * Drag-and-drop upload for PDFs, images, text/markdown
    * Live document ingestion progress
    * Chat-style Q&A interface with retrieved source chips
    * Sidebar stats panel (chunks indexed, model in use, etc.)
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import List

import streamlit as st

# Ensure project root on sys.path so `src.*` imports work when run via `streamlit run src/app.py`.
import os, sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
_cfg = load_config()
st.set_page_config(
    page_title=_cfg.streamlit.page_title,
    page_icon=_cfg.streamlit.page_icon,
    layout=_cfg.streamlit.layout,
)


# ---------------------------------------------------------------------------
# Cached pipeline (heavy models load only once)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models (embeddings + LLM)…")
def get_pipeline():
    cfg = load_config()
    return RAGPipeline(cfg)


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[dict] = []
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files: List[str] = []


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def render_sidebar(pipeline: RAGPipeline):
    with st.sidebar:
        st.title("RAG Assistant")
        st.caption("Multi-Modal Document Intelligence")

        st.divider()
        st.subheader("System Stats")
        stats = pipeline.stats()
        st.metric("Indexed Chunks", stats["num_chunks"])
        st.metric("Embedding Dim", stats["embedding_dim"])
        st.metric("Cache Size", stats["embedder_cache_size"])
        st.text(f"Embedder:\n  {stats['embedder_model']}")
        st.text(f"LLM ({stats['llm_provider']}):\n  {stats['llm_model']}")

        st.divider()
        st.subheader("Retrieval Settings")
        top_k = st.slider("Top-K chunks", min_value=1, max_value=20, value=5)
        threshold = st.slider("Score threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

        st.divider()
        st.subheader("Ingested Files")
        if st.session_state.ingested_files:
            for f in st.session_state.ingested_files:
                st.text(f"- {f}")
        else:
            st.caption("No files ingested yet.")

        with st.expander("Danger Zone", expanded=False):
            if st.button("Reset Vector Store", type="secondary"):
                pipeline.vector_store.reset()
                st.session_state.ingested_files = []
                st.session_state.chat_history = []
                st.success("Vector store reset.")
                st.rerun()

    return top_k, threshold


def render_uploader(pipeline: RAGPipeline):
    st.subheader("Upload Documents")
    st.caption("Supported formats: PDF · PNG · JPG · JPEG · TXT · MD")

    uploads = st.file_uploader(
        "Drop files here",
        type=["pdf", "png", "jpg", "jpeg", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploads and st.button("Ingest Selected Files", type="primary"):
        tmp_dir = Path(tempfile.mkdtemp(prefix="rag_uploads_"))
        progress = st.progress(0.0, text="Starting…")
        try:
            for i, up in enumerate(uploads, start=1):
                dest = tmp_dir / up.name
                with open(dest, "wb") as fh:
                    fh.write(up.getbuffer())
                progress.progress(
                    (i - 1) / len(uploads),
                    text=f"Ingesting {up.name} ({i}/{len(uploads)})…",
                )
                try:
                    chunks = pipeline.ingest_file(dest)
                    st.session_state.ingested_files.append(up.name)
                    st.success(f"[OK] {up.name} → {len(chunks)} chunks")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"[ERR] Failed to ingest {up.name}: {exc}")
            progress.progress(1.0, text="Done!")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def render_chat(pipeline: RAGPipeline, top_k: int, threshold: float):
    st.subheader("Ask Your Documents")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander(f"Sources ({len(msg['sources'])})", expanded=False):
                    for i, src in enumerate(msg["sources"], start=1):
                        st.markdown(
                            f"**[{i}] {Path(src['metadata'].get('source', '?')).name}** "
                            f"— score `{src['score']:.3f}`"
                        )
                        st.caption(src["text"])

    prompt = st.chat_input("Ask a question about your documents…")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving + generating…"):
                # Apply threshold live
                original_threshold = pipeline.retriever.score_threshold
                pipeline.retriever.score_threshold = threshold
                try:
                    result = pipeline.query(prompt, top_k=top_k)
                finally:
                    pipeline.retriever.score_threshold = original_threshold

                st.markdown(result["answer"])
                if result.get("sources"):
                    with st.expander(f"Sources ({result['num_sources']})", expanded=False):
                        for i, src in enumerate(result["sources"], start=1):
                            st.markdown(
                                f"**[{i}] {Path(src['metadata'].get('source', '?')).name}** "
                                f"— score `{src['score']:.3f}`"
                            )
                            st.caption(src["text"])

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", []),
            }
        )


def main():
    st.title("Multi-Modal RAG Assistant")
    st.caption(
        "Upload PDFs, images, or text files. Ask questions in natural language. "
        "Answers are grounded in your documents with source citations."
    )

    pipeline = get_pipeline()
    top_k, threshold = render_sidebar(pipeline)

    tab_upload, tab_chat = st.tabs(["Upload", "Chat"])
    with tab_upload:
        render_uploader(pipeline)
    with tab_chat:
        render_chat(pipeline, top_k=top_k, threshold=threshold)


if __name__ == "__main__":
    main()