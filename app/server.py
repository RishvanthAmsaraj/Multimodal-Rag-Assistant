"""
Multi-Modal RAG Assistant — Flask Backend Server

Provides ingest, query, stats, and clear API endpoints and serves the frontend.
"""

import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

_APP_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _APP_DIR.parent

# Ensure project root on sys.path
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from src.config import load_config
from src.pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag-server")

# -------------------------------------------------------------------
# Load config & create pipeline
# -------------------------------------------------------------------
cfg = load_config(str(_PROJECT_DIR / "configs" / "config.yaml"))

# Override paths to be relative to project root
cfg.paths.data_dir = str(_PROJECT_DIR / "data")
cfg.paths.vector_store_dir = str(_PROJECT_DIR / "data" / "chroma")
cfg.paths.uploads_dir = str(_PROJECT_DIR / "data" / "uploads")
cfg.paths.cache_dir = str(_PROJECT_DIR / "data" / "cache")

os.makedirs(cfg.paths.uploads_dir, exist_ok=True)
os.makedirs(cfg.paths.vector_store_dir, exist_ok=True)
os.makedirs(cfg.paths.cache_dir, exist_ok=True)

pipeline = RAGPipeline(cfg)

# -------------------------------------------------------------------
# Flask app
# -------------------------------------------------------------------
app = Flask(__name__, static_folder="static", static_url_path="/static")


# -------------------------------------------------------------------
# Serve the main page
# -------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(str(_APP_DIR), "index.html")


# -------------------------------------------------------------------
# Stats
# -------------------------------------------------------------------
@app.route("/api/stats", methods=["GET"])
def get_stats():
    try:
        st = pipeline.stats()
        return jsonify({
            "success": True,
            "num_chunks": st.get("num_chunks", 0),
            "embedder_model": st.get("embedder_model", "N/A"),
            "embedding_dim": st.get("embedding_dim", 0),
            "provider": st.get("llm_provider", cfg.generation.llm.provider),
        })
    except Exception as exc:
        logger.exception("Stats error")
        return jsonify({"success": False, "error": str(exc)}), 500


# -------------------------------------------------------------------
# Ingest a file
# -------------------------------------------------------------------
@app.route("/api/ingest", methods=["POST"])
def ingest_file():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400

    try:
        suffix = Path(file.filename).suffix.lower()
        allowed = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp", ".txt", ".md"}
        if suffix not in allowed:
            return jsonify({
                "success": False,
                "error": f"Unsupported file type: {suffix}. Allowed: {', '.join(allowed)}",
            }), 400

        # Save to temp path
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp)
            tmp_path = tmp.name

        # Ingest
        chunks = pipeline.ingest_file(tmp_path)
        os.unlink(tmp_path)

        # Also copy to uploads dir
        upload_dst = Path(cfg.paths.uploads_dir) / file.filename
        file.seek(0)
        file.save(str(upload_dst))

        return jsonify({
            "success": True,
            "filename": file.filename,
            "chunks_ingested": len(chunks),
        })

    except Exception as exc:
        logger.exception("Ingest error")
        return jsonify({"success": False, "error": str(exc)}), 500


# -------------------------------------------------------------------
# List known sources
# -------------------------------------------------------------------
@app.route("/api/sources", methods=["GET"])
def list_sources():
    try:
        uploads_dir = Path(cfg.paths.uploads_dir)
        if not uploads_dir.exists():
            return jsonify({"success": True, "sources": []})
        files = []
        for f in sorted(uploads_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if f.is_file():
                size_kb = round(f.stat().st_size / 1024, 1)
                files.append({
                    "name": f.name,
                    "size_kb": size_kb,
                    "type": f.suffix.lower(),
                })
        return jsonify({"success": True, "sources": files})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


# -------------------------------------------------------------------
# Query
# -------------------------------------------------------------------
@app.route("/api/query", methods=["POST"])
def query():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"success": False, "error": "Question is required"}), 400

    top_k = int(data.get("top_k", cfg.retrieval.retriever.top_k))

    try:
        result = pipeline.query(question, top_k=top_k)
        return jsonify({
            "success": True,
            "question": result.get("question", question),
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "num_sources": len(result.get("sources", [])),
        })
    except Exception as exc:
        logger.exception("Query error")
        return jsonify({"success": False, "error": str(exc)}), 500


# -------------------------------------------------------------------
# Clear vector store and uploads
# -------------------------------------------------------------------
@app.route("/api/clear", methods=["POST"])
def clear():
    try:
        vs_dir = Path(cfg.paths.vector_store_dir)
        if vs_dir.exists():
            shutil.rmtree(str(vs_dir))
        os.makedirs(str(vs_dir), exist_ok=True)

        uploads_dir = Path(cfg.paths.uploads_dir)
        if uploads_dir.exists():
            shutil.rmtree(str(uploads_dir))
        os.makedirs(str(uploads_dir), exist_ok=True)

        # Recreate pipeline (clear resets state)
        global pipeline
        pipeline = RAGPipeline(cfg)
        return jsonify({"success": True})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


# -------------------------------------------------------------------
# Download source file
# -------------------------------------------------------------------
@app.route("/api/source/<filename>")
def download_source(filename):
    uploads_dir = Path(cfg.paths.uploads_dir)
    return send_from_directory(str(uploads_dir), filename)


# -------------------------------------------------------------------
# Config info
# -------------------------------------------------------------------
@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify({
        "chunk_strategy": cfg.processing.chunker.strategy,
        "chunk_size": cfg.processing.chunker.chunk_size,
        "chunk_overlap": cfg.processing.chunker.chunk_overlap,
        "top_k": cfg.retrieval.retriever.top_k,
        "embedder": cfg.processing.embedder.model_name,
        "llm_provider": cfg.generation.llm.provider,
        "llm_model": cfg.generation.llm.model_name,
    })


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8502))
    debug = os.environ.get("DEBUG", "1").lower() in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=port, debug=debug)
