# Multi-Modal RAG Assistant

Production-ready Retrieval-Augmented Generation system for PDFs, images,
and text -- with OCR, smart chunking, vector search, LLM synthesis, and a
web UI.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-ready-green)](https://python.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-vector%20store-orange)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-ff4b4b?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-purple)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-pytest-brightgreen)](tests/)

---

## Features

| Capability | Details |
|------------|---------|
| PDF Ingestion | Page-aware text extraction via pypdf / PyPDF2 |
| Image OCR | pytesseract (default) with auto-grayscale + contrast preprocessing; easyocr fallback |
| Text Loading | UTF-8 .txt and .md files |
| Smart Chunking | Recursive / fixed / sentence strategies with configurable overlap |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (384-dim, CPU/CUDA/MPS) |
| Vector Store | ChromaDB with cosine / L2 / inner-product distance, persistent or in-memory |
| Retrieval | Top-k search, score-threshold filtering, optional re-ranking hook |
| LLM Generation | Pluggable providers: local (HF flan-t5), OpenAI, Anthropic |
| Evaluation | Precision@k, Recall@k, MRR, answer relevance, faithfulness, context precision |
| Web UI | Flask web app with drag-and-drop upload, chat, source citations, and live stats |
| Tests | pytest suite covering ingestion, chunking, retrieval, evaluation, end-to-end |

---

## Architecture

```
                        +----------------------------------------+
                        |         STREAMLIT WEB UI               |
                        |  (upload, chat, citations, stats)      |
                        +----------------+-----------------------+
                                         |
                                         v
                    +------------------------------------------+
                    |         RAGPipeline (orchestrator)       |
                    +------+----------+--------------+---------+
                           |          |              |
           +---------------+          |              +----------------+
           |                          |                               |
           v                          v                               v
 +-------------------+     +----------------------+      +---------------------+
 |   INGESTION       |     |     PROCESSING       |      |     GENERATION      |
 |  - PDFParser      |--|  |  - TextChunker       |  |-->|  - LLMInterface     |
 |  - ImageOCR       |  |  |    (recursive/       |  |   |    (local/OpenAI/  |
 |  - TextLoader     |  |  |     fixed/sentence)  |  |   |     Anthropic)      |
 +-------------------+  |  |  - Embedder          |  |   +---------------------+
                        |  |    (MiniLM-L6-v2)    |  |
                        |  +----------+-----------+  |
                        |             |              |
                        |             v              |
                        |  +----------------------+  |
                        |  |   RETRIEVAL          |  |
                        |->|  - VectorStore       |<-|
                           |    (ChromaDB)        |
                           |  - Retriever (top-k) |
                           +----------+-----------+
                                      |
                                      v
                           +----------------------+
                           |     EVALUATION       |
                           |  - Precision@k       |
                           |  - Recall@k          |
                           |  - MRR               |
                           |  - Faithfulness      |
                           |  - Relevance         |
                           +----------------------+
```

### Data Flow

```
   file (.pdf / .png / .txt)
        |
        v
   [ Ingestion ] --------------> raw document dict
        |
        v
   [ Chunking  ] --------------> list[Chunk]   (text + metadata)
        |
        v
   [ Embedding ] --------------> np.ndarray (n, 384)
        |
        v
   [ ChromaDB  ] --------------> persisted vectors
        |
   +----+----+
   | query   |
   +----+----+
        |
        v
   [ Retriever ] --------------> top-k RetrievalResults
        |
        v
   [ LLM      ] ---------------> grounded answer + citations
```

---

## Project Structure

```
multimodal-rag-assistant/
├── configs/
│   └── config.yaml                # all tunables in one place
├── src/
│   ├── config.py                  # config loader
│   ├── pipeline.py                # end-to-end orchestrator
│   ├── cli.py                     # python -m src.cli ...
│   ├── app/                        # Flask web UI (server + frontend)
│   ├── ingestion/
│   │   ├── pdf_parser.py          # PyPDF / PyMuPDF
│   │   ├── image_ocr.py           # pytesseract / easyocr
│   │   └── text_loader.py         # .txt / .md
│   ├── processing/
│   │   ├── chunker.py             # recursive / fixed / sentence
│   │   └── embedder.py            # sentence-transformers wrapper
│   ├── retrieval/
│   │   ├── vector_store.py        # ChromaDB facade
│   │   └── retriever.py           # embed + query + rank
│   ├── generation/
│   │   └── llm_interface.py       # local / OpenAI / Anthropic
│   └── evaluation/
│       └── metrics.py             # Precision@k, Recall@k, MRR, etc.
├── tests/
│   ├── conftest.py
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   ├── test_evaluation.py
│   ├── test_ingestion.py
│   ├── test_end_to_end.py
│   └── eval_dataset.json
├── requirements.txt
├── .gitignore
├── .env.example
└── README.md
```

---

## Quickstart

### 1. Clone and Install

```bash
git clone https://github.com/RishvanthAmsaraj/multimodal-rag-assistant.git
cd multimodal-rag-assistant

python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

**OCR note:** pytesseract requires the Tesseract binary on your system.
- macOS: `brew install tesseract`
- Ubuntu: `sudo apt install tesseract-ocr`
- Windows: [installer](https://github.com/UB-Mannheim/tesseract/wiki)

### 2. Run the Web UI

```bash
venv/bin/python app/server.py
```

Then open http://localhost:8501 and:

1. Drop a PDF / image / text file into the uploader
2. Click "Ingest Selected Files"
3. Switch to the "Chat" tab and ask questions
4. Inspect retrieved source chunks under each answer

### 3. CLI Usage

```bash
# Ingest
python -m src.cli ingest paper.pdf notes.txt scan.png

# Ask a question
python -m src.cli query "What is the main contribution of the paper?"

# Pretty stats
python -m src.cli stats

# Run evaluation
python -m src.cli eval tests/eval_dataset.json

# Reset (wipes vector store)
python -m src.cli reset
```

### 4. Programmatic Use

```python
from src.config import load_config
from src.pipeline import RAGPipeline

pipeline = RAGPipeline(load_config())

# Ingest
pipeline.ingest_file("path/to/paper.pdf")

# Query
result = pipeline.query("Summarize the methodology section.")
print(result["answer"])
for src in result["sources"]:
    print(f"  [{src['score']:.2f}] {src['metadata'].get('source')}")
```

---

## Configuration

Everything is driven by `configs/config.yaml` -- edit and restart.

Key knobs:

| Key | Default | Description |
|-----|---------|-------------|
| `processing.chunker.chunk_size` | `512` | Max characters per chunk |
| `processing.chunker.chunk_overlap` | `64` | Overlap between adjacent chunks |
| `processing.chunker.strategy` | `recursive` | `recursive` / `fixed` / `sentence` |
| `processing.embedder.model_name` | `sentence-transformers/all-MiniLM-L6-v2` | Swap in any HF sentence-transformer |
| `processing.embedder.device` | `cpu` | `cpu` / `cuda` / `mps` |
| `retrieval.vector_store.distance_metric` | `cosine` | `cosine` / `l2` / `ip` |
| `retrieval.retriever.top_k` | `5` | Number of chunks retrieved per query |
| `generation.llm.provider` | `local` | `local` / `openai` / `anthropic` |
| `generation.llm.model_name` | `google/flan-t5-base` | Any compatible HF / API model |
| `generation.llm.temperature` | `0.2` | Sampling temperature |

Environment variables override config (e.g. `LLM_PROVIDER=openai`).

---

## LLM Providers

### Local (default -- no API key needed)

Uses HuggingFace transformers + a small seq2seq model
(`google/flan-t5-base`). Great for offline / privacy-sensitive
setups. Swap in any larger seq2seq (e.g. `flan-t5-large`) by
updating `generation.llm.model_name`.

### OpenAI

```bash
export OPENAI_API_KEY=sk-...
export LLM_PROVIDER=openai
# Optionally: export LLM_MODEL_NAME=gpt-4o-mini
```

### Anthropic

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export LLM_PROVIDER=anthropic
# Optionally: export LLM_MODEL_NAME=claude-3-haiku-20240307
```

---

## Evaluation

The `src.evaluation.metrics` module ships with:

| Metric | Range | Description |
|--------|-------|-------------|
| `precision@k` | 0-1 | Fraction of top-k retrieved chunks that are relevant |
| `recall@k` | 0-1 | Fraction of relevant chunks retrieved in top-k |
| `mrr` | 0-1 | Mean Reciprocal Rank -- rewards early relevant hits |
| `answer_relevance` | 0-1 | Cosine similarity between question and answer embeddings |
| `answer_faithfulness` | 0-1 | Fraction of answer sentences supported by retrieved context |
| `context_precision` | 0-1 | Precision across the entire retrieved set |

Run the bundled eval:

```bash
python -m src.cli eval tests/eval_dataset.json --k 5
```

Or programmatically:

```python
from src.evaluation.metrics import run_evaluation

samples = [
    {
        "question": "What is RAG?",
        "relevant_chunk_ids": ["chunk_0"],
        "reference_answer": "Retrieval-augmented generation.",
    },
]
report = run_evaluation(pipeline, samples, k=5)
print(report.summary())
```

---

## Tests

```bash
pytest tests/ -v
```

Coverage report:

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

The suite uses dependency-light fakes for embeddings / LLM so it
runs in seconds without GPU, internet, or Tesseract.

---

## Tech Stack

- [LangChain](https://python.langchain.com/) -- orchestration primitives
- [ChromaDB](https://www.trychroma.com/) -- embedded vector database
- [sentence-transformers](https://www.sbert.net/) -- dense embeddings
- [PyPDF / pypdf](https://pypdf.readthedocs.io/) -- PDF extraction
- [pytesseract](https://github.com/madmaze/pytesseract) -- OCR backend
- [Pillow](https://python-pillow.org/) -- image preprocessing
- [Streamlit](https://streamlit.io/) -- web UI
- [pytest](https://docs.pytest.org/) -- testing

---

## Roadmap

- [ ] Hybrid retrieval (BM25 + dense)
- [ ] Cross-encoder re-ranking
- [ ] Streaming LLM responses in the UI
- [ ] Multi-collection support (per-project namespaces)
- [ ] Docker Compose for one-command launch
- [ ] LangSmith / Phoenix tracing

---

## License

MIT (c) 2026 Rishvanth Amsaraj

---

## Acknowledgements

Inspired by the LangChain, ChromaDB, and sentence-transformers
communities. Built as a portfolio / resume-ready reference
implementation of a production-style RAG system.
