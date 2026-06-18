"""Command-line interface for the Multi-Modal RAG Assistant.

Usage:
    python -m src.cli ingest path/to/file.pdf path/to/notes.txt ...
    python -m src.cli query "What is the Eiffel Tower?"
    python -m src.cli stats
    python -m src.cli reset
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Make `python -m src.cli` work whether run from project root or src/
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import load_config  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("cli")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="multimodal-rag", description="Multi-Modal RAG CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="Ingest one or more files")
    ing.add_argument("files", nargs="+", type=Path)

    q = sub.add_parser("query", help="Ask a question")
    q.add_argument("question", type=str)
    q.add_argument("--top-k", type=int, default=None)
    q.add_argument("--json", action="store_true")

    sub.add_parser("stats", help="Print pipeline stats")

    sub.add_parser("reset", help="Wipe the vector store")

    eval_p = sub.add_parser("eval", help="Run evaluation against a JSON file")
    eval_p.add_argument("dataset", type=Path, help="Path to eval_dataset.json")
    eval_p.add_argument("--k", type=int, default=5)

    return p


def cmd_ingest(pipeline: RAGPipeline, args):
    total = 0
    for f in args.files:
        try:
            chunks = pipeline.ingest_file(f)
            print(f"✅ {f} → {len(chunks)} chunks")
            total += len(chunks)
        except Exception as exc:
            print(f"❌ {f}: {exc}", file=sys.stderr)
    print(f"\nTotal: {total} chunks indexed.")


def cmd_query(pipeline: RAGPipeline, args):
    result = pipeline.query(args.question, top_k=args.top_k)
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"\n❓ {result['question']}")
        print(f"\n💡 {result['answer']}\n")
        if result.get("sources"):
            print(f"📚 Sources ({result['num_sources']}):")
            for i, s in enumerate(result["sources"], start=1):
                src_name = Path(s["metadata"].get("source", "?")).name
                print(f"  [{i}] {src_name} (score={s['score']:.3f})")
                print(f"      {s['text'][:120]}...")


def cmd_stats(pipeline: RAGPipeline, args):
    print(json.dumps(pipeline.stats(), indent=2))


def cmd_reset(pipeline: RAGPipeline, args):
    confirm = input("This will wipe the vector store. Continue? [y/N] ").strip().lower()
    if confirm == "y":
        pipeline.vector_store.reset()
        print("✅ Vector store reset.")
    else:
        print("Cancelled.")


def cmd_eval(pipeline: RAGPipeline, args):
    from src.evaluation.metrics import run_evaluation

    with args.dataset.open() as fh:
        samples = json.load(fh).get("samples", [])
    report = run_evaluation(pipeline, samples, k=args.k)
    print(report.summary())


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = load_config()
    pipeline = RAGPipeline(cfg)

    dispatch = {
        "ingest": cmd_ingest,
        "query": cmd_query,
        "stats": cmd_stats,
        "reset": cmd_reset,
        "eval": cmd_eval,
    }
    dispatch[args.cmd](pipeline, args)


if __name__ == "__main__":
    main()