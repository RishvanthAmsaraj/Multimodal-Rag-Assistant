# 🚀 Push to GitHub

The local git repo is ready to push — a single commit is staged on
the `main` branch. To complete the push, run **any one** of the
options below.

## Option 1 — `gh` CLI (recommended if installed)

```bash
# Authenticate if you haven't already
gh auth login

# Create the repo and push in one shot
gh repo create RishvanthAmsaraj/multimodal-rag-assistant \
    --public \
    --source=. \
    --remote=origin \
    --description "Production-ready Multi-Modal RAG system for document intelligence with PDF/image/text ingestion, OCR, ChromaDB, sentence-transformers, Streamlit UI, and evaluation metrics." \
    --push
```

## Option 2 — Manual with a Personal Access Token

```bash
# 1. Create the repo on github.com (public, no README/.gitignore)
#    https://github.com/new

# 2. Add the remote using a PAT
git remote add origin https://<YOUR_GITHUB_PAT>@github.com/RishvanthAmsaraj/multimodal-rag-assistant.git

# 3. Push
git push -u origin main
```

> Generate a PAT at: <https://github.com/settings/tokens/new>
> Required scope: `repo` (Full control of private repositories).

## Option 3 — SSH

```bash
# 1. Create the repo on github.com

# 2. Add the SSH remote
git remote add origin git@github.com:RishvanthAmsaraj/multimodal-rag-assistant.git

# 3. Push
git push -u origin main
```

## Verification

After pushing, the repo should show:
- ✅ 33 source files
- ✅ 1 commit on `main`
- ✅ Branch: `main`
- ✅ No `venv/`, `data/`, or `.env` files (gitignored)

To clone elsewhere:
```bash
git clone https://github.com/RishvanthAmsaraj/multimodal-rag-assistant.git
cd multimodal-rag-assistant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v
streamlit run src/app.py
```