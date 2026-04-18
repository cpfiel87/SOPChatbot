# project-1-sop-chatbot

## What this project is
A RAG (Retrieval-Augmented Generation) pipeline with a Streamlit chat UI, built as a GxP compliance Q&A assistant. Users upload a SOP PDF, the app indexes it locally, and they can ask questions answered by Claude using only the document's content.

## Current status — FULLY WORKING (tested 2026-04-18)
All stages implemented, integrated, and verified end-to-end including the Streamlit UI.

| Stage | File | Status |
|---|---|---|
| PDF loading + chunking | `src/loader.py` | Done |
| Local embeddings | `src/embedder.py` | Done |
| FAISS index + retrieval | `src/index.py` | Done |
| Claude API answering | `src/answerer.py` | Done |
| Streamlit UI | `app.py` | Done — tested and working |
| CLI test runner | `src/pipeline.py` | Done |

## How to run

### Streamlit app (main entry point)
```bash
cd project-1-sop-chatbot
streamlit run app.py
```
Opens at `http://localhost:8501`.

### CLI pipeline (retrieval test only, no UI)
```bash
python src/pipeline.py
```

## Setup requirements

### API key
Create `project-1-sop-chatbot/.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
```
**Important:** Credits must be on [console.anthropic.com](https://console.anthropic.com) — not claude.ai. These are separate billing systems.

### Install dependencies (once)
```bash
pip install -r requirements.txt
```

## File map
| File | Role |
|---|---|
| `app.py` | Streamlit UI — PDF upload, index, chat |
| `src/loader.py` | PDF reading + chunking (chunk_size=2000, overlap=200 chars) |
| `src/embedder.py` | Local embeddings via `all-MiniLM-L6-v2` (384-dim, ~90MB, cached) |
| `src/index.py` | FAISS `IndexFlatL2` build + top-k retrieval |
| `src/answerer.py` | Claude API call + conversation history |
| `src/pipeline.py` | CLI end-to-end test (set `PDF_PATH` and `TEST_QUERY` inside) |
| `data/` | PDFs: `sample_sop.pdf`, `OJ_L_202401689_EN_TXT.pdf` |
| `.env` | API key (not committed) |
| `requirements.txt` | All dependencies |

## Key implementation details

### answerer.py
- Model: `claude-sonnet-4-20250514`
- System role: GxP compliance assistant, answers from SOP sections only
- Returns `{ "answer": str, "sources": list[str] }`
- Empty chunks → immediate return: `"This topic is not covered in the loaded documents."`
- Conversation history via module-level `_history` list (persists across turns)
- `reset_history()` clears history — called on new PDF upload and Clear Chat

### app.py (Streamlit UI)
- Dark theme via `st.set_page_config` + custom CSS
- Sidebar: PDF uploader → Build Index (with spinner) → chunk count + top-k metrics
- Chat bubbles: user=right/blue, assistant=left/dark
- Each assistant answer has a "View source sections" expander
- Disclaimer banner at the top
- Session state: messages, FAISS index, chunks, embedding model
- Clear Chat button resets display and answerer history
- New PDF upload clears previous chat automatically

## Dependencies
`anthropic`, `langchain`, `langchain-community`, `langchain-text-splitters`, `pypdf`, `faiss-cpu`, `sentence-transformers`, `streamlit`, `python-dotenv`
