# IFC Annual Report 2024 — RAG Assistant

A production-style Retrieval-Augmented Generation (RAG) system for querying the IFC 2024 Annual Report PDF. Built with LangChain, Google Gemini (Vertex AI), and Streamlit.

## Features

- **PDF upload**: Upload any PDF via the UI to rebuild indexes on the fly
- **Dual vector stores**: FAISS (local) and Qdrant (server) with side-by-side comparison
- **Hybrid retrieval**: BM25 sparse + dense vector ensemble via `EnsembleRetriever`
- **Re-ranking**: Cross-encoder reranker (`retrieval/reranker.py`)
- **Multi-modal ingestion**: Extracts text, tables, and images from the PDF (single file open per pipeline run)
- **Multi-modal embeddings**: ColPali-based page-level embeddings (`multimodal/`)
- **Generation**: Gemini 2.5 Flash via LangChain LCEL chain
- **Evaluation**: RAGAS metrics (context precision/recall, faithfulness, answer relevancy)
- **Observability**: Langfuse tracing — shared handler singleton per process
- **UI**: Streamlit multi-page app with explicit navigation order

## Project Structure

```
├── app/
│   ├── main.py               # Entry point + st.navigation() router
│   ├── pages/
│   │   ├── 00_upload.py      # PDF upload → index rebuild
│   │   ├── 01_query.py       # Chat UI (Standard + Agentic modes)
│   │   ├── 02_evaluation.py  # RAGAS results viewer
│   │   ├── 03_comparison.py  # FAISS vs Qdrant + ColPali info
│   │   └── 04_home.py        # Welcome / status page
│   └── components/           # Sidebar, source viewer
├── ingestion/                # PDF loader, chunker, table/image extractors
├── embeddings/               # Gemini text embeddings
├── multimodal/               # ColPali embeddings + MaxSim retriever
├── vectorstore/              # FAISS store, Qdrant store, index builder
├── retrieval/                # Hybrid retriever (BM25 + vector), reranker
├── generation/               # Gemini LLM wrapper, LCEL RAG chain, prompts
├── evaluation/               # RAGAS evaluation runner + results/
├── observability/            # Langfuse callback handler (singleton)
├── config/                   # Settings, Vertex AI client init
├── scripts/                  # CLI: build_index, run_evaluation, compare_stores
└── resources/                # Source PDF + evaluation dataset CSV
```

## UI Navigation

The sidebar exposes five pages in this order:

| Page | Description |
|---|---|
| **Upload Document** | Upload a PDF and rebuild FAISS + Qdrant indexes without restarting |
| **Ask a Question** | Chat interface — Standard (retriever → LLM) or Agentic (function calling) mode |
| **Evaluation** | RAGAS metric results per phase with cross-phase comparison charts |
| **Comparison** | FAISS vs Qdrant latency/overlap analysis; ColPali collection status |
| **Home** | Welcome page and index load status |

## Prerequisites

- Python 3.11+
- Google Cloud project with Vertex AI API enabled
- `gcloud auth application-default login` configured
- Qdrant running (Docker or cloud)

## Quick Start

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Configure environment**
```bash
cp .env.example .env
# Edit .env: set GOOGLE_CLOUD_PROJECT, LANGFUSE_*, QDRANT_HOST
```

**3. Build vector indexes**
```bash
python -m scripts.build_index
# or for a custom PDF:
python -m scripts.build_index --pdf path/to/report.pdf
```

Alternatively, skip this step and upload a PDF directly from the **Upload Document** page after starting the app.

**4. Launch the app**
```bash
streamlit run app/main.py
```

## Docker Compose

Starts Qdrant + Streamlit app together:
```bash
docker-compose up --build
```

App is available at `http://localhost:8501`.

The compose file handles two environment overrides automatically — no manual `.env` edits needed for Docker:

| Override | Value | Reason |
|---|---|---|
| `QDRANT_HOST` | `qdrant` | Container DNS name instead of `localhost` |
| `GOOGLE_APPLICATION_CREDENTIALS` | `/root/.config/gcloud/application_default_credentials.json` | Maps host ADC into the container |

The host's `~/.config/gcloud` directory is mounted read-only into the container so ADC credentials are available without baking them into the image.

## Evaluation

Run RAGAS evaluation against the bundled question/answer dataset:
```bash
python -m scripts.run_evaluation
```
Results are saved to `evaluation/results/` as JSON. Phase results (phase1–phase6) are already included.

Run ColPali (Phase 6) evaluation separately:
```bash
python -m scripts.run_phase6_eval
```

## Configuration

All settings live in `config/settings.py` and are overridable via environment variables:

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_CLOUD_PROJECT` | — | GCP project ID (required) |
| `GOOGLE_CLOUD_LOCATION` | `us-central1` | Vertex AI region |
| `QDRANT_HOST` | `localhost` | Qdrant host (`qdrant` inside Docker) |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `LANGFUSE_SECRET_KEY` | — | Langfuse secret key |
| `LANGFUSE_PUBLIC_KEY` | — | Langfuse public key |
| `GOOGLE_APPLICATION_CREDENTIALS` | — | Path to ADC JSON (set automatically in Docker) |

Key model/chunking constants (edit `config/settings.py`):
- `EMBEDDING_MODEL`: `text-embedding-005`
- `GENERATION_MODEL`: `gemini-2.5-flash`
- `CHUNK_SIZE`: 512 tokens, `CHUNK_OVERLAP`: 64
- `RETRIEVAL_TOP_K`: 5

**Author:** Vasyl Khomenko <vkhomenko@griddynamics.com>