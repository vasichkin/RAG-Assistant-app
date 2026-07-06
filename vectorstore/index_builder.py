"""
Orchestrates the full ingestion → chunk → embed → store pipeline.
"""

import fitz
from langchain_qdrant import QdrantVectorStore
from langchain_community.vectorstores import FAISS

from ingestion.pdf_loader import load_pdf
from ingestion.image_extractor import extract_images
from ingestion.table_extractor import extract_tables
from ingestion.chunker import chunk_documents
from embeddings.gemini_embeddings import GeminiEmbeddings
from vectorstore.faiss_store import build_faiss, save_faiss
from vectorstore.qdrant_store import build_qdrant
from config.settings import FAISS_INDEX_PATH


def build_index(pdf_path: str) -> tuple[FAISS, QdrantVectorStore]:
    """Load PDF, ingest all content types, chunk, embed, store in FAISS + Qdrant."""
    with fitz.open(pdf_path) as pdf_doc:
        # 1. Load text pages
        documents = load_pdf(pdf_path, _doc=pdf_doc)

        # 2. Extract images (captions via Gemini)
        image_docs = extract_images(pdf_path, _doc=pdf_doc)

        # 3. Extract tables as markdown
        table_docs = extract_tables(pdf_path, _doc=pdf_doc)

    # 4. Combine all document sources
    all_docs = documents + image_docs + table_docs

    # 5. Chunk (tables/images pass through as single chunks)
    chunked = chunk_documents(all_docs)

    # 6. Create embeddings
    embeddings = GeminiEmbeddings()

    # 7. Build FAISS
    faiss_store = build_faiss(chunked, embeddings)

    # 8. Persist FAISS to disk
    save_faiss(faiss_store, FAISS_INDEX_PATH)

    # 9. Build Qdrant
    qdrant_store = build_qdrant(chunked, embeddings)

    return faiss_store, qdrant_store
