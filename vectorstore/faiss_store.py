"""
FAISS vector store helpers: build, save, and load.
"""

from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS


def build_faiss(documents: List[Document], embeddings: Embeddings) -> FAISS:
    """Build a FAISS index from a list of documents."""
    return FAISS.from_documents(documents, embeddings)


def save_faiss(store: FAISS, path: str) -> None:
    """Persist a FAISS index to disk."""
    store.save_local(path)


def load_faiss(path: str, embeddings: Embeddings) -> FAISS:
    """Load a FAISS index from disk."""
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
