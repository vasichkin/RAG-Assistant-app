"""
Hybrid retriever: BM25 + vector ensemble.

Usage:
    from retrieval.hybrid_retriever import build_hybrid_retriever
    hybrid = build_hybrid_retriever(documents, vector_retriever, top_k=5)
"""

import importlib.util
import os

from langchain_community.retrievers import BM25Retriever

try:
    from langchain.retrievers import EnsembleRetriever
except Exception:
    # Fallback for environments where langchain.retrievers.__init__ triggers a
    # broken import chain (observed on Python 3.14): load the module directly.
    _ensemble_path = os.path.join(
        os.path.dirname(importlib.util.find_spec("langchain").origin),
        "retrievers", "ensemble.py",
    )
    _spec = importlib.util.spec_from_file_location("langchain.retrievers.ensemble", _ensemble_path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    EnsembleRetriever = _mod.EnsembleRetriever


def build_hybrid_retriever(
    documents: list,
    vector_retriever,
    top_k: int = 5,
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
):
    """Build BM25 + vector ensemble retriever."""
    bm25_retriever = BM25Retriever.from_documents(documents, k=top_k)
    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[bm25_weight, vector_weight],
    )
