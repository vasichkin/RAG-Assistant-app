"""
Streamlit multi-page app entry point.

Run with:
    streamlit run app/main.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from embeddings.gemini_embeddings import GeminiEmbeddings
from vectorstore.faiss_store import load_faiss
from vectorstore.qdrant_store import load_qdrant
from config.settings import FAISS_INDEX_PATH


@st.cache_resource
def load_indexes():
    """Load FAISS and Qdrant indexes once at startup (cached across reruns)."""
    embeddings = GeminiEmbeddings()
    faiss_store = load_faiss(FAISS_INDEX_PATH, embeddings)
    qdrant_store = load_qdrant(embeddings)
    return faiss_store, qdrant_store


# Load indexes into session state once; skip if already present (e.g. after upload).
if "faiss_store" not in st.session_state or "qdrant_store" not in st.session_state:
    try:
        faiss_store, qdrant_store = load_indexes()
        st.session_state["faiss_store"] = faiss_store
        st.session_state["qdrant_store"] = qdrant_store
    except Exception as e:
        st.session_state["index_load_error"] = str(e)

pg = st.navigation([
    st.Page("pages/00_upload.py", title="Upload Document"),
    st.Page("pages/01_query.py", title="Ask a Question"),
    st.Page("pages/02_evaluation.py", title="Evaluation"),
    st.Page("pages/03_comparison.py", title="Comparison"),
    st.Page("pages/04_home.py", title="Home"),
])
pg.run()
