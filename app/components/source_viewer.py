"""
Source viewer component: display retrieved chunks with their metadata.
"""

import streamlit as st
from langchain_core.documents import Document


def render_sources(docs: list[Document]) -> None:
    """Display retrieved chunks with their metadata.

    Args:
        docs: List of retrieved LangChain Documents.
    """
    if not docs:
        st.info("No sources retrieved.")
        return

    st.subheader("Retrieved Sources")
    for i, doc in enumerate(docs):
        ct = doc.metadata.get("content_type", "text")
        page = doc.metadata.get("page_number", "?")
        with st.expander(f"Source {i + 1} — {ct.upper()} (Page {page})"):
            preview = doc.page_content[:500] + ("…" if len(doc.page_content) > 500 else "")
            st.text(preview)
            st.json(doc.metadata)
