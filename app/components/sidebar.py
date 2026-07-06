"""
Sidebar component: vector store selector, top-k slider, content type filter,
and chain mode toggle.
"""

import streamlit as st


def render_sidebar() -> dict:
    """Render sidebar controls.

    Returns:
        dict with keys:
            store (str): "FAISS" or "Qdrant"
            top_k (int): number of chunks to retrieve
            content_type (list[str]): selected content type filters
            mode (str): "Standard" or "Agentic"
    """
    with st.sidebar:
        st.header("Settings")
        store = st.selectbox("Vector Store", ["FAISS", "Qdrant"], key="store_select")
        top_k = st.slider("Top-K results", min_value=1, max_value=10, value=5, key="top_k")
        content_type = st.multiselect(
            "Content Type Filter",
            ["text", "table", "image"],
            default=["text", "table", "image"],
            key="content_type_filter",
        )
        mode = st.selectbox(
            "Chain Mode",
            ["Standard", "Agentic (Function Calling)"],
            key="chain_mode",
            help=(
                "Standard: retriever → prompt → LLM. "
                "Agentic: Gemini calls search_ifc_report tool, then returns structured JSON."
            ),
        )
    return {
        "store": store,
        "top_k": top_k,
        "content_type": content_type,
        "mode": "Agentic" if mode.startswith("Agentic") else "Standard",
    }
