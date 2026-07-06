"""
Home / About page.
"""

import streamlit as st

st.title("IFC Annual Report 2024 — RAG Assistant")
st.markdown(
    """
    Welcome to the IFC 2024 Annual Report RAG Assistant.

    **How to use:**
    - Navigate to **Upload Document** to index a new PDF.
    - Navigate to **Ask a Question** to query the report using FAISS or Qdrant retrieval.
    - Navigate to **Evaluation** to review RAGAS metric results.
    - Navigate to **Comparison** to see a side-by-side FAISS vs Qdrant latency analysis.

    **Before using:** make sure the vector indexes have been built by running:
    ```bash
    python -m scripts.build_index
    ```
    """
)

if "faiss_store" in st.session_state:
    active_doc = st.session_state.get("active_document", "IFC Annual Report 2024")
    st.success(f"Vector indexes loaded — active document: **{active_doc}**")
else:
    err = st.session_state.get("index_load_error")
    if err:
        st.error(f"Could not load vector indexes: {err}")
    else:
        st.warning(
            "Vector indexes are not loaded. "
            "Run `python -m scripts.build_index` to build the indexes first, "
            "or upload a PDF on the **Upload Document** page."
        )
