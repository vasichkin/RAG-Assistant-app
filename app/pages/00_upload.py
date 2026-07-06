"""
Document upload page — upload a PDF and rebuild the vector indexes.
"""

import os
import tempfile
import streamlit as st

from vectorstore.index_builder import build_index

st.title("Upload Document")
st.markdown(
    "Upload a PDF to replace the active document. "
    "The vector indexes will be rebuilt and immediately available on the **Ask a Question** page."
)

with st.form("upload_form"):
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Only PDF files are supported.",
    )
    submitted = st.form_submit_button("Build Index", type="primary")

if submitted:
    if uploaded_file is None:
        st.error("Please select a PDF file before submitting.")
    else:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        try:
            with st.spinner(f"Indexing **{uploaded_file.name}** — this may take a few minutes…"):
                faiss_store, qdrant_store = build_index(tmp_path)

            st.session_state["faiss_store"] = faiss_store
            st.session_state["qdrant_store"] = qdrant_store
            st.session_state["active_document"] = uploaded_file.name

            st.success(
                f"**{uploaded_file.name}** indexed successfully. "
                "Go to **Ask a Question** to query the document."
            )
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")
        finally:
            os.unlink(tmp_path)

active_doc = st.session_state.get("active_document")
if active_doc:
    st.info(f"Active document: **{active_doc}**")
