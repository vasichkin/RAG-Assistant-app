"""
Query page — chat UI for the IFC 2024 RAG Assistant.
"""

import streamlit as st

from app.components.sidebar import render_sidebar
from app.components.source_viewer import render_sources
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from generation.function_calling_chain import run_agentic_rag
from generation.gemini_llm import GeminiChatModel
from generation.prompts import MULTIMODAL_PROMPT
from generation.rag_chain import format_docs
from observability.langfuse_handler import get_handler
from retrieval.reranker import CrossEncoderReranker

st.title("Ask a Question")


@st.cache_resource
def _get_llm():
    return GeminiChatModel()


@st.cache_resource
def _get_reranker():
    return CrossEncoderReranker()


def _get_store_and_retriever(settings: dict):
    """Return (store, retriever) based on sidebar selection."""
    store_name = settings["store"]
    top_k = settings["top_k"]

    if store_name == "FAISS":
        store = st.session_state.get("faiss_store")
    else:
        store = st.session_state.get("qdrant_store")

    if store is None:
        return None, None

    retriever = store.as_retriever(search_kwargs={"k": top_k * 4})
    return store, retriever


def main():
    settings = render_sidebar()

    # Guard: indexes must be loaded from main.py first
    if "faiss_store" not in st.session_state or "qdrant_store" not in st.session_state:
        st.warning(
            "Vector indexes are not loaded. "
            "Please run `streamlit run app/main.py` from the project root, "
            "or run `python -m scripts.build_index` to build the indexes first."
        )
        return

    # Initialise chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Render chat history
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input("Ask about the IFC 2024 Annual Report...")

    if query:
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state["messages"].append({"role": "user", "content": query})

        _, retriever = _get_store_and_retriever(settings)
        if retriever is None:
            st.error("Selected vector store is not available.")
            return

        response_text = ""

        if settings.get("mode") == "Agentic":
            # --- Agentic mode: Gemini function calling + structured JSON output ---
            with st.chat_message("assistant"):
                with st.spinner("Searching report…"):
                    result = run_agentic_rag(query, retriever)
                answer_text = result.get("answer", "")
                page_refs = result.get("page_references", [])
                st.markdown(answer_text)
                if page_refs:
                    st.caption(f"Pages referenced: {', '.join(str(p) for p in page_refs)}")
            response_text = answer_text

        else:
            # --- Standard mode: retrieve → format → prompt → stream ---
            handler = get_handler()
            retrieved_docs = retriever.invoke(query)
            retrieved_docs = _get_reranker().rerank(query, retrieved_docs, top_k=settings["top_k"])
            content_type_filter = settings.get("content_type", ["text", "table", "image"])
            filtered_docs = [
                d for d in retrieved_docs
                if d.metadata.get("content_type", "text") in content_type_filter
            ] or retrieved_docs

            context_str = format_docs(filtered_docs)
            llm = _get_llm()
            chain = (
                RunnablePassthrough()
                | (lambda q: {"context": context_str, "question": q})
                | MULTIMODAL_PROMPT
                | llm
                | StrOutputParser()
            )

            with st.chat_message("assistant"):
                response_text = st.write_stream(
                    chain.stream(query, config={"callbacks": [handler]})
                )

            handler.flush()
            render_sources(filtered_docs)

        st.session_state["messages"].append({"role": "assistant", "content": response_text})


main()
