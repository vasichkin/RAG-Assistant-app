"""
LCEL RAG chain builder.

Usage:
    from generation.rag_chain import build_rag_chain
    chain = build_rag_chain(retriever)
    answer = chain.invoke(query, config={"callbacks": [langfuse_handler]})
    langfuse_handler.flush()
"""

from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from generation.gemini_llm import GeminiChatModel
from generation.prompts import RAG_PROMPT


def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents into a context string."""
    parts = []
    for doc in docs:
        ct = doc.metadata.get("content_type", "text")
        if ct == "table":
            parts.append(
                f"[TABLE from page {doc.metadata.get('page_number', '?')}]\n{doc.page_content}"
            )
        elif ct == "image":
            parts.append(
                f"[IMAGE from page {doc.metadata.get('page_number', '?')}]\n{doc.page_content}"
            )
        else:
            parts.append(
                f"[Page {doc.metadata.get('page_number', '?')}]\n{doc.page_content}"
            )
    return "\n\n---\n\n".join(parts)


def build_rag_chain(
    retriever,
    llm: Optional[GeminiChatModel] = None,
    prompt=None,
):
    """Build LCEL RAG chain: retriever | format | prompt | llm | parser.

    The returned chain is invokable with a plain query string:
        answer = chain.invoke(query, config={"callbacks": [langfuse_handler]})

    Callers are responsible for calling langfuse_handler.flush() afterwards.
    """
    if llm is None:
        llm = GeminiChatModel()
    if prompt is None:
        prompt = RAG_PROMPT

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
