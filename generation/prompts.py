"""
Prompt templates for RAG chains.

Usage:
    from generation.prompts import RAG_PROMPT, MULTIMODAL_PROMPT
"""

from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Use the provided context to answer the question. "
        "If the context doesn't contain enough information, say so.",
    ),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

MULTIMODAL_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a helpful assistant analyzing an IFC Annual Report. "
            "The context may include text excerpts, data tables (formatted as markdown), "
            "and image descriptions. Use all available information to answer comprehensively. "
            "When referencing tables, summarize the key data. "
            "When referencing image descriptions, note what was depicted."
        ),
    ),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])
