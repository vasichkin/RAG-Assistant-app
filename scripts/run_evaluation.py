"""
CLI: Run RAGAS evaluation against the live RAG pipeline.

Usage:
    python -m scripts.run_evaluation
    python -m scripts.run_evaluation --csv resources/datasets/rag_evaluation_dataset.csv \\
        --output evaluation/results/phase1_ragas.json --store faiss
"""

import argparse

from evaluation.ragas_eval import run_ragas_evaluation
from vectorstore.faiss_store import load_faiss
from embeddings.gemini_embeddings import GeminiEmbeddings
from generation.gemini_llm import GeminiChatModel
from generation.rag_chain import build_rag_chain
from observability.langfuse_handler import get_handler
from config.settings import FAISS_INDEX_PATH


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on the RAG pipeline.")
    parser.add_argument(
        "--csv",
        default="resources/datasets/rag_evaluation_dataset.csv",
        help="Path to evaluation CSV (default: resources/datasets/rag_evaluation_dataset.csv)",
    )
    parser.add_argument(
        "--output",
        default="evaluation/results/phase1_ragas.json",
        help="Output path for JSON results (default: evaluation/results/phase1_ragas.json)",
    )
    parser.add_argument(
        "--store",
        choices=["faiss", "qdrant"],
        default="faiss",
        help="Vector store to use for retrieval (default: faiss)",
    )
    args = parser.parse_args()

    embeddings = GeminiEmbeddings()

    if args.store == "faiss":
        store = load_faiss(FAISS_INDEX_PATH, embeddings)
    else:
        from vectorstore.qdrant_store import load_qdrant
        store = load_qdrant(embeddings)

    retriever = store.as_retriever(search_kwargs={"k": 5})
    llm = GeminiChatModel()
    langfuse_handler = get_handler()
    chain = build_rag_chain(retriever, llm)

    scores = run_ragas_evaluation(args.csv, retriever, chain, args.output, langfuse_handler)
    print(f"RAGAS scores: {scores}")


if __name__ == "__main__":
    main()
