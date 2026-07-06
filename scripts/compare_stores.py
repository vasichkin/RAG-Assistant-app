"""
CLI script: run 10 queries against both FAISS and Qdrant, measure latency and
chunk overlap, then save results to evaluation/results/faiss_vs_qdrant.csv.

Usage:
    python -m scripts.compare_stores
"""

import time
import pathlib
from typing import List

import pandas as pd

from embeddings.gemini_embeddings import GeminiEmbeddings
from vectorstore.faiss_store import load_faiss
from vectorstore.qdrant_store import load_qdrant
from config.settings import FAISS_INDEX_PATH

DATASET_PATH = "resources/datasets/rag_evaluation_dataset.csv"
OUTPUT_PATH = "evaluation/results/faiss_vs_qdrant.csv"
TOP_K = 5


def load_queries(n: int = 10) -> List[str]:
    """Load the first n questions from the RAGAS evaluation CSV."""
    df = pd.read_csv(DATASET_PATH)
    df.columns = df.columns.str.lower()
    return df["question"].head(n).tolist()


def retrieve_with_timing(retriever, query: str):
    """Run a retrieval query, returning (docs, latency_ms)."""
    start = time.perf_counter()
    docs = retriever.invoke(query)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return docs, elapsed_ms


def compute_overlap(faiss_docs, qdrant_docs) -> int:
    """Count chunks that appear in both result sets (by page_content)."""
    faiss_contents = {d.page_content for d in faiss_docs}
    qdrant_contents = {d.page_content for d in qdrant_docs}
    return len(faiss_contents & qdrant_contents)


def main():
    print("Loading embeddings and vector stores...")
    embeddings = GeminiEmbeddings()

    faiss_store = load_faiss(FAISS_INDEX_PATH, embeddings)
    qdrant_store = load_qdrant(embeddings)

    faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": TOP_K})
    qdrant_retriever = qdrant_store.as_retriever(search_kwargs={"k": TOP_K})

    queries = load_queries(10)
    print(f"Loaded {len(queries)} queries from {DATASET_PATH}\n")

    rows = []
    for i, query in enumerate(queries, start=1):
        print(f"Query {i}/{len(queries)}: {query[:80]}")

        faiss_docs, faiss_latency = retrieve_with_timing(faiss_retriever, query)
        qdrant_docs, qdrant_latency = retrieve_with_timing(qdrant_retriever, query)
        overlap = compute_overlap(faiss_docs, qdrant_docs)

        print(
            f"  FAISS: {faiss_latency:.1f} ms, {len(faiss_docs)} chunks | "
            f"Qdrant: {qdrant_latency:.1f} ms, {len(qdrant_docs)} chunks | "
            f"Overlap: {overlap}"
        )

        rows.append(
            {
                "query": query,
                "faiss_latency_ms": round(faiss_latency, 2),
                "qdrant_latency_ms": round(qdrant_latency, 2),
                "faiss_chunks": len(faiss_docs),
                "qdrant_chunks": len(qdrant_docs),
                "overlap_count": overlap,
            }
        )

    df = pd.DataFrame(rows)

    output_path = pathlib.Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {OUTPUT_PATH}")

    # Summary
    print("\n--- Summary ---")
    print(f"Avg FAISS latency  : {df['faiss_latency_ms'].mean():.1f} ms")
    print(f"Avg Qdrant latency : {df['qdrant_latency_ms'].mean():.1f} ms")
    print(f"Avg overlap count  : {df['overlap_count'].mean():.1f} / {TOP_K}")
    print(f"Total queries      : {len(df)}")


if __name__ == "__main__":
    main()
