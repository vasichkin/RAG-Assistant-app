"""
CLI script: run 10 queries against both FAISS and Qdrant, measure latency,
chunk overlap, LLM-as-judge quality scores, and RAGAS metrics, then save
results to evaluation/results/faiss_vs_qdrant.csv and
evaluation/results/faiss_vs_qdrant_ragas.json.

Usage:
    python -m scripts.compare_stores
"""

import json
import re
import time
import pathlib
from typing import List, Tuple

import pandas as pd

from embeddings.gemini_embeddings import GeminiEmbeddings
from vectorstore.faiss_store import load_faiss
from vectorstore.qdrant_store import load_qdrant
from config.settings import FAISS_INDEX_PATH

DATASET_PATH = "resources/datasets/rag_evaluation_dataset.csv"
OUTPUT_PATH = "evaluation/results/faiss_vs_qdrant.csv"
RAGAS_OUTPUT_PATH = "evaluation/results/faiss_vs_qdrant_ragas.json"
TOP_K = 5

_JUDGE_PROMPT = """\
You are evaluating retrieval quality for a RAG system.

Question: {question}

Retrieved chunks:
{chunks}

Rate how relevant and useful these chunks are for answering the question.
Consider: Do the chunks contain the information needed? Are they on-topic?

Respond with JSON only: {{"score": <float 0.0-1.0>, "reason": "<one sentence>"}}"""


def load_queries_with_ground_truth(n: int = 10) -> Tuple[List[str], List[str]]:
    """Load the first n questions and ground-truth answers from the evaluation CSV."""
    df = pd.read_csv(DATASET_PATH)
    df.columns = df.columns.str.lower()
    questions = df["question"].head(n).tolist()
    ground_truths = df["ground_truth_answer"].head(n).tolist()
    return questions, ground_truths


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


def judge_context_quality(query: str, docs: list, llm) -> tuple[float, str]:
    """Ask the LLM to rate retrieved context relevance. Returns (score, reason)."""
    chunks_text = "\n\n".join(
        f"[Chunk {i + 1}]\n{doc.page_content[:400]}" for i, doc in enumerate(docs)
    )
    prompt = _JUDGE_PROMPT.format(question=query, chunks=chunks_text)
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        content = re.sub(r"```(?:json)?\s*|\s*```", "", content).strip()
        parsed = json.loads(content)
        score = float(parsed.get("score", 0.5))
        reason = str(parsed.get("reason", ""))
        return max(0.0, min(1.0, score)), reason
    except Exception:
        nums = re.findall(r"\b(0\.\d+|1\.0|0|1)\b", content if "content" in dir() else "")
        score = float(nums[0]) if nums else 0.5
        return score, ""


def run_ragas(questions, ground_truths, answers_a, contexts_a, answers_b, contexts_b):
    """
    Run RAGAS on pre-collected answers/contexts for two stores.

    Returns (scores_a, scores_b, per_query_df_a, per_query_df_b) where scores are
    dicts of metric → float and per_query_dfs have one row per question.
    """
    from ragas import evaluate
    from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness, AnswerRelevancy
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from datasets import Dataset
    from generation.gemini_llm import GeminiChatModel
    from embeddings.gemini_embeddings import GeminiEmbeddings as _GeminiEmbeddings

    evaluator_llm = LangchainLLMWrapper(GeminiChatModel())
    evaluator_embeddings = LangchainEmbeddingsWrapper(_GeminiEmbeddings())
    metrics = [
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
    ]
    metric_names = [m.name for m in metrics]

    def _evaluate_store(answers, contexts):
        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(dataset, metrics=metrics)
        df_result = result.to_pandas()
        scores = {
            col: float(df_result[col].mean())
            for col in metric_names
            if col in df_result.columns
        }
        return scores, df_result[metric_names] if all(c in df_result.columns for c in metric_names) else df_result

    print("\nRunning RAGAS for FAISS...")
    scores_faiss, df_faiss = _evaluate_store(answers_a, contexts_a)
    print("Running RAGAS for Qdrant...")
    scores_qdrant, df_qdrant = _evaluate_store(answers_b, contexts_b)

    return scores_faiss, scores_qdrant, df_faiss, df_qdrant


def main():
    print("Loading embeddings and vector stores...")
    embeddings = GeminiEmbeddings()

    faiss_store = load_faiss(FAISS_INDEX_PATH, embeddings)
    qdrant_store = load_qdrant(embeddings)

    faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": TOP_K})
    qdrant_retriever = qdrant_store.as_retriever(search_kwargs={"k": TOP_K})

    from generation.gemini_llm import GeminiChatModel
    from generation.rag_chain import build_rag_chain

    llm = GeminiChatModel()
    faiss_chain = build_rag_chain(faiss_retriever, llm)
    qdrant_chain = build_rag_chain(qdrant_retriever, llm)

    questions, ground_truths = load_queries_with_ground_truth(10)
    print(f"Loaded {len(questions)} queries from {DATASET_PATH}\n")

    rows = []
    faiss_answers, faiss_contexts_list = [], []
    qdrant_answers, qdrant_contexts_list = [], []

    for i, query in enumerate(questions, start=1):
        print(f"Query {i}/{len(questions)}: {query[:80]}")

        faiss_docs, faiss_latency = retrieve_with_timing(faiss_retriever, query)
        qdrant_docs, qdrant_latency = retrieve_with_timing(qdrant_retriever, query)
        overlap = compute_overlap(faiss_docs, qdrant_docs)

        faiss_quality, faiss_reason = judge_context_quality(query, faiss_docs, llm)
        qdrant_quality, qdrant_reason = judge_context_quality(query, qdrant_docs, llm)

        delta = faiss_quality - qdrant_quality
        winner = "tie" if abs(delta) < 0.05 else ("faiss" if delta > 0 else "qdrant")

        print(
            f"  FAISS : {faiss_latency:.1f} ms, quality={faiss_quality:.2f} | "
            f"Qdrant: {qdrant_latency:.1f} ms, quality={qdrant_quality:.2f} | "
            f"Overlap: {overlap} | Winner: {winner}"
        )

        # Collect answers + contexts for RAGAS (generate answer here, reuse contexts)
        try:
            faiss_answer = faiss_chain.invoke(query)
        except Exception:
            faiss_answer = ""
        try:
            qdrant_answer = qdrant_chain.invoke(query)
        except Exception:
            qdrant_answer = ""

        faiss_answers.append(faiss_answer)
        qdrant_answers.append(qdrant_answer)
        faiss_contexts_list.append([d.page_content for d in faiss_docs])
        qdrant_contexts_list.append([d.page_content for d in qdrant_docs])

        rows.append(
            {
                "query": query,
                "faiss_latency_ms": round(faiss_latency, 2),
                "qdrant_latency_ms": round(qdrant_latency, 2),
                "faiss_chunks": len(faiss_docs),
                "qdrant_chunks": len(qdrant_docs),
                "overlap_count": overlap,
                "faiss_quality": round(faiss_quality, 3),
                "qdrant_quality": round(qdrant_quality, 3),
                "quality_winner": winner,
                "faiss_reason": faiss_reason,
                "qdrant_reason": qdrant_reason,
            }
        )

    # --- RAGAS evaluation ---
    ragas_faiss, ragas_qdrant, df_ragas_faiss, df_ragas_qdrant = run_ragas(
        questions, ground_truths,
        faiss_answers, faiss_contexts_list,
        qdrant_answers, qdrant_contexts_list,
    )

    # Merge per-query RAGAS scores into the rows DataFrame
    df = pd.DataFrame(rows)
    for col in df_ragas_faiss.columns:
        df[f"faiss_{col}"] = df_ragas_faiss[col].values
    for col in df_ragas_qdrant.columns:
        df[f"qdrant_{col}"] = df_ragas_qdrant[col].values

    output_path = pathlib.Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {OUTPUT_PATH}")

    # Save aggregate RAGAS scores as JSON
    ragas_summary = {"faiss": ragas_faiss, "qdrant": ragas_qdrant}
    ragas_output_path = pathlib.Path(RAGAS_OUTPUT_PATH)
    with open(ragas_output_path, "w") as fh:
        json.dump(ragas_summary, fh, indent=2)
    print(f"RAGAS scores saved to {RAGAS_OUTPUT_PATH}")

    # Summary
    winner_counts = df["quality_winner"].value_counts()
    ragas_metrics = list(ragas_faiss.keys())

    print("\n--- Summary ---")
    print(f"Avg FAISS latency  : {df['faiss_latency_ms'].mean():.1f} ms")
    print(f"Avg Qdrant latency : {df['qdrant_latency_ms'].mean():.1f} ms")
    print(f"Avg overlap count  : {df['overlap_count'].mean():.1f} / {TOP_K}")
    print(f"Avg FAISS quality  : {df['faiss_quality'].mean():.3f}")
    print(f"Avg Qdrant quality : {df['qdrant_quality'].mean():.3f}")
    print(f"Quality wins       : FAISS={winner_counts.get('faiss', 0)}  "
          f"Qdrant={winner_counts.get('qdrant', 0)}  "
          f"Tie={winner_counts.get('tie', 0)}")
    print("\n--- RAGAS Metrics ---")
    print(f"{'Metric':<25} {'FAISS':>8} {'Qdrant':>8} {'Winner':>8}")
    print("-" * 55)
    for metric in ragas_metrics:
        f_val = ragas_faiss.get(metric)
        q_val = ragas_qdrant.get(metric)
        if f_val is None or q_val is None:
            w = "n/a"
        elif abs(f_val - q_val) < 0.01:
            w = "tie"
        elif f_val > q_val:
            w = "faiss"
        else:
            w = "qdrant"
        f_str = f"{f_val:.4f}" if f_val is not None else "n/a"
        q_str = f"{q_val:.4f}" if q_val is not None else "n/a"
        print(f"  {metric:<23} {f_str:>8} {q_str:>8} {w:>8}")
    print(f"\nTotal queries      : {len(df)}")


if __name__ == "__main__":
    main()
