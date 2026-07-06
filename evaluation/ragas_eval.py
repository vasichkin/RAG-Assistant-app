"""
RAGAS evaluation module.

Usage:
    from evaluation.ragas_eval import run_ragas_evaluation
    scores = run_ragas_evaluation(
        csv_path="resources/datasets/rag_evaluation_dataset.csv",
        retriever=retriever,
        chain=chain,
        output_path="evaluation/results/phase1_ragas.json",
    )
"""

import json
import logging
import os
import pandas as pd

logger = logging.getLogger(__name__)


def run_ragas_evaluation(
    csv_path: str,
    retriever,
    chain,
    output_path: str,
    langfuse_handler=None,
) -> dict:
    """
    Load evaluation CSV, run chain for each question, evaluate with RAGAS.

    CSV columns expected: question (or Question), ground_truth / Ground_Truth_Answer,
    contexts (optional).

    Returns dict of metric name → score.
    """
    # Import here so that the module can be imported without ragas installed
    from ragas import evaluate
    from ragas.metrics import (
        ContextPrecision,
        ContextRecall,
        Faithfulness,
        AnswerRelevancy,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from datasets import Dataset

    from generation.gemini_llm import GeminiChatModel
    from embeddings.gemini_embeddings import GeminiEmbeddings as _GeminiEmbeddings

    evaluator_llm = LangchainLLMWrapper(GeminiChatModel())
    evaluator_embeddings = LangchainEmbeddingsWrapper(_GeminiEmbeddings())

    def _make_metrics():
        return [
            ContextPrecision(llm=evaluator_llm),
            ContextRecall(llm=evaluator_llm),
            Faithfulness(llm=evaluator_llm),
            AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        ]

    # --- Load CSV ---
    df = pd.read_csv(csv_path)

    df.columns = df.columns.str.lower()

    # Resolve question column
    if "question" in df.columns:
        question_col = "question"
    else:
        raise ValueError(f"CSV must have a 'question' column. Found: {list(df.columns)}")

    # Resolve ground_truth column
    has_ground_truth = False
    if "ground_truth" in df.columns:
        ground_truth_col = "ground_truth"
        has_ground_truth = True
    elif "ground_truth_answer" in df.columns:
        ground_truth_col = "ground_truth_answer"
        has_ground_truth = True
    else:
        logger.warning(
            "No 'ground_truth' column found; skipping context_precision, context_recall, and faithfulness."
        )

    questions = df[question_col].tolist()
    ground_truths = df[ground_truth_col].tolist() if has_ground_truth else [""] * len(questions)

    # --- Run chain + retriever for each question ---
    answers = []
    contexts_list = []

    for question in questions:
        # Get answer from the RAG chain
        try:
            callbacks = [langfuse_handler] if langfuse_handler else []
            answer = chain.invoke(question, config={"callbacks": callbacks})
            if langfuse_handler:
                langfuse_handler.flush()
        except Exception as exc:
            logger.warning("Chain failed for question %r: %s", question, exc)
            answer = ""
        answers.append(answer)

        # Retrieve context documents
        try:
            docs = retriever.invoke(question)
            contexts = [doc.page_content for doc in docs]
        except Exception as exc:
            logger.warning("Retriever failed for question %r: %s", question, exc)
            contexts = []
        contexts_list.append(contexts)

    # --- Build HuggingFace Dataset ---
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    }
    dataset = Dataset.from_dict(data)

    # --- Select metrics based on available ground truth ---
    if has_ground_truth:
        metrics = _make_metrics()
    else:
        metrics = [AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)]

    # --- Run RAGAS evaluation ---
    result = evaluate(dataset, metrics=metrics)

    # Convert EvaluationResult to a plain dict with float values
    df_result = result.to_pandas()
    metric_cols = [m.name for m in metrics]
    scores: dict = {}
    for col in metric_cols:
        if col in df_result.columns:
            try:
                scores[col] = float(df_result[col].mean())
            except (TypeError, ValueError):
                scores[col] = None

    # --- Save results as JSON ---
    if dirname := os.path.dirname(output_path):
        os.makedirs(dirname, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(scores, fh, indent=2)
    logger.info("RAGAS results saved to %s", output_path)

    return scores
