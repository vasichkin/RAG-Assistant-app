"""
RAGAS Evaluation Results page.

Loads JSON result files from evaluation/results/ matching *_ragas.json
and displays metric scores, bar charts, and cross-phase comparisons.

Run evaluation first:
    python -m scripts.run_evaluation
"""

import json
import pathlib
import re

import pandas as pd
import streamlit as st

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "evaluation" / "results"

METRIC_LABELS = {
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
}

st.title("RAGAS Evaluation Results")


def load_ragas_results() -> dict[str, dict]:
    """Load all *_ragas.json files from the results directory.

    Returns a dict mapping phase label → scores dict.
    """
    if not RESULTS_DIR.exists():
        return {}

    results = {}
    for json_file in sorted(RESULTS_DIR.glob("*_ragas.json")):
        try:
            with open(json_file) as fh:
                data = json.load(fh)
            # Derive a human-readable phase label from the filename
            stem = json_file.stem  # e.g. "phase1_ragas"
            label = re.sub(r"(\d+)", r" \1", stem.replace("_ragas", "")).strip().title()
            results[label] = data
        except (json.JSONDecodeError, OSError) as exc:
            st.warning(f"Could not read {json_file.name}: {exc}")

    return results


def render_phase_results(label: str, scores: dict) -> None:
    """Render metric scores and a bar chart for one phase."""
    st.subheader(f"Phase: {label}")

    # Metric score cards
    known_metrics = [k for k in METRIC_LABELS if k in scores]
    other_metrics = [k for k in scores if k not in METRIC_LABELS]
    all_metrics = known_metrics + other_metrics

    if all_metrics:
        cols = st.columns(min(len(all_metrics), 4))
        for i, metric in enumerate(all_metrics):
            col = cols[i % 4]
            display_name = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
            raw_val = scores[metric]
            try:
                formatted = f"{float(raw_val):.3f}"
            except (TypeError, ValueError):
                formatted = str(raw_val)
            with col:
                st.metric(label=display_name, value=formatted)

    # Bar chart
    chart_data = {}
    for metric, val in scores.items():
        try:
            chart_data[METRIC_LABELS.get(metric, metric)] = float(val)
        except (TypeError, ValueError):
            pass

    if chart_data:
        chart_df = pd.DataFrame(
            {"Score": list(chart_data.values())},
            index=list(chart_data.keys()),
        )
        st.bar_chart(chart_df, use_container_width=True)

    with st.expander("Raw JSON"):
        st.json(scores)


def render_comparison(all_results: dict[str, dict]) -> None:
    """Render a side-by-side comparison table across phases."""
    st.subheader("Cross-Phase Comparison")

    # Build a combined DataFrame: rows = metrics, columns = phases
    all_metric_keys = set()
    for scores in all_results.values():
        all_metric_keys.update(scores.keys())

    rows = {}
    for metric in sorted(all_metric_keys):
        display_name = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
        row = {}
        for phase_label, scores in all_results.items():
            try:
                row[phase_label] = round(float(scores.get(metric, float("nan"))), 4)
            except (TypeError, ValueError):
                row[phase_label] = None
        rows[display_name] = row

    comparison_df = pd.DataFrame(rows).T
    st.dataframe(comparison_df, use_container_width=True)

    # Multi-phase bar chart
    try:
        st.bar_chart(comparison_df, use_container_width=True)
    except Exception as e:
        st.warning(f"Chart unavailable: {e}")


# --- Main render logic ---

all_results = load_ragas_results()

if not all_results:
    st.info(
        "No RAGAS evaluation results found yet.\n\n"
        "Run the evaluation script to generate results:\n\n"
        "```bash\n"
        "python -m scripts.run_evaluation\n"
        "```\n\n"
        f"Results will be saved as JSON files in `{RESULTS_DIR}`.\n\n"
        "The evaluation runs 4 metrics: **Context Precision**, **Context Recall**, "
        "**Faithfulness**, and **Answer Relevancy**."
    )
else:
    # Render each phase individually
    for phase_label, scores in all_results.items():
        render_phase_results(phase_label, scores)
        st.divider()

    # Show comparison only when there are multiple phases
    if len(all_results) > 1:
        render_comparison(all_results)
