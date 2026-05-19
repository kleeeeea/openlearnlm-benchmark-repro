"""Streamlit report for OpenLearnLM benchmark results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
RESPONSES_ROOT = PROJECT_ROOT / "outputs" / "responses"

DEFAULT_EXPERIMENT_MODEL_REPORT_FILE = (
    REPORTS_DIR / "report_20260518_222155.json"
)

DEFAULT_EXPERIMENT_GROUP_RESPONSE_FILE = (
    RESPONSES_ROOT / "01_기능_skills" / "Qwen3-4B-Instruct-2507.jsonl"
)

DEFAULT_BASELINE_GROUP_RESPONSE_FILE = (
    RESPONSES_ROOT / "01_기능_skills" / "Qwen3-4B-Instruct-2507-Official.jsonl"
)


def find_latest_report() -> Path | None:
    reports = sorted(REPORTS_DIR.glob("report_*.json"))
    return reports[-1] if reports else None


def list_response_files() -> list[Path]:
    return sorted(RESPONSES_ROOT.glob("*/*.jsonl"))


@st.cache_data(show_spinner=False)
def load_report(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def load_responses(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                rows.append(
                    {
                        "line_no": line_no,
                        "success": False,
                        "error": f"Invalid JSON: {exc}",
                    }
                )
                continue
            row["line_no"] = line_no
            rows.append(row)
    return rows


def model_summary_frame(report: dict[str, Any]) -> pd.DataFrame:
    records = []
    for model, stats in report.get("models", {}).items():
        records.append(
            {
                "model": model,
                "total": stats.get("total", 0),
                "score": stats.get("score"),
                "accuracy": stats.get("accuracy", 0.0),
                "successful": stats.get("successful", 0),
                "failed": stats.get("failed", 0),
                "correct": stats.get("correct", 0),
                "avg_latency_ms": stats.get("avg_latency_ms", 0.0),
                "prompt_tokens": stats.get("total_prompt_tokens", 0),
                "completion_tokens": stats.get("total_completion_tokens", 0),
            }
        )
    return pd.DataFrame.from_records(records)


def response_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    records = []
    for row in rows:
        check_result = row.get("check_result") or {}
        metadata = row.get("metadata") or {}
        usage = row.get("usage") or {}
        records.append(
            {
                "line_no": row.get("line_no"),
                "item_id": row.get("item_id"),
                "model": row.get("model", ""),
                "success": row.get("success", False),
                "is_correct": row.get("is_correct"),
                "score": check_result.get("score"),
                "check_type": check_result.get("check_type", ""),
                "rubric_source": check_result.get("rubric_source", ""),
                "reasoning": check_result.get("reasoning", row.get("error", "")),
                "latency_ms": row.get("latency_ms"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "question_type": metadata.get("question_type", ""),
                "difficulty": metadata.get("difficulty", ""),
                "domain": metadata.get("domain", ""),
                "scenario": metadata.get("scenario", ""),
                "question": row.get("question", ""),
                "model_answer": row.get("model_answer", ""),
            }
        )
    return pd.DataFrame.from_records(records)


def summarize_response_frame(label: str, responses: pd.DataFrame) -> dict[str, Any]:
    if responses.empty:
        return {
            "group": label,
            "model": "",
            "total": 0,
            "successful": 0,
            "correct": 0,
            "accuracy": 0.0,
            "score": 0.0,
            "avg_latency_ms": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    model = ""
    if "model" in responses and responses["model"].notna().any():
        model = str(responses["model"].dropna().iloc[0])

    score_series = pd.to_numeric(responses["score"], errors="coerce")
    latency_series = pd.to_numeric(responses["latency_ms"], errors="coerce")
    prompt_tokens = pd.to_numeric(responses["prompt_tokens"], errors="coerce").fillna(0)
    completion_tokens = pd.to_numeric(
        responses["completion_tokens"], errors="coerce"
    ).fillna(0)

    total = int(len(responses))
    successful = int(responses["success"].fillna(False).sum())
    correct = int(responses["is_correct"].fillna(False).sum())
    return {
        "group": label,
        "model": model,
        "total": total,
        "successful": successful,
        "correct": correct,
        "accuracy": correct / successful if successful else 0.0,
        "score": safe_float(score_series.dropna().mean()),
        "avg_latency_ms": safe_float(latency_series.dropna().mean()),
        "prompt_tokens": int(prompt_tokens.sum()),
        "completion_tokens": int(completion_tokens.sum()),
    }


def comparison_summary_frame(
    baseline: pd.DataFrame, experiment: pd.DataFrame
) -> pd.DataFrame:
    summary = pd.DataFrame.from_records(
        [
            summarize_response_frame("baseline", baseline),
            summarize_response_frame("experiment", experiment),
        ]
    )
    if summary.empty:
        return summary

    baseline_row = summary[summary["group"] == "baseline"].iloc[0]
    deltas = []
    for row in summary.itertuples():
        deltas.append(
            {
                "score_delta_vs_baseline": safe_float(row.score)
                - safe_float(baseline_row["score"]),
                "accuracy_delta_vs_baseline": safe_float(row.accuracy)
                - safe_float(baseline_row["accuracy"]),
                "latency_delta_ms_vs_baseline": safe_float(row.avg_latency_ms)
                - safe_float(baseline_row["avg_latency_ms"]),
            }
        )
    return pd.concat([summary, pd.DataFrame.from_records(deltas)], axis=1)


def pair_response_frames(
    baseline: pd.DataFrame, experiment: pd.DataFrame
) -> pd.DataFrame:
    columns = [
        "item_id",
        "question",
        "baseline_score",
        "experiment_score",
        "score_delta",
        "baseline_correct",
        "experiment_correct",
        "correct_changed",
        "baseline_latency_ms",
        "experiment_latency_ms",
        "latency_delta_ms",
        "difficulty",
        "domain",
        "scenario",
        "baseline_reasoning",
        "experiment_reasoning",
    ]
    if baseline.empty or experiment.empty:
        return pd.DataFrame(columns=columns)

    merged = baseline.merge(
        experiment,
        on="item_id",
        how="inner",
        suffixes=("_baseline", "_experiment"),
    )

    def first_available(row: pd.Series, *names: str) -> Any:
        for name in names:
            value = row.get(name)
            if pd.notna(value):
                return value
        return ""

    records = []
    for _, row in merged.iterrows():
        baseline_score = row.get("score_baseline")
        experiment_score = row.get("score_experiment")
        baseline_latency = row.get("latency_ms_baseline")
        experiment_latency = row.get("latency_ms_experiment")
        records.append(
            {
                "item_id": row.get("item_id"),
                "question": first_available(row, "question_experiment", "question_baseline"),
                "baseline_score": baseline_score,
                "experiment_score": experiment_score,
                "score_delta": safe_float(experiment_score)
                - safe_float(baseline_score),
                "baseline_correct": row.get("is_correct_baseline"),
                "experiment_correct": row.get("is_correct_experiment"),
                "correct_changed": row.get("is_correct_baseline")
                != row.get("is_correct_experiment"),
                "baseline_latency_ms": baseline_latency,
                "experiment_latency_ms": experiment_latency,
                "latency_delta_ms": safe_float(experiment_latency)
                - safe_float(baseline_latency),
                "difficulty": first_available(
                    row, "difficulty_experiment", "difficulty_baseline"
                ),
                "domain": first_available(row, "domain_experiment", "domain_baseline"),
                "scenario": first_available(
                    row, "scenario_experiment", "scenario_baseline"
                ),
                "baseline_reasoning": row.get("reasoning_baseline", ""),
                "experiment_reasoning": row.get("reasoning_experiment", ""),
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def render_metric_row(summary: pd.DataFrame, responses: pd.DataFrame) -> None:
    total_items = int(responses.shape[0])
    avg_score = safe_float(responses["score"].dropna().mean()) if "score" in responses else 0.0
    accuracy = (
        safe_float(responses["is_correct"].fillna(False).mean())
        if "is_correct" in responses and total_items
        else 0.0
    )
    avg_latency = (
        safe_float(responses["latency_ms"].dropna().mean())
        if "latency_ms" in responses
        else 0.0
    )

    cols = st.columns(4)
    cols[0].metric("Evaluated items", f"{total_items:,}")
    cols[1].metric("Average score", f"{avg_score:.2f}")
    cols[2].metric("Accuracy", f"{accuracy * 100:.1f}%")
    cols[3].metric("Avg latency", f"{avg_latency:,.0f} ms")

    if not summary.empty:
        best_row = summary.sort_values(["score", "accuracy"], ascending=False).iloc[0]
        st.caption(
            f"Top model in selected report: {best_row['model']} "
            f"(score {safe_float(best_row['score']):.2f}, "
            f"accuracy {safe_float(best_row['accuracy']) * 100:.1f}%)."
        )


def render_performance(summary: pd.DataFrame) -> None:
    st.subheader("Overall performance")
    if summary.empty:
        st.warning("No model summary data found in the selected report.")
        return

    chart_data = summary[["model", "score", "accuracy"]].set_index("model")
    st.bar_chart(chart_data, height=280)
    st.dataframe(
        summary,
        width="stretch",
        hide_index=True,
        column_config={
            "accuracy": st.column_config.NumberColumn("accuracy", format="%.2f"),
            "score": st.column_config.NumberColumn("score", format="%.2f"),
            "avg_latency_ms": st.column_config.NumberColumn(
                "avg_latency_ms", format="%.0f"
            ),
        },
    )


def render_overall_comparison(
    baseline: pd.DataFrame, experiment: pd.DataFrame
) -> pd.DataFrame:
    st.subheader("Overall side-by-side comparison")
    comparison = comparison_summary_frame(baseline, experiment)
    if comparison.empty:
        st.warning("No baseline or experiment rows are available for comparison.")
        return comparison

    baseline_row = comparison[comparison["group"] == "baseline"].iloc[0]
    experiment_row = comparison[comparison["group"] == "experiment"].iloc[0]

    cols = st.columns(4)
    cols[0].metric(
        "Score",
        f"{safe_float(experiment_row['score']):.2f}",
        f"{safe_float(experiment_row['score_delta_vs_baseline']):+.2f}",
    )
    cols[1].metric(
        "Accuracy",
        f"{safe_float(experiment_row['accuracy']) * 100:.1f}%",
        f"{safe_float(experiment_row['accuracy_delta_vs_baseline']) * 100:+.1f}%",
    )
    cols[2].metric(
        "Avg latency",
        f"{safe_float(experiment_row['avg_latency_ms']):,.0f} ms",
        f"{safe_float(experiment_row['latency_delta_ms_vs_baseline']):+,.0f} ms",
        delta_color="inverse",
    )
    cols[3].metric(
        "Items",
        f"{int(experiment_row['total']):,}",
        f"{int(experiment_row['total']) - int(baseline_row['total']):+,}",
    )

    chart_data = comparison[["group", "score", "accuracy", "avg_latency_ms"]].set_index(
        "group"
    )
    st.bar_chart(chart_data[["score", "accuracy"]], height=260)
    st.dataframe(
        comparison,
        width="stretch",
        hide_index=True,
        column_config={
            "score": st.column_config.NumberColumn("score", format="%.2f"),
            "accuracy": st.column_config.NumberColumn("accuracy", format="%.3f"),
            "avg_latency_ms": st.column_config.NumberColumn(
                "avg_latency_ms", format="%.0f"
            ),
            "score_delta_vs_baseline": st.column_config.NumberColumn(
                "score_delta_vs_baseline", format="%+.2f"
            ),
            "accuracy_delta_vs_baseline": st.column_config.NumberColumn(
                "accuracy_delta_vs_baseline", format="%+.3f"
            ),
            "latency_delta_ms_vs_baseline": st.column_config.NumberColumn(
                "latency_delta_ms_vs_baseline", format="%+.0f"
            ),
        },
    )
    return comparison


def render_question_comparison(
    baseline_rows: list[dict[str, Any]],
    experiment_rows: list[dict[str, Any]],
    baseline: pd.DataFrame,
    experiment: pd.DataFrame,
) -> None:
    st.subheader("Question-level side-by-side comparison")
    paired = pair_response_frames(baseline, experiment)
    baseline_ids = set(baseline["item_id"].dropna()) if "item_id" in baseline else set()
    experiment_ids = (
        set(experiment["item_id"].dropna()) if "item_id" in experiment else set()
    )
    common_ids = baseline_ids & experiment_ids
    st.caption(
        f"Common answered items: {len(common_ids):,}. "
        f"Baseline-only items hidden: {len(baseline_ids - experiment_ids):,}. "
        f"Experiment-only items hidden: {len(experiment_ids - baseline_ids):,}."
    )
    if paired.empty:
        st.warning("No paired question rows are available.")
        return

    with st.sidebar:
        st.header("Comparison filters")
        only_changed = st.checkbox("Only changed correctness", value=False)
        search = st.text_input("Search paired question / answer / reasoning")
        score_min = safe_float(
            pd.to_numeric(
                pd.concat([paired["baseline_score"], paired["experiment_score"]]),
                errors="coerce",
            )
            .dropna()
            .min(),
            0.0,
        )
        score_max = safe_float(
            pd.to_numeric(
                pd.concat([paired["baseline_score"], paired["experiment_score"]]),
                errors="coerce",
            )
            .dropna()
            .max(),
            10.0,
        )
        score_range = st.slider(
            "Paired score range",
            min_value=0.0,
            max_value=10.0,
            value=(score_min, score_max),
            step=0.5,
        )

    filtered = paired.copy()
    if only_changed:
        filtered = filtered[filtered["correct_changed"]]

    score_values = pd.to_numeric(filtered["experiment_score"], errors="coerce")
    filtered = filtered[
        score_values.isna()
        | ((score_values >= score_range[0]) & (score_values <= score_range[1]))
    ]

    if search:
        text = search.casefold()
        baseline_answers = baseline.set_index("item_id")["model_answer"]
        experiment_answers = experiment.set_index("item_id")["model_answer"]
        haystack = (
            filtered["question"].fillna("")
            + "\n"
            + filtered["baseline_reasoning"].fillna("")
            + "\n"
            + filtered["experiment_reasoning"].fillna("")
            + "\n"
            + filtered["item_id"].map(baseline_answers).fillna("")
            + "\n"
            + filtered["item_id"].map(experiment_answers).fillna("")
        ).str.casefold()
        filtered = filtered[haystack.str.contains(text, regex=False)]

    display_cols = [
        "item_id",
        "baseline_score",
        "experiment_score",
        "score_delta",
        "baseline_correct",
        "experiment_correct",
        "latency_delta_ms",
        "difficulty",
        "domain",
        "question",
    ]
    st.dataframe(
        filtered[display_cols],
        width="stretch",
        hide_index=True,
        height=320,
        column_config={
            "question": st.column_config.TextColumn("question", width="large"),
            "baseline_score": st.column_config.NumberColumn(
                "baseline_score", format="%.1f"
            ),
            "experiment_score": st.column_config.NumberColumn(
                "experiment_score", format="%.1f"
            ),
            "score_delta": st.column_config.NumberColumn(
                "score_delta", format="%+.1f"
            ),
            "latency_delta_ms": st.column_config.NumberColumn(
                "latency_delta_ms", format="%+.0f"
            ),
        },
    )
    st.caption(f"Showing {len(filtered):,} of {len(paired):,} paired questions.")

    labels = [
        f"item {row.item_id} | baseline {row.baseline_score} | experiment {row.experiment_score}"
        for row in filtered.itertuples()
    ]
    if not labels:
        st.info("No paired item matches the current filters.")
        return

    selected_label = st.selectbox("Select a paired item", labels)
    selected_item_id = int(selected_label.split(" | ", 1)[0].replace("item ", ""))
    baseline_by_item = {row.get("item_id"): row for row in baseline_rows}
    experiment_by_item = {row.get("item_id"): row for row in experiment_rows}
    baseline_item = baseline_by_item.get(selected_item_id, {})
    experiment_item = experiment_by_item.get(selected_item_id, {})

    st.markdown("**Question**")
    st.write(
        experiment_item.get("question")
        or baseline_item.get("question")
        or filtered[filtered["item_id"] == selected_item_id]["question"].iloc[0]
    )

    baseline_col, experiment_col = st.columns(2)
    render_compared_item(baseline_col, "Baseline", baseline_item)
    render_compared_item(experiment_col, "Experiment", experiment_item)


def render_compared_item(container: Any, title: str, item: dict[str, Any]) -> None:
    check_result = item.get("check_result") or {}
    with container:
        st.markdown(f"**{title}: {item.get('model', 'missing')}**")
        metric_cols = st.columns(3)
        metric_cols[0].metric("Score", check_result.get("score", "n/a"))
        metric_cols[1].metric("Correct", str(item.get("is_correct", "n/a")))
        metric_cols[2].metric("Latency", f"{item.get('latency_ms', 0):,} ms")
        tabs = st.tabs(["Response", "Scoring", "Raw"])
        with tabs[0]:
            st.markdown(item.get("model_answer") or item.get("raw_content") or "")
        with tabs[1]:
            st.json(check_result)
        with tabs[2]:
            st.write(item.get("raw_content") or "")


def render_response_table(responses: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Per-instance results")
    if responses.empty:
        st.warning("No per-instance responses found in the selected JSONL file.")
        return responses

    with st.sidebar:
        st.header("Filters")
        correctness = st.multiselect(
            "Correctness",
            ["correct", "incorrect", "unknown"],
            default=["correct", "incorrect", "unknown"],
        )
        check_types = sorted(v for v in responses["check_type"].dropna().unique() if v)
        selected_check_types = st.multiselect(
            "Check type", check_types, default=check_types
        )
        search = st.text_input("Search question / answer / reasoning")
        score_min, score_max = 0.0, 10.0
        if responses["score"].notna().any():
            score_min = safe_float(responses["score"].min())
            score_max = safe_float(responses["score"].max())
        score_range = st.slider(
            "Score range",
            min_value=0.0,
            max_value=10.0,
            value=(score_min, score_max),
            step=0.5,
        )

    filtered = responses.copy()
    status_map = {
        True: "correct",
        False: "incorrect",
    }
    filtered["_correctness"] = filtered["is_correct"].map(status_map).fillna("unknown")
    filtered = filtered[filtered["_correctness"].isin(correctness)]

    if selected_check_types:
        filtered = filtered[filtered["check_type"].isin(selected_check_types)]

    score_values = pd.to_numeric(filtered["score"], errors="coerce")
    filtered = filtered[
        score_values.isna()
        | ((score_values >= score_range[0]) & (score_values <= score_range[1]))
    ]

    if search:
        text = search.casefold()
        haystack = (
            filtered["question"].fillna("")
            + "\n"
            + filtered["model_answer"].fillna("")
            + "\n"
            + filtered["reasoning"].fillna("")
        ).str.casefold()
        filtered = filtered[haystack.str.contains(text, regex=False)]

    display_cols = [
        "line_no",
        "item_id",
        "success",
        "is_correct",
        "score",
        "check_type",
        "rubric_source",
        "latency_ms",
        "difficulty",
        "domain",
        "question",
        "reasoning",
    ]
    st.dataframe(
        filtered[display_cols],
        width="stretch",
        hide_index=True,
        height=360,
        column_config={
            "question": st.column_config.TextColumn("question", width="large"),
            "reasoning": st.column_config.TextColumn("reasoning", width="large"),
            "score": st.column_config.NumberColumn("score", format="%.1f"),
            "latency_ms": st.column_config.NumberColumn("latency_ms", format="%.0f"),
        },
    )
    st.caption(f"Showing {len(filtered):,} of {len(responses):,} response rows.")
    return filtered


def render_detail(rows: list[dict[str, Any]], filtered: pd.DataFrame) -> None:
    st.subheader("Response detail")
    if filtered.empty:
        st.info("No row matches the current filters.")
        return

    labels = [
        f"line {int(row.line_no)} | item {row.item_id} | score {row.score}"
        for row in filtered.itertuples()
    ]
    selected_label = st.selectbox("Select an item", labels)
    selected_line = int(selected_label.split(" | ", 1)[0].replace("line ", ""))
    selected = next(row for row in rows if row.get("line_no") == selected_line)
    check_result = selected.get("check_result") or {}
    metadata = selected.get("metadata") or {}

    meta_cols = st.columns(5)
    meta_cols[0].metric("Item", selected.get("item_id", ""))
    meta_cols[1].metric("Score", check_result.get("score", "n/a"))
    meta_cols[2].metric("Correct", str(selected.get("is_correct", "n/a")))
    meta_cols[3].metric("Latency", f"{selected.get('latency_ms', 0):,} ms")
    meta_cols[4].metric("Check", check_result.get("check_type", "n/a"))

    st.markdown("**Question**")
    st.write(selected.get("question", ""))

    answer_tabs = st.tabs(["Model response", "Expected answer", "Scoring", "Metadata"])
    with answer_tabs[0]:
        st.markdown(selected.get("model_answer") or selected.get("raw_content") or "")
        with st.expander("Raw content", expanded=False):
            st.write(selected.get("raw_content") or "")
        thinking = selected.get("thinking_content")
        if thinking:
            with st.expander("Thinking content", expanded=False):
                st.write(thinking)
    with answer_tabs[1]:
        st.write(selected.get("expected_answer", ""))
    with answer_tabs[2]:
        st.json(check_result)
    with answer_tabs[3]:
        st.json(metadata)


def main() -> None:
    st.set_page_config(
        page_title="OpenLearnLM Benchmark Report",
        page_icon="",
        layout="wide",
    )
    st.title("OpenLearnLM Benchmark Report")

    response_files = list_response_files()
    latest_report = find_latest_report()
    default_report = (
        DEFAULT_EXPERIMENT_MODEL_REPORT_FILE
        if DEFAULT_EXPERIMENT_MODEL_REPORT_FILE.exists()
        else latest_report
    )
    default_experiment_response = (
        DEFAULT_EXPERIMENT_GROUP_RESPONSE_FILE
        if DEFAULT_EXPERIMENT_GROUP_RESPONSE_FILE.exists()
        else (response_files[0] if response_files else None)
    )
    default_baseline_response = (
        DEFAULT_BASELINE_GROUP_RESPONSE_FILE
        if DEFAULT_BASELINE_GROUP_RESPONSE_FILE.exists()
        else (response_files[0] if response_files else None)
    )

    with st.sidebar:
        st.header("Data sources")
        report_options = sorted(REPORTS_DIR.glob("report_*.json"))
        if not report_options:
            st.error(f"No report JSON files found under {REPORTS_DIR}")
            st.stop()
        report_path = st.selectbox(
            "Overall report",
            report_options,
            index=report_options.index(default_report) if default_report in report_options else 0,
            format_func=lambda path: path.name,
        )

        if not response_files:
            st.error(f"No response JSONL files found under {RESPONSES_ROOT}")
            st.stop()
        experiment_response_path = st.selectbox(
            "Experiment responses",
            response_files,
            index=response_files.index(default_experiment_response)
            if default_experiment_response in response_files
            else 0,
            format_func=lambda path: f"{path.parent.name}/{path.name}",
        )
        baseline_response_path = st.selectbox(
            "Baseline responses",
            response_files,
            index=response_files.index(default_baseline_response)
            if default_baseline_response in response_files
            else 0,
            format_func=lambda path: f"{path.parent.name}/{path.name}",
        )

    report = load_report(str(report_path))
    experiment_rows = load_responses(str(experiment_response_path))
    baseline_rows = load_responses(str(baseline_response_path))
    summary = model_summary_frame(report)
    experiment_responses = response_frame(experiment_rows)
    baseline_responses = response_frame(baseline_rows)

    st.caption(
        f"Generated at {report.get('generated_at', 'unknown')} | "
        f"Report: {report_path.name} | "
        f"Baseline: {baseline_response_path.name} | "
        f"Experiment: {experiment_response_path.name}"
    )
    render_overall_comparison(baseline_responses, experiment_responses)
    render_question_comparison(
        baseline_rows,
        experiment_rows,
        baseline_responses,
        experiment_responses,
    )
    render_performance(summary)

    with st.expander("Single-file experiment detail", expanded=False):
        render_metric_row(summary, experiment_responses)
        filtered = render_response_table(experiment_responses)
        render_detail(experiment_rows, filtered)


if __name__ == "__main__":
    main()
