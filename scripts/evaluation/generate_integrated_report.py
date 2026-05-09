#!/usr/bin/env python3
"""
OpenLearnLM Integrated Benchmark Report Generator

Generates integrated reports across 4 evaluation categories:
- 01_skills (Long Answer)
- 02_content_knowledge (MCQ)
- 03_pedagogical_knowledge (MCQ)
- 04_attitude (Attitude - LLM-as-Judge)
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


class IntegratedReportGenerator:
    """Generate integrated evaluation reports across all benchmark categories"""

    CATEGORIES = {
        "01_기능_skills": {
            "name": "Skills Assessment",
            "type": "long_answer",
            "description": "Educational interaction ability assessment",
        },
        "02_교과지식_content": {
            "name": "Content Knowledge Assessment",
            "type": "mcq",
            "description": "Subject content knowledge assessment (GPQA + CJ-Eval)",
        },
        "03_교육학지식_pedagogical": {
            "name": "Pedagogical Knowledge Assessment",
            "type": "mcq",
            "description": "Teaching methodology knowledge assessment",
        },
        "04_태도_attitude": {
            "name": "Attitude Assessment",
            "type": "attitude",
            "description": "Epistemological/instructional/ethical attitude assessment",
        },
    }

    MODEL_DISPLAY_NAMES = {
        "gpt-5.2": "GPT-5.2",
        "google_gemini-3-pro-preview": "Gemini-3-Pro",
        "anthropic_claude-opus-4.5": "Claude-Opus-4.5",
        "deepseek_deepseek-v3.2": "DeepSeek-v3.2",
        "x-ai_grok-4.1-fast": "Grok-4.1-fast",
        "z-ai_glm-4.7": "GLM-4.7",
        "moonshotai_kimi-k2-thinking": "Kimi-K2-thinking",
    }

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.responses_base = base_dir / "evaluation_responses"
        self.reports_dir = base_dir / "evaluation_reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def load_category_results(self, category: str) -> Dict[str, List[Dict[str, Any]]]:
        """Load results for a specific category"""
        results = {}
        category_dir = self.responses_base / category

        if not category_dir.exists():
            return results

        for response_file in category_dir.glob("*_responses.jsonl"):
            model_id = response_file.stem.replace("_responses", "")
            results[model_id] = []

            with open(response_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        result = json.loads(line)
                        results[model_id].append(result)
                    except json.JSONDecodeError:
                        continue

        return results

    def calculate_category_metrics(
        self, results: Dict[str, List[Dict[str, Any]]], category_type: str
    ) -> Dict[str, Any]:
        """Calculate metrics for a category"""
        metrics = {"models": {}}

        for model_id, model_results in results.items():
            total = len(model_results)
            correct = sum(1 for r in model_results if r.get("is_correct", False))
            failed = sum(1 for r in model_results if not r.get("success", True))

            latencies = [r.get("latency_ms", 0) for r in model_results if r.get("success", True)]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0

            model_metrics = {
                "total": total,
                "correct": correct,
                "failed": failed,
                "accuracy": correct / total if total > 0 else 0,
                "avg_latency_ms": avg_latency,
            }

            # Long answer specific: calculate average score
            if category_type == "long_answer":
                scores = []
                for r in model_results:
                    if r.get("success", True):
                        check_result = r.get("check_result", {})
                        if check_result.get("score") is not None:
                            scores.append(check_result["score"])
                if scores:
                    model_metrics["avg_score"] = sum(scores) / len(scores)
                    model_metrics["pass_rate"] = sum(1 for s in scores if s >= 5) / len(scores)

            # Attitude specific: calculate scores by attitude category and dimension
            if category_type == "attitude":
                scores = []
                by_attitude = defaultdict(lambda: {"scores": [], "total": 0})
                by_dimension = defaultdict(lambda: {"scores": [], "total": 0})
                for r in model_results:
                    if r.get("success", True):
                        check_result = r.get("check_result", {})
                        if check_result.get("score") is not None:
                            score = check_result["score"]
                            scores.append(score)
                            # Group by attitude category
                            attitude_cat = check_result.get("attitude_category") or r.get("metadata", {}).get("attitude_category", "Unknown")
                            by_attitude[attitude_cat]["scores"].append(score)
                            by_attitude[attitude_cat]["total"] += 1
                            # Group by dimension
                            dimension = check_result.get("dimension") or r.get("metadata", {}).get("dimension", "Unknown")
                            by_dimension[dimension]["scores"].append(score)
                            by_dimension[dimension]["total"] += 1
                if scores:
                    model_metrics["avg_score"] = sum(scores) / len(scores)
                    model_metrics["pass_rate"] = sum(1 for s in scores if s >= 5) / len(scores)
                model_metrics["by_attitude_category"] = dict(by_attitude)
                model_metrics["by_dimension"] = dict(by_dimension)

            # Difficulty breakdown
            model_metrics["by_difficulty"] = defaultdict(lambda: {"total": 0, "correct": 0, "scores": []})
            for r in model_results:
                if not r.get("success", True):
                    continue
                difficulty = r.get("metadata", {}).get("difficulty", "Unknown")
                model_metrics["by_difficulty"][difficulty]["total"] += 1
                if r.get("is_correct", False):
                    model_metrics["by_difficulty"][difficulty]["correct"] += 1
                # Collect scores for long answer by difficulty
                if category_type == "long_answer":
                    check_result = r.get("check_result", {})
                    if check_result.get("score") is not None:
                        model_metrics["by_difficulty"][difficulty]["scores"].append(check_result["score"])

            metrics["models"][model_id] = model_metrics

        return metrics

    def get_display_name(self, model_id: str) -> str:
        """Get display name for model"""
        return self.MODEL_DISPLAY_NAMES.get(model_id, model_id)

    def generate_markdown_report(self, all_metrics: Dict[str, Dict[str, Any]]) -> str:
        """Generate integrated Markdown report"""
        lines = []

        lines.append("# OpenLearnLM Integrated Benchmark Evaluation Report")
        lines.append("")
        lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Summary
        lines.append("## Overview")
        lines.append("")
        lines.append("This report presents integrated results across 4 evaluation areas of the OpenLearnLM benchmark.")
        lines.append("")
        lines.append("| Category | Evaluation Type | Description |")
        lines.append("|----------|-----------------|-------------|")
        for cat_id, cat_info in self.CATEGORIES.items():
            if cat_info["type"] == "long_answer":
                eval_type = "Long Answer (10-point scale)"
            elif cat_info["type"] == "attitude":
                eval_type = "Attitude (10-point scale)"
            else:
                eval_type = "MCQ (Exact match)"
            lines.append(f"| {cat_info['name']} | {eval_type} | {cat_info['description']} |")
        lines.append("")

        # Section for each category
        section_num = 1

        for cat_id, cat_info in self.CATEGORIES.items():
            if cat_id not in all_metrics:
                continue

            metrics = all_metrics[cat_id]
            if not metrics.get("models"):
                continue

            section_num += 1
            lines.append(f"## {section_num - 1}. {cat_info['name']}")
            lines.append("")
            lines.append(f"> {cat_info['description']}")
            lines.append("")

            # Get total items count
            total_items = max(m["total"] for m in metrics["models"].values()) if metrics["models"] else 0
            lines.append(f"**Total Items**: {total_items:,}")
            lines.append("")

            if cat_info["type"] == "long_answer":
                # Long answer format
                lines.append("| Model | Pass Rate | Avg Score | Avg Latency |")
                lines.append("|-------|-----------|-----------|-------------|")

                # Sort by pass_rate descending
                sorted_models = sorted(
                    metrics["models"].items(),
                    key=lambda x: x[1].get("pass_rate", 0),
                    reverse=True
                )

                for model_id, stats in sorted_models:
                    display_name = self.get_display_name(model_id)
                    pass_rate = stats.get("pass_rate", stats.get("accuracy", 0)) * 100
                    avg_score = stats.get("avg_score", 0)
                    latency = stats.get("avg_latency_ms", 0) / 1000  # Convert to seconds

                    lines.append(
                        f"| {display_name} | {pass_rate:.2f}% | {avg_score:.2f}/10 | {latency:.1f}s |"
                    )

            elif cat_info["type"] == "attitude":
                # Attitude format (similar to long_answer but with attitude breakdown)
                lines.append("| Model | Avg Score | Avg Latency |")
                lines.append("|-------|-----------|-------------|")

                # Sort by avg_score descending
                sorted_models = sorted(
                    metrics["models"].items(),
                    key=lambda x: x[1].get("avg_score", 0),
                    reverse=True
                )

                for model_id, stats in sorted_models:
                    display_name = self.get_display_name(model_id)
                    avg_score = stats.get("avg_score", 0)
                    latency = stats.get("avg_latency_ms", 0) / 1000

                    lines.append(
                        f"| {display_name} | {avg_score:.2f}/10 | {latency:.1f}s |"
                    )

                lines.append("")

                # Attitude category breakdown
                if sorted_models:
                    first_model_stats = sorted_models[0][1]
                    attitude_categories = list(first_model_stats.get("by_attitude_category", {}).keys())

                    if attitude_categories:
                        lines.append("### Average Score by Attitude Category")
                        lines.append("")
                        header = "| Model |" + " | ".join(attitude_categories) + " |"
                        separator = "|------|" + " | ".join(["--------"] * len(attitude_categories)) + " |"
                        lines.append(header)
                        lines.append(separator)

                        for model_id, stats in sorted_models:
                            display_name = self.get_display_name(model_id)
                            row = f"| {display_name} |"
                            for att_cat in attitude_categories:
                                att_stats = stats.get("by_attitude_category", {}).get(att_cat, {"scores": []})
                                att_scores = att_stats.get("scores", [])
                                if att_scores:
                                    avg = sum(att_scores) / len(att_scores)
                                    row += f" {avg:.2f} |"
                                else:
                                    row += " N/A |"
                            lines.append(row)

            else:
                # MCQ format
                lines.append("| Model | Accuracy | Correct | Avg Latency |")
                lines.append("|-------|----------|---------|-------------|")

                # Sort by accuracy descending
                sorted_models = sorted(
                    metrics["models"].items(),
                    key=lambda x: x[1].get("accuracy", 0),
                    reverse=True
                )

                for model_id, stats in sorted_models:
                    display_name = self.get_display_name(model_id)
                    accuracy = stats.get("accuracy", 0) * 100
                    correct = stats.get("correct", 0)
                    total = stats.get("total", 0)
                    latency = stats.get("avg_latency_ms", 0) / 1000

                    lines.append(
                        f"| {display_name} | {accuracy:.2f}% | {correct}/{total} | {latency:.1f}s |"
                    )

            lines.append("")

            # Difficulty breakdown (if available)
            if metrics["models"]:
                first_model = list(metrics["models"].values())[0]
                difficulties = list(first_model.get("by_difficulty", {}).keys())
                difficulties = [d for d in ["Easy", "Medium", "Hard"] if d in difficulties]

                if difficulties and cat_info["type"] == "long_answer":
                    # Long Answer: Average score by difficulty
                    lines.append("### Average Score by Difficulty")
                    lines.append("")
                    header = "| Model |" + " | ".join(difficulties) + " |"
                    separator = "|------|" + " | ".join(["--------"] * len(difficulties)) + " |"
                    lines.append(header)
                    lines.append(separator)

                    for model_id, stats in sorted_models:
                        display_name = self.get_display_name(model_id)
                        row = f"| {display_name} |"
                        for diff in difficulties:
                            diff_stats = stats.get("by_difficulty", {}).get(diff, {"total": 0, "correct": 0, "scores": []})
                            scores = diff_stats.get("scores", [])
                            if scores:
                                avg_score = sum(scores) / len(scores)
                                row += f" {avg_score:.2f} |"
                            else:
                                row += " N/A |"
                        lines.append(row)

                    lines.append("")

        # Final summary section
        lines.append("---")
        lines.append("")
        lines.append("## Summary Analysis")
        lines.append("")

        # Find best model per category
        lines.append("### Best Performing Model by Category")
        lines.append("")
        lines.append("| Category | Best Model | Score |")
        lines.append("|----------|------------|-------|")

        for cat_id, cat_info in self.CATEGORIES.items():
            if cat_id not in all_metrics or not all_metrics[cat_id].get("models"):
                continue

            metrics = all_metrics[cat_id]["models"]
            if cat_info["type"] == "long_answer":
                best_model = max(metrics.items(), key=lambda x: x[1].get("pass_rate", 0))
                score = f"{best_model[1].get('pass_rate', 0) * 100:.2f}%"
            elif cat_info["type"] == "attitude":
                best_model = max(metrics.items(), key=lambda x: x[1].get("avg_score", 0))
                score = f"{best_model[1].get('avg_score', 0):.2f}/10"
            else:
                best_model = max(metrics.items(), key=lambda x: x[1].get("accuracy", 0))
                score = f"{best_model[1].get('accuracy', 0) * 100:.2f}%"

            display_name = self.get_display_name(best_model[0])
            lines.append(f"| {cat_info['name']} | {display_name} | {score} |")

        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("*Generated by OpenLearnLM Benchmark Evaluation System*")

        return "\n".join(lines)

    def generate_report(self) -> Dict[str, Any]:
        """Generate integrated report across all categories"""
        all_metrics = {}

        print("Loading evaluation results...")

        for cat_id, cat_info in self.CATEGORIES.items():
            print(f"  - {cat_info['name']}")
            results = self.load_category_results(cat_id)
            if results:
                all_metrics[cat_id] = self.calculate_category_metrics(results, cat_info["type"])
                total = sum(len(r) for r in results.values())
                print(f"    Loaded {total:,} evaluations from {len(results)} models")
            else:
                print(f"    No results found")

        if not all_metrics:
            return {"error": "No results found for any category"}

        # Generate Markdown report
        md_report = self.generate_markdown_report(all_metrics)
        md_path = self.reports_dir / f"integrated_benchmark_report_{self.timestamp}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_report)
        print(f"\nMarkdown report saved to: {md_path}")

        # Generate JSON report
        json_path = self.reports_dir / f"integrated_benchmark_report_{self.timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            # Convert defaultdicts to regular dicts
            json_metrics = {}
            for cat_id, cat_metrics in all_metrics.items():
                json_metrics[cat_id] = {
                    "models": {}
                }
                for model_id, model_data in cat_metrics.get("models", {}).items():
                    model_copy = dict(model_data)
                    if "by_difficulty" in model_copy:
                        model_copy["by_difficulty"] = dict(model_copy["by_difficulty"])
                    json_metrics[cat_id]["models"][model_id] = model_copy

            json.dump(json_metrics, f, ensure_ascii=False, indent=2)
        print(f"JSON report saved to: {json_path}")

        return {
            "markdown_report": md_path,
            "json_report": json_path,
            "metrics": all_metrics,
        }


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate integrated OpenLearnLM benchmark report")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(Path(__file__).parent.parent.parent),
        help="Base directory for benchmark data"
    )

    args = parser.parse_args()

    generator = IntegratedReportGenerator(Path(args.base_dir))
    result = generator.generate_report()

    if "error" in result:
        print(f"Error: {result['error']}")
        return 1

    print("\nIntegrated report generation completed!")
    return 0


if __name__ == "__main__":
    exit(main())
