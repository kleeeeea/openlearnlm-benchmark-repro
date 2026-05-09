#!/usr/bin/env python3
"""
OpenLearnLM Quality Report Generator

Analyzes quality check results and generates summary report.

Usage:
    python quality_reporter.py [--results-file PATH]
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import Config


def load_results(results_file: Path) -> List[Dict[str, Any]]:
    """Load quality check results from JSONL file"""
    results = []
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def analyze_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze quality check results"""

    total = len(results)
    passed = sum(1 for r in results if r.get("evaluation", {}).get("pass", False))
    failed = total - passed

    # Score statistics
    score_sums = defaultdict(float)
    score_counts = defaultdict(int)

    # Domain breakdown
    domain_stats = defaultdict(lambda: {"total": 0, "passed": 0})

    # Difficulty breakdown
    difficulty_stats = defaultdict(lambda: {"total": 0, "passed": 0})

    # Issue frequency
    issue_frequency = defaultdict(int)

    # Failed items details
    failed_items = []

    for result in results:
        eval_data = result.get("evaluation", {})
        metadata = result.get("metadata", {})
        scores = eval_data.get("scores", {})
        is_pass = eval_data.get("pass", False)

        # Aggregate scores
        for criterion, score in scores.items():
            if isinstance(score, (int, float)):
                score_sums[criterion] += score
                score_counts[criterion] += 1

        # Domain stats
        domain = metadata.get("domain", "unknown")
        domain_stats[domain]["total"] += 1
        if is_pass:
            domain_stats[domain]["passed"] += 1

        # Difficulty stats
        difficulty = metadata.get("difficulty", "unknown")
        difficulty_stats[difficulty]["total"] += 1
        if is_pass:
            difficulty_stats[difficulty]["passed"] += 1

        # Issue tracking
        for issue in eval_data.get("issues", []):
            # Normalize issue text (first 50 chars)
            issue_key = issue[:80] if len(issue) > 80 else issue
            issue_frequency[issue_key] += 1

        # Collect failed items
        if not is_pass:
            failed_items.append({
                "item_id": result.get("item_id"),
                "scores": scores,
                "total_score": eval_data.get("total_score"),
                "issues": eval_data.get("issues", []),
                "metadata": metadata
            })

    # Calculate averages
    score_averages = {
        criterion: score_sums[criterion] / score_counts[criterion]
        for criterion in score_sums
    }

    # Sort issues by frequency
    top_issues = sorted(issue_frequency.items(), key=lambda x: -x[1])[:20]

    return {
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total * 100 if total > 0 else 0
        },
        "score_averages": score_averages,
        "domain_breakdown": dict(domain_stats),
        "difficulty_breakdown": dict(difficulty_stats),
        "top_issues": top_issues,
        "failed_items": failed_items[:100]  # Limit to first 100
    }


def generate_markdown_report(analysis: Dict[str, Any], output_path: Path):
    """Generate markdown report from analysis"""

    summary = analysis["summary"]
    scores = analysis["score_averages"]
    domain = analysis["domain_breakdown"]
    difficulty = analysis["difficulty_breakdown"]
    issues = analysis["top_issues"]
    failed = analysis["failed_items"]

    report = f"""# OpenLearnLM Quality Check Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 1. Overall Summary

| Item | Value |
|------|-------|
| **Total Items** | {summary['total']:,} |
| **Passed** | {summary['passed']:,} |
| **Failed** | {summary['failed']:,} |
| **Pass Rate** | {summary['pass_rate']:.1f}% |

---

## 2. Average Score by Evaluation Criteria

| Criterion | Average Score (out of 5) |
|-----------|--------------------------|
"""

    for criterion, avg in sorted(scores.items()):
        criterion_name = criterion.replace("_", " ").title()
        report += f"| {criterion_name} | {avg:.2f} |\n"

    report += f"""
---

## 3. Analysis by Domain

| Domain | Total | Passed | Pass Rate |
|--------|-------|--------|-----------|
"""

    for d, stats in sorted(domain.items()):
        rate = stats['passed'] / stats['total'] * 100 if stats['total'] > 0 else 0
        report += f"| {d} | {stats['total']:,} | {stats['passed']:,} | {rate:.1f}% |\n"

    report += f"""
---

## 4. Analysis by Difficulty

| Difficulty | Total | Passed | Pass Rate |
|------------|-------|--------|-----------|
"""

    for d, stats in sorted(difficulty.items()):
        rate = stats['passed'] / stats['total'] * 100 if stats['total'] > 0 else 0
        report += f"| {d} | {stats['total']:,} | {stats['passed']:,} | {rate:.1f}% |\n"

    report += f"""
---

## 5. Top Issues (Top 20)

| Rank | Issue | Frequency |
|------|-------|-----------|
"""

    for i, (issue, count) in enumerate(issues[:20], 1):
        # Escape markdown special characters
        issue_escaped = issue.replace("|", "\\|").replace("\n", " ")
        report += f"| {i} | {issue_escaped} | {count} |\n"

    if failed:
        report += f"""
---

## 6. Failed Items Sample (Max 20)

"""
        for i, item in enumerate(failed[:20], 1):
            report += f"""### {i}. Item ID: {item['item_id']}

- **Total Score**: {item['total_score']}
- **Domain**: {item['metadata'].get('domain', 'N/A')}
- **Difficulty**: {item['metadata'].get('difficulty', 'N/A')}
- **Scores**: {json.dumps(item['scores'], ensure_ascii=False)}
- **Issues**:
"""
            for issue in item['issues'][:3]:
                report += f"  - {issue}\n"
            report += "\n"

    report += """
---

## 7. Conclusions and Recommendations

### 7.1 Issue #3 Related

Based on the quality check results, Issue #3 (answer accuracy and Affective Domain handling) status may need to be updated.

### 7.2 Follow-up Actions

1. Regenerate or manually fix failed items
2. Focus review on low-scoring criteria (especially answer_accuracy)
3. Improve prompts for recurring issue patterns

---

*Generated by OpenLearnLM Quality Reporter*
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"Report saved to: {output_path}")


def find_latest_results(output_dir: Path) -> Optional[Path]:
    """Find the most recent quality results file"""
    results_files = list(output_dir.glob("quality_results_*.jsonl"))
    if not results_files:
        return None
    return max(results_files, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(
        description="OpenLearnLM Quality Report Generator"
    )
    parser.add_argument(
        "--results-file", type=str, default=None,
        help="Path to quality results JSONL file (default: latest)"
    )

    args = parser.parse_args()

    config = Config()
    output_dir = config.OUTPUT_DIR

    # Find results file
    if args.results_file:
        results_file = Path(args.results_file)
    else:
        results_file = find_latest_results(output_dir)

    if not results_file or not results_file.exists():
        print("Error: No quality results file found.")
        print("Run quality_checker.py first to generate results.")
        return

    print(f"Loading results from: {results_file}")
    results = load_results(results_file)
    print(f"Loaded {len(results):,} results")

    print("Analyzing results...")
    analysis = analyze_results(results)

    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"quality_report_{timestamp}.md"
    generate_markdown_report(analysis, report_path)

    # Print summary to console
    summary = analysis["summary"]
    print("\n" + "=" * 50)
    print("Quality Check Summary")
    print("=" * 50)
    print(f"Total checked: {summary['total']:,}")
    print(f"Passed: {summary['passed']:,} ({summary['pass_rate']:.1f}%)")
    print(f"Failed: {summary['failed']:,}")
    print("=" * 50)

    print("\nAverage by criteria:")
    for criterion, avg in sorted(analysis["score_averages"].items()):
        print(f"  {criterion}: {avg:.2f}")

    print(f"\nDetailed report: {report_path}")


if __name__ == "__main__":
    main()
