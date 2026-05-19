"""Regenerate benchmark reports and refresh report_gen data files."""

from __future__ import annotations

import argparse
import contextlib
import io
import re
import runpy
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
RUN_EVALUATION = SCRIPTS_DIR / "evaluation" / "run_evaluation.py"
REPORTS_DIR = Path('/Users/l/klee_code/git_repos/openlearnlm-benchmark-17D4/scripts/report_gen/data')
LATEST_JSON = REPORTS_DIR / "latest.json"
LATEST_MARKDOWN = REPORTS_DIR / "latest.md"
CATEGORY_CHOICES = [
    "01_기능_skills",
    "02_교과지식_content",
    "03_교육학지식_pedagogical",
    "04_태도_attitude",
]


def parse_report_path(output: str, label: str) -> Path:
    pattern = rf"^\s*{re.escape(label)}:\s*(.+?)\s*$"
    for line in output.splitlines():
        match = re.match(pattern, line)
        if match:
            return Path(match.group(1)).expanduser().resolve()
    raise RuntimeError(f"Could not find {label!r} report path in run_evaluation output.")


def run_report_only(category: str | None) -> tuple[int, str]:
    argv = ["run_evaluation.py", "--report-only"]
    if category:
        argv.extend(["--category", category])

    old_argv = sys.argv[:]
    old_path = sys.path[:]
    output = io.StringIO()
    exit_code = 0

    sys.argv = argv
    sys.path.insert(0, str(SCRIPTS_DIR))
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        try:
            runpy.run_path(str(RUN_EVALUATION), run_name="__main__")
        except SystemExit as exc:
            if exc.code is None:
                exit_code = 0
            elif isinstance(exc.code, int):
                exit_code = exc.code
            else:
                print(exc.code)
                exit_code = 1
        finally:
            sys.argv = old_argv
            sys.path[:] = old_path

    return exit_code, output.getvalue()


def refresh_latest(json_report: Path, markdown_report: Path | None) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(json_report, LATEST_JSON)
    if markdown_report and markdown_report.exists():
        shutil.copy2(markdown_report, LATEST_MARKDOWN)


def safe_category_name(category: str) -> str:
    return category.replace("/", "_").replace(" ", "_")


def refresh_category_latest(
    category: str, json_report: Path, markdown_report: Path | None
) -> tuple[Path, Path | None]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"latest_{safe_category_name(category)}"
    category_json = REPORTS_DIR / f"{stem}.json"
    category_markdown = REPORTS_DIR / f"{stem}.md"
    shutil.copy2(json_report, category_json)
    copied_markdown = None
    if markdown_report and markdown_report.exists():
        shutil.copy2(markdown_report, category_markdown)
        copied_markdown = category_markdown
    return category_json, copied_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run evaluation/run_evaluation.py --report-only for benchmark categories "
            "and refresh report_gen/data latest files."
        )
    )
    parser.add_argument(
        "--category",
        choices=CATEGORY_CHOICES,
        help="Only regenerate this benchmark category. Defaults to all categories.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    categories = [args.category] if args.category else CATEGORY_CHOICES
    latest_json_report = None
    latest_markdown_report = None

    for category in categories:
        print("=" * 60)
        print(f"Regenerating report for category: {category}")
        print("=" * 60)
        exit_code, output = run_report_only(category)
        print(output, end="")

        if exit_code != 0:
            return exit_code

        json_report = parse_report_path(output, "JSON")
        markdown_report = parse_report_path(output, "Markdown")
        category_json, category_markdown = refresh_category_latest(
            category, json_report, markdown_report
        )
        latest_json_report = json_report
        latest_markdown_report = markdown_report

        print(f"Category latest JSON refreshed: {category_json}")
        if category_markdown:
            print(f"Category latest Markdown refreshed: {category_markdown}")

    if latest_json_report:
        refresh_latest(latest_json_report, latest_markdown_report)
        print(f"Compatibility latest JSON refreshed: {LATEST_JSON}")
        print(f"Compatibility latest Markdown refreshed: {LATEST_MARKDOWN}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
