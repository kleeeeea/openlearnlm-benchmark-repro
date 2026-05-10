#!/usr/bin/env python3
"""
Python driver for the OpenLearnLM benchmark.

Mirrors run.sh — override defaults with environment variables:
  CUSTOM_API_URL   base URL of the API        (default: innospark endpoint)
  CUSTOM_API_KEY   bearer token               (default: bundled test key)
  CUSTOM_MODEL     model name to test         (default: gemini-2.0-flash)
  JUDGE_MODEL      model used for LLM scoring (default: same as CUSTOM_MODEL)

Any extra arguments are forwarded to run_evaluation.py.
"""

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

CUSTOM_API_URL = os.environ.get("CUSTOM_API_URL", "https://api.innospark.cn/v1")
CUSTOM_API_KEY = os.environ.get("CUSTOM_API_KEY", "sk-yJdHSUrrZNYkBS5f5dHOPgoxRw5Q8qRJPFTbKh6jOqnAUZNF")
CUSTOM_MODEL   = os.environ.get("CUSTOM_MODEL",   "gemini-2.0-flash")
JUDGE_MODEL    = os.environ.get("JUDGE_MODEL",    CUSTOM_MODEL)

os.environ["OPENAI_BASE_URL"]     = CUSTOM_API_URL
os.environ["OPENAI_API_KEY"]      = CUSTOM_API_KEY
os.environ["OPENROUTER_BASE_URL"] = CUSTOM_API_URL
os.environ["OPENROUTER_API_KEY"]  = CUSTOM_API_KEY
os.environ["JUDGE_MODEL"]         = JUDGE_MODEL

print(f"API URL : {CUSTOM_API_URL}")
print(f"Model   : {CUSTOM_MODEL}")

os.chdir(SCRIPT_DIR / "scripts")
sys.path.insert(0, str(SCRIPT_DIR / "scripts"))

sys.argv = ["run_evaluation.py", "--models", CUSTOM_MODEL, '--no-resume', '--workers', '1', '--report-only'] + sys.argv[1:]

from evaluation.engine.orchestrator import EvaluationOrchestrator  # noqa: F401 — trigger path setup
import runpy
runpy.run_path(str(SCRIPT_DIR / "scripts" / "evaluation" / "run_evaluation.py"), run_name="__main__")
