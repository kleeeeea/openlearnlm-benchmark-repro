#!/usr/bin/env python3
"""
Python driver for the OpenLearnLM benchmark.

Mirrors run.sh — override defaults with environment variables:
  CUSTOM_API_URL   base URL of the API        (default: production endpoint)
  CUSTOM_API_KEY   bearer token               (default: production test key)
  CUSTOM_MODEL     model name to test         (default: Qwen3-4B-Instruct-2507)
  JUDGE_API_URL    base URL used for LLM scoring
  JUDGE_API_KEY    bearer token used for LLM scoring
  JUDGE_MODEL      model used for LLM scoring

Any extra arguments are forwarded to run_evaluation.py.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from apicongif import DEFAULT_BASELINE_API
from apicongif import DEFAULT_JUDGE_API
from apicongif import GLM51FP8_API
from evaluation.run_evaluation import CATEGORY_CHOICES

SCRIPT_DIR = Path(__file__).parent.resolve()

SAMPLE_PROD_CURL = '''
curl -sS --fail -X POST "https://e8p5ocom8hcgcecckoc8jbhhohqhahhg.openapi-qb.sii.edu.cn/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer QF8XqjmZhi9sygq1MT9Nr0rk8xZzVlid/aLvCRurzWw=" \
    -d '{
      "model": "Qwen3-4B-Instruct-2507",
      "messages": [
        { "role": "user", "content": "hi" }
      ]
    }'
	 '''
def main():


    from apicongif import DEFAULT_INNOSPARK_API
    for MAIN_MODEL_API in  [
            # DEFAULT_INNOSPARK_API,
            DEFAULT_BASELINE_API
    ]:
        # ApiConfig.from_env(
        #     defaults=DEFAULT_EVAL_API,
        #     url_env="CUSTOM_API_URL",
        #     key_env="CUSTOM_API_KEY",
        #     model_env="CUSTOM_MODEL",
        # )
        JUDGE_API = GLM51FP8_API

        os.environ["OPENAI_BASE_URL"]     = MAIN_MODEL_API.base_url
        os.environ["OPENAI_API_KEY"]      = MAIN_MODEL_API.api_key
        os.environ["OPENROUTER_BASE_URL"] = MAIN_MODEL_API.base_url
        os.environ["OPENROUTER_API_KEY"]  = MAIN_MODEL_API.api_key
        os.environ["JUDGE_API_URL"]       = JUDGE_API.base_url
        os.environ["JUDGE_API_KEY"]       = JUDGE_API.api_key
        os.environ["JUDGE_MODEL"]         = JUDGE_API.model

        print(f"API URL : {MAIN_MODEL_API.base_url}")
        print(f"Model   : {MAIN_MODEL_API.model}")
        print(f"Judge API URL : {JUDGE_API.base_url}")
        print(f"Judge Model   : {JUDGE_API.model}")

        os.chdir(SCRIPT_DIR )
        sys.path.insert(0, str(SCRIPT_DIR ))
        limit = 5
        workers = 1 if limit is not None and limit < 20 else 2
        extra_args = sys.argv[1:]
        import runpy

        for cat in CATEGORY_CHOICES[:]:
            print("=" * 60)
            print(f"Running category: {cat}")
            print("=" * 60)

            sys.argv = [
                "run_evaluation.py",
                "--models",
                MAIN_MODEL_API.model,
                "--workers",
                str(workers),
                "--limit",
                str(limit),
            ] + extra_args + [
                "--category",
                cat,
                # "--no-resume",
            ]


            from evaluation.engine.orchestrator import EvaluationOrchestrator  # noqa: F401 — trigger path setup
            try:
                runpy.run_path(str(SCRIPT_DIR  / "evaluation" / "run_evaluation.py"), run_name="__main__")
            except SystemExit as exc:
                if exc.code in (0, None):
                    continue
                return exc.code if isinstance(exc.code, int) else 1

    return 0
if __name__ == '__main__':
    sys.exit(main())
