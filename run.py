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

@dataclass(frozen=True)
class ApiConfig:
    base_url: str
    api_key: str
    model: str

    @classmethod
    def from_env(
        cls,
        *,
        defaults: "ApiConfig",
        url_env: str,
        key_env: str,
        model_env: str,
    ) -> "ApiConfig":
        return cls(
            base_url=os.environ.get(url_env, defaults.base_url),
            api_key=os.environ.get(key_env, defaults.api_key),
            model=os.environ.get(model_env, defaults.model),
        )


DEFAULT_EVAL_API = ApiConfig(
    base_url="https://e8p5ocom8hcgcecckoc8jbhhohqhahhg.openapi-qb.sii.edu.cn/v1",
    api_key="QF8XqjmZhi9sygq1MT9Nr0rk8xZzVlid/aLvCRurzWw=",
    model="Qwen3-4B-Instruct-2507",
)


DEFAULT_BASELINE_API = ApiConfig(
    #     https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-667f46c5-314e-46ac-87e8-9ee38d91fab3?spaceId=ws-33f55cbb-1e6b-4b37-b69d-3b52568e0a61
    base_url="https://eeg5ceodb9cqcekohgqhjqqbhpj95kmb.openapi-qb.sii.edu.cn/v1",
    api_key="QF8XqjmZhi9sygq1MT9Nr0rk8xZzVlid/aLvCRurzWw=",
    model="Qwen3-4B-Instruct-2507-Official",
)

GEMINI_API = ApiConfig(
    base_url="https://api.innospark.cn/v1",
    api_key="sk-yJdHSUrrZNYkBS5f5dHOPgoxRw5Q8qRJPFTbKh6jOqnAUZNF",
    model="gemini-2.5-flash",
)
DEFAULT_JUDGE_API = GEMINI_API
# ApiConfig(
#     base_url="https://ea5maamppajpchmpmqk5obbpemep5p5k.openapi-qb.sii.edu.cn/v1",
#     api_key="QF8XqjmZhi9sygq1MT9Nr0rk8xZzVlid/aLvCRurzWw=",
#     model="Qwen3.6-27B",
# )


EVAL_API = DEFAULT_EVAL_API
# ApiConfig.from_env(
#     defaults=DEFAULT_EVAL_API,
#     url_env="CUSTOM_API_URL",
#     key_env="CUSTOM_API_KEY",
#     model_env="CUSTOM_MODEL",
# )
JUDGE_API = DEFAULT_JUDGE_API

os.environ["OPENAI_BASE_URL"]     = EVAL_API.base_url
os.environ["OPENAI_API_KEY"]      = EVAL_API.api_key
os.environ["OPENROUTER_BASE_URL"] = EVAL_API.base_url
os.environ["OPENROUTER_API_KEY"]  = EVAL_API.api_key
os.environ["JUDGE_API_URL"]       = JUDGE_API.base_url
os.environ["JUDGE_API_KEY"]       = JUDGE_API.api_key
os.environ["JUDGE_MODEL"]         = JUDGE_API.model

print(f"API URL : {EVAL_API.base_url}")
print(f"Model   : {EVAL_API.model}")
print(f"Judge API URL : {JUDGE_API.base_url}")
print(f"Judge Model   : {JUDGE_API.model}")

os.chdir(SCRIPT_DIR / "scripts")
sys.path.insert(0, str(SCRIPT_DIR / "scripts"))
#
sys.argv = ["run_evaluation.py", "--models", EVAL_API.model, "--workers", str(2), '--limit', '12'] + ["--no-resume"] + sys.argv[1:]

from evaluation.engine.orchestrator import EvaluationOrchestrator  # noqa: F401 — trigger path setup
import runpy
runpy.run_path(str(SCRIPT_DIR / "scripts" / "evaluation" / "run_evaluation.py"), run_name="__main__")
