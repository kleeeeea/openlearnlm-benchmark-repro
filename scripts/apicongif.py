#!/usr/bin/env python3
"""Generate a sample curl command from the API config in run.py."""

from __future__ import annotations

import ast
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path



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


DEFAULT_INNOSPARK_API = ApiConfig(
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
Qwen27b_API = ApiConfig(
    base_url="https://ea5maamppajpchmpmqk5obbpemep5p5k.openapi-qb.sii.edu.cn/v1",
    api_key="QF8XqjmZhi9sygq1MT9Nr0rk8xZzVlid/aLvCRurzWw=",
    model="Qwen3.6-27B",
)

GLM51FP8_API = ApiConfig(
    # https://qz.sii.edu.cn/jobs/modelDeplayDetail/sv-d46fd98b-a669-4acb-92b6-85966acbf383?spaceId=ws-33f55cbb-1e6b-4b37-b69d-3b52568e0a61
    base_url="https://pce5pjhmkeejckomjamehdjeekom5bhb.openapi-qb.sii.edu.cn/v1",
    api_key="QF8XqjmZhi9sygq1MT9Nr0rk8xZzVlid/aLvCRurzWw=",
    model="GLM-5.1-FP8",
)

SCRIPT_DIR = Path(__file__).parent.resolve()
RUN_PY_CANDIDATES = [
    SCRIPT_DIR / "run.py",
    SCRIPT_DIR / "scripts" / "run.py",
]
OUTPUT_SH = SCRIPT_DIR / "apicongif.sh"
CONFIG_NAME = os.environ.get("API_CONFIG_NAME", "DEFAULT_JUDGE_API")



def literal_keyword(call: ast.Call, name: str) -> str:
    for keyword in call.keywords:
        if keyword.arg == name:
            value = ast.literal_eval(keyword.value)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string literal")
            return value
    raise KeyError(f"Missing ApiConfig keyword: {name}")


def parse_api_configs(path: Path) -> dict[str, ApiConfig]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    configs: dict[str, ApiConfig] = {}
    aliases: dict[str, str] = {}

    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue

        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "ApiConfig"
        ):
            configs[target.id] = ApiConfig(
                base_url=literal_keyword(node.value, "base_url"),
                api_key=literal_keyword(node.value, "api_key"),
                model=literal_keyword(node.value, "model"),
            )
        elif isinstance(node.value, ast.Name):
            aliases[target.id] = node.value.id

    for alias, source in aliases.items():
        if source in configs:
            configs[alias] = configs[source]

    return configs


def render_curl(config: ApiConfig) -> str:
    payload = {
        "model": config.model,
        "messages": [
            {"role": "user", "content": "hi"},
        ],
    }
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    return f"""#!/usr/bin/env bash

curl -sS --fail -X POST "{config.base_url.rstrip('/')}/chat/completions" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer {config.api_key}" \\
  -d '{body}'
"""




def main() -> int:
    # run_py = next((path for path in RUN_PY_CANDIDATES if path.exists()), None)
    # if run_py is None:
    #     candidates = ", ".join(str(path) for path in RUN_PY_CANDIDATES)
    #     raise SystemExit(f"Could not find run.py. Tried: {candidates}")

    # configs = parse_api_configs(run_py)
    # if CONFIG_NAME not in configs:
    #     names = ", ".join(sorted(configs))
    #     raise SystemExit(f"Unknown API config {CONFIG_NAME!r}. Available: {names}")

    content = render_curl(DEFAULT_BASELINE_API)
    OUTPUT_SH.write_text(content, encoding="utf-8")
    OUTPUT_SH.chmod(OUTPUT_SH.stat().st_mode | stat.S_IXUSR)
    print(f"Wrote {OUTPUT_SH}")
    print(f"Config: {CONFIG_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
