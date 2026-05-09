#!/usr/bin/env bash
# Run the OpenLearnLM benchmark against a custom OpenAI-compatible API.
# Override defaults with environment variables:
#   CUSTOM_API_URL   - base URL of the API  (default: innospark endpoint)
#   CUSTOM_API_KEY   - bearer token          (default: key from test function)
#   CUSTOM_MODEL     - model name to test    (default: gemini-2.0-flash)
#   JUDGE_MODEL      - model used for LLM-as-Judge scoring (default: gemini-2.0-flash)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- defaults (matching the test() function in the original stub) ---
CUSTOM_API_URL="${CUSTOM_API_URL:-https://api.innospark.cn/v1}"
CUSTOM_API_KEY="${CUSTOM_API_KEY:-sk-yJdHSUrrZNYkBS5f5dHOPgoxRw5Q8qRJPFTbKh6jOqnAUZNF}"
CUSTOM_MODEL="${CUSTOM_MODEL:-gemini-2.0-flash}"

# Judge model defaults to the same model as the test model
JUDGE_MODEL="${JUDGE_MODEL:-$CUSTOM_MODEL}"

# --- expose as env vars that config.py reads ---
export OPENAI_BASE_URL="$CUSTOM_API_URL"
export OPENAI_API_KEY="$CUSTOM_API_KEY"
export OPENROUTER_BASE_URL="$CUSTOM_API_URL"
export OPENROUTER_API_KEY="$CUSTOM_API_KEY"
export JUDGE_MODEL

echo "API URL : $CUSTOM_API_URL"
echo "Model   : $CUSTOM_MODEL"
cd /Users/l/klee_code/git_repos/openlearnlm-benchmark-17D4/scripts/
exec python "$SCRIPT_DIR/scripts/evaluation/run_evaluation.py" \
  --models "$CUSTOM_MODEL" \
  "$@"
