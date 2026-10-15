#!/usr/bin/env bash
set -euo pipefail

export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-http://127.0.0.1:11534}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-local-onram}"
export ANTHROPIC_MODEL="${ANTHROPIC_MODEL:-llama3.2-1b-instruct:latest}"

exec claude "$@"
