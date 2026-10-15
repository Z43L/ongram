#!/usr/bin/env python3
"""Unified command-line front end for upstream Onram and the local NRA-M tools."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
LOCAL_BASE_URL = "http://127.0.0.1:11534"
LOCAL_MODEL = "llama3.2-1b-instruct:latest"


def _runtime_python() -> str:
    candidates = [
        os.environ.get("ONRAM_PYTHON"),
        str(ROOT / ".venv" / "bin" / "python"),
        "/Users/david/Desktop/gemma4_pentesting_edge/.venv/bin/python",
        str(ROOT / "../../.venv-gemma4" / "bin" / "python"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    return sys.executable

COMMANDS = {
    # Upstream Onram utilities.
    "serve": (ROOT / "onram.py", ROOT),
    "infer": (ROOT / "nram_infer.py", ROOT),
    "infer-generic": (ROOT / "nram_infer_generic.py", ROOT),
    "nram-debug": (ROOT / "nram_debug.py", ROOT),
    "convert": (ROOT / "nram_convert.py", ROOT),
    "evolve": (ROOT / "nram_evolve_generic.py", ROOT),
    "evolve-distilgpt2": (ROOT / "nram_evolve.py", ROOT),
    "merge": (ROOT / "nram_merge_generic.py", ROOT),
    "merge-distilgpt2": (ROOT / "nram_merge.py", ROOT),
    "export": (ROOT / "nram_export_generic.py", ROOT),
    "export-distilgpt2": (ROOT / "export_evolved.sh", ROOT),
    "export-gguf-tools": (ROOT / "export_gguf.sh", ROOT),
    "verify": (ROOT / "verify_generic.py", ROOT),
    "verify-local": (ROOT / "verify_local.py", ROOT),
    "verify-evolved": (ROOT / "verify_evolved.py", ROOT),
    "verify-generic": (ROOT / "verify_generic.py", ROOT),
    "download-model": (ROOT / "download_model.py", ROOT),
    "runtime-setup": (ROOT / "setup_onram.sh", ROOT),
    "demo": (ROOT / "demo.py", ROOT),
    "build": (ROOT / "build_onram.sh", ROOT),
    # The existing Gemma NRA-M v1 package and its tools.
    "nram": (PROJECT_ROOT / "nra_m" / "nram.py", PROJECT_ROOT),
    "nram-eval": (PROJECT_ROOT / "nra_m" / "run_eval.py", PROJECT_ROOT),
    "nram-sweep": (PROJECT_ROOT / "nra_m" / "evolve_routes.py", PROJECT_ROOT),
    "nram-memory": (PROJECT_ROOT / "nra_m" / "memory.py", PROJECT_ROOT),
    "nram-adapt": (PROJECT_ROOT / "nra_m" / "weight_mutation.py", PROJECT_ROOT),
    "nram-prepare-data": (PROJECT_ROOT / "nra_m" / "to_lora_jsonl.py", PROJECT_ROOT),
}


def help_text() -> str:
    commands = "\n".join(f"  {name:<21} {description}" for name, description in (
        ("serve", "Run the local Ollama API and coding-agent provider proxy"),
        ("launch", "Start Onram if needed, then launch claude, codex, or pi against Llama 3.2 1B"),
        ("infer", "Generate text with the original DistilGPT2 bundle"),
        ("infer-generic", "Generate text from a generic LoRA bundle (text-only)"),
        ("nram-debug", "Run a traced Transformers generation and report layer/adapter signals"),
        ("convert", "Create a generic Hugging Face LoRA bundle"),
        ("evolve", "Train/evaluate a generic LoRA candidate"),
        ("evolve-distilgpt2", "Run the original DistilGPT2 evolution experiment"),
        ("merge", "Merge a generic LoRA bundle into its base checkpoint"),
        ("merge-distilgpt2", "Merge the original DistilGPT2 adapter"),
        ("export", "Export a generic bundle as F16/Q4 GGUF"),
        ("export-distilgpt2", "Export the original DistilGPT2 adapter"),
        ("export-gguf-tools", "Run the upstream GGUF conversion helper"),
        ("verify", "Verify generic adapter and merged-checkpoint logits"),
        ("verify-generic", "Alias for generic bundle verification"),
        ("verify-local", "Run upstream local Transformers/GGUF smoke checks"),
        ("verify-evolved", "Verify the saved DistilGPT2 evolution result"),
        ("download-model", "Download the pinned DistilGPT2 checkpoint"),
        ("runtime-setup", "Build the llama.cpp server and export utilities"),
        ("build", "Build the native Onram command launcher"),
        ("nram", "Create, inspect, promote, or roll back a local .nram v1"),
        ("nram-eval", "Evaluate a local Gemma route"),
        ("nram-sweep", "Rank candidate Gemma layer routes"),
        ("nram-memory", "Encode or commit associative-memory keys"),
        ("nram-adapt", "Propose a train-only selective output delta"),
        ("nram-prepare-data", "Convert this workspace's YES/NO data to prompt/completion JSONL"),
        ("test", "Run upstream and workspace unit suites"),
    ))
    return (
        "Onram unified CLI (adapted for the NRA research workspace)\n\n"
        "Usage: ./onram <command> [command options]\n"
        "Legacy compatibility: ./onram [server options] starts `serve`.\n\n"
        "Commands:\n" + commands + "\n\n"
        "Set ONRAM_PYTHON to select a Python environment. The native launcher "
        "still requires Python packages, model files, and (for serving) llama-server.\n"
    )


def resolve(argv: list[str]) -> tuple[Path | None, Path, list[str], str | None]:
    """Resolve a command; unknown leading options remain legacy server arguments."""
    if argv and argv[0] in {"-h", "--help", "help"}:
        return None, ROOT, [], "help"
    if argv and argv[0] in {"--version", "version"}:
        return None, ROOT, [], "version"
    if argv and argv[0] == "test":
        return None, ROOT, argv[1:], "test"
    if argv and argv[0] == "launch":
        return None, ROOT, argv[1:], "launch"
    if not argv:
        command, forwarded = "serve", []
    elif argv[0] in COMMANDS:
        command, forwarded = argv[0], argv[1:]
    else:
        command, forwarded = "serve", argv
    script, cwd = COMMANDS[command]
    return script, cwd, forwarded, None


def _local_model_is_ready() -> bool:
    request = urllib.request.Request(
        f"{LOCAL_BASE_URL}/v1/models",
        headers={"Authorization": f"Bearer {os.environ.get('ONRAM_API_KEY', 'local-onram')}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            body = json.loads(response.read().decode("utf-8"))
        return any(item.get("id") == LOCAL_MODEL for item in body.get("data", []))
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, AttributeError):
        return False


def _ensure_local_server() -> None:
    if _local_model_is_ready():
        return

    model = ROOT / "models" / "llama3.2-1b-instruct-current.gguf"
    bundle = ROOT / "models" / "llama3.2-1b-instruct.nram"
    if not model.is_file() or not bundle.is_file():
        raise RuntimeError(
            "Llama 3.2 1B files were not found. `--model` must point to a .gguf file "
            "and `--bundle` to the .nram adapter file. Expected:\n"
            f"  --model {model}\n  --bundle {bundle}"
        )

    log_path = ROOT / "onram-server.log"
    log = log_path.open("a", encoding="utf-8")
    command = [
        _runtime_python(), str(ROOT / "onram.py"),
        "--model", str(model), "--bundle", str(bundle),
        "--model-name", "llama3.2-1b-instruct",
        "--port", "11534", "--backend-port", "11538", "--auto-learn", "--inference-debug",
        "--learn-batch-size", "24",
    ]
    try:
        subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    finally:
        log.close()

    deadline = time.monotonic() + 125
    while time.monotonic() < deadline:
        if _local_model_is_ready():
            print(f"Onram listo en {LOCAL_BASE_URL}; aprendizaje automático activado.", flush=True)
            return
        time.sleep(0.5)
    try:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
    except OSError:
        tail = ""
    raise RuntimeError(f"Onram no arrancó; revisa {log_path}.\n{tail}")


def launch_client(arguments: list[str]) -> int:
    if not arguments or arguments[0] in {"-h", "--help", "help"}:
        print("Usage: ./onram-cli launch <claude|codex|pi> [client options]\n"
              "Starts Onram with Llama 3.2 1B Instruct and automatic learning if needed.")
        return 0
    if arguments[0] not in {"claude", "claude-code", "codex", "pi"}:
        print("Usage: ./onram-cli launch <claude|codex|pi> [client options]", file=sys.stderr)
        return 2
    name, *client_args = arguments
    client = {"claude-code": "claude"}.get(name, name)
    executable = shutil.which(client)
    if not executable:
        print(f"onram: no encuentro `{client}` en PATH", file=sys.stderr)
        return 127
    if client_args[:1] == ["--"]:
        client_args = client_args[1:]

    try:
        _ensure_local_server()
    except (OSError, RuntimeError) as exc:
        print(f"onram: {exc}", file=sys.stderr)
        return 1

    env = os.environ.copy()
    if client == "claude":
        env["ANTHROPIC_BASE_URL"] = LOCAL_BASE_URL
        env.setdefault("ANTHROPIC_API_KEY", os.environ.get("ONRAM_API_KEY", "local-onram"))
        env["ANTHROPIC_MODEL"] = LOCAL_MODEL
        command = [executable, *client_args]
    elif client == "codex":
        command = [executable, "--profile", "onram", *client_args]
    else:
        command = [executable, "--provider", "onram", "--model", LOCAL_MODEL, *client_args]
    os.execvpe(executable, command, env)
    return 127


def run_tests() -> int:
    suites = (
        (ROOT, [sys.executable, "-m", "unittest", "discover", "-s", ".", "-v"]),
        (PROJECT_ROOT, [sys.executable, "-m", "unittest", "discover", "-s", "nra_m/tests", "-v"]),
    )
    for cwd, command in suites:
        result = subprocess.run(command, cwd=cwd, check=False)
        if result.returncode:
            return result.returncode
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    script, cwd, forwarded, action = resolve(args)
    if action == "help":
        print(help_text(), end="")
        return 0
    if action == "version":
        print("Onram unified CLI 0.1 (upstream base 8c368ce)")
        return 0
    if action == "test":
        return run_tests()
    if action == "launch":
        return launch_client(forwarded)
    assert script is not None
    if not script.is_file():
        print(f"onram: utility is missing: {script}", file=sys.stderr)
        return 2
    if script.suffix == ".sh":
        return subprocess.run([str(script), *forwarded], cwd=cwd, check=False).returncode
    os.chdir(cwd)
    os.execv(sys.executable, [sys.executable, str(script), *forwarded])
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
