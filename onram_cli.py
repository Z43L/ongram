#!/usr/bin/env python3
"""Unified command-line front end for upstream Onram and the local NRA-M tools."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]

COMMANDS = {
    # Upstream Onram utilities.
    "serve": (ROOT / "onram.py", ROOT),
    "infer": (ROOT / "nram_infer.py", ROOT),
    "infer-generic": (ROOT / "nram_infer_generic.py", ROOT),
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

AGENT_ALIASES = {
    "claude": "claude",
    "claude-code": "claude",
    "codex": "codex",
    "pi": "pi",
    "pi-agent": "pi",
}
DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "local-coder:latest"


def launch_help_text() -> str:
    return (
        "Launch a coding agent with its API pointed at the Onram server.\n\n"
        "Usage: ./onram launch {claude|codex|pi} [Onram options] [-- agent options]\n\n"
        "Options:\n"
        "  --base-url URL    Onram URL (default: $ONRAM_BASE_URL or "
        f"{DEFAULT_BASE_URL})\n"
        "  --model ID        Model ID (default: discover it from Onram /v1/models)\n"
        "  --agent-dir PATH  Pi agent config directory (Pi only)\n"
        "  --no-check        Skip the server check; uses $ONRAM_MODEL or "
        f"{DEFAULT_MODEL} when --model is omitted\n\n"
        "Examples:\n"
        "  ./onram launch claude\n"
        "  ./build/onram-cli launch codex --model local-coder:latest\n"
        "  ./onram launch pi -- --print 'Review this repository'\n\n"
        "Start the Onram API first with `./onram serve`. Set ONRAM_API_KEY if the\n"
        "server requires authentication. Pi launch adds or updates the `onram`\n"
        "provider in its models.json while preserving other providers.\n"
    )


def check_onram(base_url: str, api_key: str | None = None, timeout: float = 2.0) -> str:
    """Return the first model ID advertised by a reachable Onram server."""
    base_url = base_url.rstrip("/")
    request = urllib.request.Request(f"{base_url}/v1/models")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise RuntimeError(
                f"Onram at {base_url} requires authentication; set ONRAM_API_KEY."
            ) from exc
        raise RuntimeError(f"Onram returned HTTP {exc.code} at {base_url}.") from exc
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot reach Onram at {base_url}. Start it with `./onram serve` "
            "or pass the correct --base-url."
        ) from exc

    models = payload.get("data", []) if isinstance(payload, dict) else []
    for item in models:
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]:
            return item["id"]
    raise RuntimeError(f"Onram at {base_url} did not advertise a model at /v1/models.")


def _json_toml_string(value: str) -> str:
    """JSON strings are valid basic TOML strings for the values used here."""
    return json.dumps(value, ensure_ascii=False)


def configure_pi_provider(agent_dir: Path, base_url: str, model: str,
                          api_key: str | None) -> Path:
    """Merge the Onram OpenAI-compatible provider into Pi's models.json."""
    agent_dir.mkdir(parents=True, exist_ok=True)
    config_path = agent_dir / "models.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read Pi model config {config_path}: {exc}") from exc
        if not isinstance(config, dict):
            raise RuntimeError(f"Pi model config must contain a JSON object: {config_path}")
    else:
        config = {"$schema": "https://pi.dev/schemas/models.schema.json"}

    providers = config.setdefault("providers", {})
    if not isinstance(providers, dict):
        raise RuntimeError(f"Pi model config `providers` must be an object: {config_path}")
    previous = providers.get("onram", {})
    if not isinstance(previous, dict):
        raise RuntimeError(f"Pi `onram` provider must be an object: {config_path}")
    provider = dict(previous)
    provider.update({
        "baseUrl": f"{base_url.rstrip('/')}/v1",
        "api": "openai-completions",
        "apiKey": "${ONRAM_API_KEY}" if api_key else "local-onram",
        "models": [{"id": model, "name": f"{model} (Onram)"}],
    })
    providers["onram"] = provider

    old_mode = config_path.stat().st_mode & 0o777 if config_path.exists() else 0o600
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=agent_dir,
            prefix=".models.", suffix=".tmp", delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(config, temp_file, indent=2, ensure_ascii=False)
            temp_file.write("\n")
        os.chmod(temp_path, old_mode)
        os.replace(temp_path, config_path)
    except OSError as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Cannot update Pi model config {config_path}: {exc}") from exc
    return config_path


def build_agent_invocation(agent: str, executable: str, forwarded: list[str], *,
                           base_url: str, model: str, env: dict[str, str],
                           agent_dir: Path | None = None) -> tuple[list[str], dict[str, str]]:
    """Build an agent argv/env pair without starting the external program."""
    child_env = dict(env)
    api_key = child_env.get("ONRAM_API_KEY")
    v1_url = f"{base_url.rstrip('/')}/v1"

    if agent == "claude":
        child_env["ANTHROPIC_BASE_URL"] = base_url.rstrip("/")
        child_env["ANTHROPIC_MODEL"] = model
        child_env["ANTHROPIC_AUTH_TOKEN"] = api_key or "local-onram"
        # Never send a pre-existing Claude credential to the Onram endpoint.
        child_env.pop("ANTHROPIC_API_KEY", None)
        return [executable, *forwarded], child_env

    if agent == "codex":
        # A custom Onram provider must not inherit an unrelated cloud key or
        # endpoint from the caller's shell.
        child_env.pop("OPENAI_API_KEY", None)
        child_env.pop("OPENAI_BASE_URL", None)
        overrides = [
            ("model_provider", _json_toml_string("onram")),
            ("model", _json_toml_string(model)),
            ("model_providers.onram.name", _json_toml_string("Onram local")),
            ("model_providers.onram.base_url", _json_toml_string(v1_url)),
            ("model_providers.onram.wire_api", _json_toml_string("responses")),
            ("model_providers.onram.requires_openai_auth", "false"),
            ("model_providers.onram.supports_websockets", "false"),
        ]
        if api_key:
            overrides.append(("model_providers.onram.env_key", _json_toml_string("ONRAM_API_KEY")))
        command = [executable]
        for key, value in overrides:
            command.extend(("--config", f"{key}={value}"))
        return [*command, *forwarded], child_env

    if agent == "pi":
        if agent_dir is None:
            configured_dir = child_env.get("PI_CODING_AGENT_DIR")
            agent_dir = Path(configured_dir).expanduser() if configured_dir else Path.home() / ".pi" / "agent"
        agent_dir = agent_dir.expanduser().resolve()
        configure_pi_provider(agent_dir, base_url, model, api_key)
        child_env["PI_CODING_AGENT_DIR"] = str(agent_dir)
        return [executable, "--provider", "onram", "--model", model, *forwarded], child_env

    raise ValueError(f"Unsupported coding agent: {agent}")


def launch_agent(argv: list[str], environ: dict[str, str] | None = None) -> int:
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(launch_help_text(), end="")
        return 0

    agent = AGENT_ALIASES.get(argv[0].lower())
    if agent is None:
        print(f"onram launch: unsupported agent {argv[0]!r}; choose claude, codex, or pi", file=sys.stderr)
        return 2

    if "--" in argv[1:]:
        separator = argv.index("--", 1)
        option_args = argv[1:separator]
        agent_args = argv[separator + 1:]
    else:
        option_args = argv[1:]
        agent_args = []

    parser = argparse.ArgumentParser(
        prog=f"onram launch {agent}",
        description="Configure and launch a coding agent against Onram.",
    )
    source_env = environ if environ is not None else os.environ
    parser.add_argument("--base-url", default=source_env.get("ONRAM_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=source_env.get("ONRAM_MODEL"))
    parser.add_argument("--agent-dir", type=Path, help="Pi agent config directory (Pi only)")
    parser.add_argument("--no-check", action="store_true", help="skip the Onram /v1/models readiness check")
    try:
        options = parser.parse_args(option_args)
    except SystemExit as exc:
        return int(exc.code)
    if options.agent_dir and agent != "pi":
        parser.error("--agent-dir is only valid with Pi")

    child_env = dict(os.environ if environ is None else environ)
    base_url = options.base_url.rstrip("/")
    model = options.model
    if not options.no_check:
        try:
            discovered_model = check_onram(base_url, child_env.get("ONRAM_API_KEY"))
        except RuntimeError as exc:
            print(f"onram launch: {exc}", file=sys.stderr)
            return 1
        model = model or discovered_model
    else:
        model = model or DEFAULT_MODEL

    executable = shutil.which(agent, path=child_env.get("PATH"))
    if not executable:
        print(f"onram launch: `{agent}` was not found on PATH", file=sys.stderr)
        return 127
    try:
        command, agent_env = build_agent_invocation(
            agent, executable, agent_args, base_url=base_url, model=model,
            env=child_env, agent_dir=options.agent_dir,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"onram launch: {exc}", file=sys.stderr)
        return 1

    print(f"onram launch: starting {agent} with {model} via {base_url}", file=sys.stderr)
    os.execvpe(executable, command, agent_env)
    return 127


def help_text() -> str:
    commands = "\n".join(f"  {name:<21} {description}" for name, description in (
        ("serve", "Run the local Ollama API and coding-agent provider proxy"),
        ("infer", "Generate text with the original DistilGPT2 bundle"),
        ("infer-generic", "Generate text from a generic LoRA bundle (text-only)"),
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
        ("launch", "Launch Claude Code, Codex, or Pi through the Onram API"),
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
        return launch_agent(forwarded)
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
