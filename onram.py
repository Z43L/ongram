#!/usr/bin/env python3
"""Onram: an Ollama-shaped local API with supervised NRA-M evolution.

This is an experimental compatibility layer around llama.cpp and the existing
NRA-M training/export pipeline. It is not a fork or feature-complete clone of
Ollama.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
MODEL_POINTER = ROOT / "models" / "onram-current.gguf"
DEFAULT_MODEL = MODEL_POINTER
DEFAULT_BUNDLE = ROOT / "models" / "distilgpt2-nra-qa.nram"
def _default_llama_server() -> Path:
    candidates = []
    if os.environ.get("ONRAM_LLAMA_SERVER"):
        candidates.append(Path(os.environ["ONRAM_LLAMA_SERVER"]))
    candidates.extend((ROOT / "build" / "bin" / "llama-server",
                       ROOT.parent.parent / "deps" / "llama.cpp" / "build" / "bin" / "llama-server"))
    discovered = shutil.which("llama-server")
    if discovered:
        candidates.append(Path(discovered))
    return next((path for path in candidates if path.is_file()), candidates[0])


DEFAULT_LLAMA_SERVER = _default_llama_server()
ALLOWED_DATASETS = {
    "nra_qa_train.jsonl",
    "nra_qa_validation.jsonl",
    "general_guardrail.jsonl",
}


class Onram:
    def __init__(self, model_path: Path, llama_server: Path, host: str, port: int,
                 backend_port: int, bundle: Path, api_key: str | None = None,
                 model_name: str | None = None, ctx_size: int = 0):
        self._stable_model_alias = model_name or ("distilgpt2-nram:latest" if model_path.name == MODEL_POINTER.name else None)
        self.model_path = model_path.resolve()
        self.llama_server = llama_server.resolve()
        self.host = host
        self.port = port
        self.backend_port = backend_port
        self.bundle = bundle.resolve()
        self.ctx_size = ctx_size
        self.model_pointer = MODEL_POINTER if model_path.name == MODEL_POINTER.name else self.bundle.with_name(self.bundle.stem + "-current.gguf")
        self.api_key = api_key
        self.backend: subprocess.Popen[str] | None = None
        self.backend_log = (ROOT / "onram-backend.log").open("a", encoding="utf-8")
        self.lock = threading.RLock()
        self.evolution_lock = threading.Lock()
        self.evolution: dict[str, Any] = {"state": "idle"}
        self.started_at = datetime.now(timezone.utc).isoformat()

    @property
    def backend_url(self) -> str:
        return f"http://127.0.0.1:{self.backend_port}"

    @property
    def model_name(self) -> str:
        stable_alias = getattr(self, "_stable_model_alias", None)
        if stable_alias:
            return stable_alias
        name = re.sub(r"-Q[^-]+$", "", self.model_path.stem)
        return f"{name}:latest"

    def start_backend(self) -> None:
        if not self.model_path.is_file():
            raise FileNotFoundError(f"GGUF not found: {self.model_path}")
        if not self.llama_server.is_file():
            raise FileNotFoundError(
                f"llama-server not found: {self.llama_server}. Build it with `./onram runtime-setup` "
                "or pass --llama-server /path/to/llama-server."
            )
        self.backend = subprocess.Popen(
            [str(self.llama_server), "--model", str(self.model_path), "--host", "127.0.0.1",
             "--port", str(self.backend_port), "--ctx-size", str(self.ctx_size), "--parallel", "1",
             "--alias", self.model_name, "--jinja"],
            cwd=ROOT,
            stdout=self.backend_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if self.backend.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self.backend.returncode}; see onram-backend.log")
            try:
                self._backend_request("GET", "/health", None, timeout=1)
                return
            except (OSError, urllib.error.URLError, RuntimeError):
                time.sleep(0.25)
        self.stop_backend()
        raise TimeoutError("Timed out waiting for llama-server health")

    def stop_backend(self) -> None:
        process = self.backend
        self.backend = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def close(self) -> None:
        self.stop_backend()
        self.backend_log.close()

    def _backend_request(self, method: str, path: str, body: dict[str, Any] | None,
                         timeout: float = 120) -> dict[str, Any]:
        raw = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(self.backend_url + path, data=raw, method=method,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read()) if response.status != 204 else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"llama-server returned HTTP {exc.code}: {detail}") from exc

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        params = self._completion_options(prompt, options)
        params["stream"] = False
        with self.lock:
            return self._backend_request("POST", "/completion", params)

    @staticmethod
    def _completion_options(prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        num_predict = int(options.get("num_predict", 128))
        temperature = float(options.get("temperature", 0.8))
        if not 1 <= num_predict <= 512:
            raise ValueError("num_predict must be between 1 and 512 for this prototype")
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        params: dict[str, Any] = {
            "prompt": prompt,
            "n_predict": num_predict,
            "temperature": temperature,
            "top_k": int(options.get("top_k", 40)),
            "top_p": float(options.get("top_p", 0.95)),
            "repeat_penalty": float(options.get("repeat_penalty", 1.1)),
            "stream": False,
        }
        if options.get("stop"):
            params["stop"] = options["stop"]
        return params

    def stream_generate(self, prompt: str, options: dict[str, Any]):
        params = self._completion_options(prompt, options)
        params["stream"] = True
        raw = json.dumps(params).encode()
        request = urllib.request.Request(self.backend_url + "/completion", data=raw, method="POST",
                                         headers={"Content-Type": "application/json"})
        with self.lock:
            try:
                response = urllib.request.urlopen(request, timeout=3600)
                for raw_line in response:
                    line = raw_line.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    item = json.loads(payload)
                    yield item
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")
                raise RuntimeError(f"llama-server returned HTTP {exc.code}: {detail}") from exc

    def begin_evolution(self, body: dict[str, Any]) -> dict[str, Any]:
        if not self.evolution_lock.acquire(blocking=False):
            raise RuntimeError("An evolution run is already active")
        try:
            train = self._dataset(body.get("train", "nra_qa_train.jsonl"))
            validation = self._dataset(body.get("validation", "nra_qa_validation.jsonl"))
            guardrail = self._dataset(body.get("guardrail", "general_guardrail.jsonl"))
            epochs = int(body.get("epochs", 2))
            if not 1 <= epochs <= 100:
                raise ValueError("epochs must be between 1 and 100")
            learning_rate = float(body.get("learning_rate", 0.0005))
            if not 0 < learning_rate <= 0.01:
                raise ValueError("learning_rate must be > 0 and <= 0.01")
            self.evolution = {"state": "running", "started_at": datetime.now(timezone.utc).isoformat(),
                              "stage": "training", "accepted": None}
            threading.Thread(target=self._evolve_worker,
                             args=(train, validation, guardrail, epochs, learning_rate), daemon=True).start()
            return dict(self.evolution)
        except BaseException:
            self.evolution_lock.release()
            raise

    @staticmethod
    def _dataset(name: str) -> Path:
        if name not in ALLOWED_DATASETS:
            raise ValueError(f"dataset must be one of: {', '.join(sorted(ALLOWED_DATASETS))}")
        path = (ROOT / "data" / name).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"dataset not found: {path}")
        return path

    def _evolve_worker(self, train: Path, validation: Path, guardrail: Path,
                       epochs: int, learning_rate: float) -> None:
        try:
            python = str(ROOT / ".venv" / "bin" / "python")
            from nram_container import load_bundle
            manifest, _ = load_bundle(self.bundle)
            generic = manifest.get("model_family") == "hf-causal-lm-dense-lora"
            if generic:
                command = [python, str(ROOT / "nram_evolve_generic.py"),
                           "--bundle", str(self.bundle), "--train", str(train),
                           "--validation", str(validation), "--guardrail", str(guardrail),
                           "--epochs", str(epochs), "--learning-rate", str(learning_rate),
                           "--local-files-only"]
            else:
                command = [python, str(ROOT / "nram_evolve.py"),
                           "--train", str(train), "--validation", str(validation),
                           "--guardrail", str(guardrail), "--output", str(self.bundle),
                           "--epochs", str(epochs), "--learning-rate", str(learning_rate)]
            proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=6 * 60 * 60)
            output = proc.stdout + "\n" + proc.stderr
            if proc.returncode:
                raise RuntimeError(output[-6000:] or f"evolver exited {proc.returncode}")
            match = re.search(r'"accepted"\s*:\s*(true|false)', output)
            if match is None:
                raise RuntimeError("evolver completed without a validation report: " + output[-3000:])
            accepted = match.group(1) == "true"
            report = {"state": "completed", "accepted": accepted, "stage": "validated",
                      "finished_at": datetime.now(timezone.utc).isoformat(), "log_tail": output[-3000:]}
            if accepted:
                report["stage"] = "exporting"
                version = self._bundle_version()
                if generic:
                    name = f"onram-{self.bundle.stem}-v{version}"
                    export = subprocess.run([python, str(ROOT / "nram_export_generic.py"),
                                             "--bundle", str(self.bundle), "--output-prefix",
                                             str(ROOT / "models" / name)], cwd=ROOT, text=True,
                                            capture_output=True, timeout=6 * 60 * 60)
                    q4_path = ROOT / "models" / f"{name}-Q4_K_M.gguf"
                    export_text = export.stdout
                else:
                    name = f"onram-nram-v{version}"
                    export = subprocess.run([str(ROOT / "export_evolved.sh"), str(self.bundle), name],
                                            cwd=ROOT, text=True, capture_output=True, timeout=6 * 60 * 60)
                    q4_path = ROOT / "models" / f"{name}-Q4_K_M.gguf"
                    export_text = export.stdout
                if export.returncode:
                    raise RuntimeError((export.stdout + "\n" + export.stderr)[-6000:])
                report["stage"] = "reloading"
                self._swap_model(q4_path)
                report["model_path"] = str(q4_path)
                report["export_log"] = export_text[-2000:]
                report["stage"] = "ready"
            self.evolution = report
        except BaseException as exc:
            self.evolution = {"state": "failed", "stage": "error", "error": str(exc),
                              "finished_at": datetime.now(timezone.utc).isoformat()}
        finally:
            self.evolution_lock.release()

    def _bundle_version(self) -> int:
        from nram_container import load_bundle
        manifest, _ = load_bundle(self.bundle)
        return int(manifest["version"])

    def _swap_model(self, candidate: Path) -> None:
        old = self.model_path
        pointer = self.model_pointer
        with self.lock:
            self.stop_backend()
            try:
                self.model_path = candidate.resolve()
                self.start_backend()
                temp_pointer = pointer.with_name(f".{pointer.name}.{os.getpid()}.tmp")
                temp_pointer.unlink(missing_ok=True)
                temp_pointer.symlink_to(self.model_path.name)
                os.replace(temp_pointer, pointer)
            except BaseException:
                self.model_path = old
                self.start_backend()
                raise

    def status(self) -> dict[str, Any]:
        return {"name": "onram", "version": "0.1.0", "model": self.model_name,
                "model_path": str(self.model_path), "backend": "llama.cpp",
                "backend_running": self.backend is not None and self.backend.poll() is None,
                "evolution": dict(self.evolution), "started_at": self.started_at}


def handler_for(app: Onram):
    class Handler(BaseHTTPRequestHandler):
        server_version = "onram/0.1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            print("[onram] " + fmt % args, flush=True)

        def _json(self, value: Any, status: int = 200) -> None:
            data = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 64_000_000:
                raise ValueError("request body exceeds 64 MB")
            value = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(value, dict):
                raise ValueError("request body must be a JSON object")
            return value

        def _read_proxy_body(self) -> bytes | None:
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                return None
            try:
                length = int(raw_length)
            except ValueError as exc:
                raise ValueError("invalid Content-Length") from exc
            if length < 0 or length > 64_000_000:
                raise ValueError("request body must be between 0 and 64 MB")
            return self.rfile.read(length)

        @staticmethod
        def _proxy_paths() -> set[str]:
            # Keep this list explicit: Onram exposes the inference APIs used by
            # coding agents, not llama.cpp's administrative/model-loading API.
            return {
                "/v1/chat/completions",
                "/v1/chat/completions/control",
                "/v1/chat/completions/input_tokens",
                "/v1/responses",
                "/v1/responses/input_tokens",
                "/v1/messages",
                "/v1/messages/count_tokens",
            }

        def _proxy_request(self, method: str, body: bytes | None = None) -> None:
            # Preserve protocol-specific headers (Anthropic version/beta,
            # content negotiation, bearer/API-key auth) while stripping
            # hop-by-hop headers that belong to the client connection.
            hop_by_hop = {
                "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
                "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
                "accept-encoding",
            }
            connection_tokens = {
                token.strip().lower()
                for value in self.headers.get_all("Connection", [])
                for token in value.split(",")
                if token.strip()
            }
            headers = {"Accept-Encoding": "identity"}
            for key, value in self.headers.items():
                if key.lower() not in hop_by_hop | connection_tokens:
                    headers[key] = value
            request = urllib.request.Request(
                app.backend_url + self.path, data=body, headers=headers, method=method
            )
            with app.lock:
                try:
                    upstream = urllib.request.urlopen(request, timeout=24 * 60 * 60)
                except urllib.error.HTTPError as exc:
                    upstream = exc
                except (urllib.error.URLError, TimeoutError, OSError) as exc:
                    self._json({"error": f"llama-server unavailable: {exc}"}, 502)
                    return

                with upstream:
                    self.send_response(upstream.status)
                    has_content_length = False
                    for key, value in upstream.headers.items():
                        if key.lower() in {
                            "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
                            "te", "trailers", "transfer-encoding", "upgrade",
                        }:
                            continue
                        if key.lower() == "content-length":
                            has_content_length = True
                        self.send_header(key, value)
                    if not has_content_length:
                        # The backend's chunk framing has been removed by urllib.
                        # Close-delimited HTTP/1.0 responses preserve SSE incrementally.
                        self.send_header("Connection", "close")
                        self.close_connection = True
                    self.end_headers()
                    try:
                        while True:
                            read_chunk = getattr(upstream, "read1", upstream.read)
                            chunk = read_chunk(16_384)
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, http.client.HTTPException, OSError):
                        # A client cancelling a generation should stop consuming
                        # the local response without crashing the HTTP worker.
                        self.close_connection = True

        def _authorized(self) -> bool:
            if not app.api_key:
                return True
            authorization = self.headers.get("Authorization", "")
            bearer = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
            api_key = self.headers.get("x-api-key", "")
            return bearer == app.api_key or api_key == app.api_key

        def do_GET(self) -> None:
            try:
                if not self._authorized():
                    return self._json({"error": "unauthorized"}, 401)
                if self.path == "/api/version":
                    return self._json({"version": "0.1.0"})
                if self.path == "/api/health":
                    return self._json(app.status())
                if self.path == "/api/evolve/status":
                    return self._json(app.status()["evolution"])
                if self.path == "/v1/models":
                    return self._json({"object": "list", "data": [{
                        "id": app.model_name,
                        "object": "model",
                        "created": int(datetime.fromtimestamp(app.model_path.stat().st_mtime, timezone.utc).timestamp()),
                        "owned_by": "onram",
                    }]})
                if self.path == "/models":
                    return self._json({"object": "list", "data": [{
                        "id": app.model_name, "object": "model", "owned_by": "onram",
                    }]})
                if self.path == "/api/ps":
                    return self._json({"models": [{"name": app.model_name, "model": app.model_name,
                        "size": app.model_path.stat().st_size, "digest": "local-gguf",
                        "expires_at": ""}] if app.model_path.exists() else []})
                if self.path == "/api/tags":
                    info = {"name": app.model_name, "model": app.model_name,
                            "modified_at": datetime.fromtimestamp(app.model_path.stat().st_mtime, timezone.utc).isoformat(),
                            "size": app.model_path.stat().st_size, "digest": "local-gguf",
                            "details": {"format": "gguf", "family": "gpt2", "parameter_size": "82M", "quantization_level": "Q4_K_M"}}
                    return self._json({"models": [info]})
                if self.path == "/":
                    return self._json({"message": "onram is running", "docs": "/api/health"})
                return self._json({"error": "not found"}, 404)
            except Exception as exc:
                self._json({"error": str(exc)}, 500)

        def do_POST(self) -> None:
            try:
                if not self._authorized():
                    return self._json({"error": "unauthorized"}, 401)
                path = urllib.parse.urlsplit(self.path).path
                if path in self._proxy_paths():
                    return self._proxy_request("POST", self._read_proxy_body())
                body = self._read_json()
                if self.path in ("/api/generate", "/api/chat"):
                    if body.get("model") not in (None, "", app.model_name, "distilgpt2-nram"):
                        return self._json({"error": f"model not found: {body['model']}"}, 404)
                    prompt = body.get("prompt", "")
                    if self.path == "/api/chat":
                        messages = body.get("messages", [])
                        if not isinstance(messages, list):
                            return self._json({"error": "messages must be an array"}, 400)
                        prompt = "".join(f"{m.get('role', 'user')}: {m.get('content', '')}\n" for m in messages)
                        prompt += "assistant:"
                    if not isinstance(prompt, str) or not prompt:
                        return self._json({"error": "prompt or messages are required"}, 400)
                    started = time.monotonic()
                    if body.get("stream", True):
                        self.send_response(200)
                        self.send_header("Content-Type", "application/x-ndjson")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Transfer-Encoding", "chunked")
                        self.end_headers()
                        try:
                            for item in app.stream_generate(prompt, body.get("options") or {}):
                                piece = item.get("content", item.get("completion", ""))
                                done = bool(item.get("stop", False))
                                if self.path == "/api/chat":
                                    chunk = {"model": app.model_name,
                                        "created_at": datetime.now(timezone.utc).isoformat(),
                                        "message": {"role": "assistant", "content": piece}, "done": done}
                                else:
                                    chunk = {"model": app.model_name,
                                        "created_at": datetime.now(timezone.utc).isoformat(),
                                        "response": piece, "done": done}
                                if done:
                                    chunk["eval_count"] = item.get("tokens_predicted", 0)
                                    chunk["total_duration"] = int((time.monotonic() - started) * 1e9)
                                data = (json.dumps(chunk, ensure_ascii=False) + "\n").encode("utf-8")
                                self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
                                self.wfile.flush()
                            self.wfile.write(b"0\r\n\r\n")
                        except Exception as exc:
                            print(f"[onram] streaming error: {exc}", flush=True)
                        return
                    result = app.generate(prompt, body.get("options") or {})
                    text = result.get("content", result.get("completion", ""))
                    if self.path == "/api/chat":
                        response = {"model": app.model_name, "created_at": datetime.now(timezone.utc).isoformat(),
                                    "message": {"role": "assistant", "content": text}, "done": True,
                                    "total_duration": int((time.monotonic() - started) * 1e9),
                                    "eval_count": result.get("tokens_predicted", 0)}
                    else:
                        response = {"model": app.model_name, "created_at": datetime.now(timezone.utc).isoformat(),
                                    "response": text, "done": True,
                                    "total_duration": int((time.monotonic() - started) * 1e9),
                                    "eval_count": result.get("tokens_predicted", 0)}
                    return self._json(response)
                if self.path == "/api/evolve":
                    return self._json(app.begin_evolution(body), 202)
                if self.path == "/api/show":
                    return self._json({"modelfile": f"FROM {app.model_path}", "parameters": "",
                        "template": "{{ .Prompt }}", "details": {"format": "gguf", "family": "gpt2",
                        "parameter_size": "82M", "quantization_level": "Q4_K_M"},
                        "model_info": {"general.architecture": "gpt2"}, "license": ""})
                if self.path == "/api/pull":
                    return self._json({"error": "onram prototype only manages local models; use /api/import to add a GGUF"}, 501)
                if self.path == "/api/import":
                    return self._json({"error": "import accepts local files through the CLI only in this version"}, 501)
                return self._json({"error": "not found"}, 404)
            except (ValueError, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, 400)
            except RuntimeError as exc:
                self._json({"error": str(exc)}, 409)
            except Exception as exc:
                self._json({"error": str(exc)}, 500)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--llama-server", type=Path, default=DEFAULT_LLAMA_SERVER)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11434)
    parser.add_argument("--backend-port", type=int, default=11436)
    parser.add_argument("--ctx-size", type=int, default=0,
                        help="llama.cpp context size; 0 uses the GGUF's model default")
    parser.add_argument("--api-key", default=os.environ.get("ONRAM_API_KEY"))
    parser.add_argument("--model-name", help="Stable model name exposed by /api/tags")
    args = parser.parse_args()
    if args.host not in ("127.0.0.1", "localhost", "::1") and not args.api_key:
        parser.error("binding to a non-local address requires --api-key or ONRAM_API_KEY")
    if args.ctx_size < 0:
        parser.error("--ctx-size must be 0 or greater")
    model_name = args.model_name
    if model_name and ":" not in model_name:
        model_name += ":latest"
    app = Onram(args.model, args.llama_server, args.host, args.port, args.backend_port,
                args.bundle, args.api_key, model_name, args.ctx_size)
    app.start_backend()
    server = ThreadingHTTPServer((args.host, args.port), handler_for(app))
    print(f"onram listening at http://{args.host}:{args.port} with {app.model_path.name}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        app.close()


if __name__ == "__main__":
    main()
