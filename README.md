# NRA-M with DistilGPT2

This is a CPU-runnable research prototype using **DistilGPT2**, a 82M-parameter
distilled GPT-2 model. The downloaded checkpoint is pinned to
`2290a62682d06624634c1f46a6ad5be0f47f38aa`. The project includes a CPU-only
PyTorch environment, the local Hugging Face checkpoint, a CPU build of
llama.cpp, and F16/Q4 GGUF exports. [Model card](https://huggingface.co/distilbert/distilgpt2)

Weights, `.nram` bundles, GGUF exports, build products, and the Python virtual
environment are generated locally and are excluded from Git. To bootstrap the
default DistilGPT2 NRAM model from a clean checkout:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python download_model.py
./setup_onram.sh
.venv/bin/python nram_convert.py --model models/distilgpt2 \
  --output models/distilgpt2-nra-qa.nram
.venv/bin/python nram_export_generic.py --bundle models/distilgpt2-nra-qa.nram \
  --output-prefix models/onram-bootstrap
ln -s onram-bootstrap-Q4_K_M.gguf models/onram-current.gguf
./onram --bundle models/distilgpt2-nra-qa.nram
```

## Unified Onram command

The adapted `onram` command is the single entry point for the upstream
utilities and the workspace's Gemma NRA-M tools. Build its small native
launcher once, then use subcommands instead of invoking utility scripts
directly:

```sh
./build_onram.sh
./build/onram-cli --help
./build/onram-cli test
./build/onram-cli convert --help
./build/onram-cli nram info models/Gemma4-E2B-NRA.nram
./build/onram-cli nram-eval --help
./build/onram-cli launch --help
```

In one terminal, start the Onram API with a coding model:

```sh
./onram serve --model /path/to/coder-model.gguf --model-name local-coder
```

In another terminal, launch an installed agent through its native protocol
adapter (choose one):

```sh
./onram launch claude
./onram launch codex --model local-coder:latest
./onram launch pi -- --print "Review this repository"
```

`launch` targets `http://127.0.0.1:11434` by default. It checks `/v1/models`
and selects the first advertised model; if the server is down and the local
Llama 3.2 model and bundle are present, it starts Onram with interaction
learning enabled. `--model` or `ONRAM_MODEL` overrides model selection;
`--base-url` or `ONRAM_BASE_URL` selects another Onram instance, which must
already be running. `ONRAM_API_KEY` is forwarded as the local provider
credential. Put agent-specific arguments after `--`. For Pi, the command adds
or updates the `onram` provider in `PI_CODING_AGENT_DIR/models.json` (default
`~/.pi/agent/models.json`) and preserves the other providers.

For compatibility, `./onram --model ...` and `./onram --bundle ...` still
start the HTTP server; explicit `./onram serve ...` does the same. The launcher
can select the Python environment through `ONRAM_PYTHON`, otherwise it uses
this checkout's `.venv`, the parent workspace's `.venv-gemma4`, or `python3`.

The native launcher is a command dispatcher, not a self-contained model
runtime. Python dependencies, checkpoint files, and the `llama-server`
executable remain separate runtime requirements. The upstream v2 LoRA bundles
also remain a distinct format from the workspace's Gemma `.nram` v1 packages;
the `nram*` commands operate on v1 and the upstream training/export commands
operate on v2.

For this workspace, requirements use Transformers 5.19.0 so the generic path
can load the locally tested Gemma 4 architecture. A v1 Gemma package can serve
as the frozen base for a v2 LoRA experiment:

```sh
./onram convert --model ../../models/Gemma4-E2B-NRA.nram \
  --dtype bfloat16 --rank 4 --output models/gemma4-adapter.nram
./onram infer-generic "Answer briefly: what is a recurrent neural state?" \
  --bundle models/gemma4-adapter.nram --local-files-only
```

This conversion loads the full base model and saves only a LoRA adapter in v2.
It reads the v1 Safetensors tensor bank plus its tokenizer/config assets, but
does not apply v1 routes, associative memory, or output deltas. It preserves
Gemma's Transformer execution; it is not an architectural conversion to NRA-2.

The clone snapshot does not include an upstream `LICENSE` file. It is kept in
the workspace for internal research; clarify upstream licensing before
redistributing the clone or derived binaries.

The initial adapter is zero initialized, so the bootstrap GGUF is equivalent
to the base checkpoint. Use `/api/evolve` with verified training examples to
start changing it; Onram serves the current GGUF while validating candidates.

## What the evolving path does

`nram_evolve.py` freezes the original DistilGPT2 weights and attaches rank-4
LoRA projections to the last block's attention and MLP projections. It trains
only those 27,648 adapter parameters. A candidate is committed atomically to a
single `.nram` file only if:

1. Held-out completion-token negative log-likelihood improves.
2. Mean rank of the expected first answer token does not get worse.
3. Optional general-language guardrail loss stays within the configured drift
   limit (5% by default).

The `.nram` v2 container stores a JSON manifest, graph instructions, provenance,
version journal, checksummed Safetensors adapter tensors, and validation
metrics. Failed candidates leave the previously accepted file unchanged.

Train or continue the included NRA-M Q&A experiment:

```sh
.venv/bin/python nram_evolve.py \
  --train data/nra_qa_train.jsonl \
  --validation data/nra_qa_validation.jsonl \
  --guardrail data/general_guardrail.jsonl \
  --epochs 8 --learning-rate 0.0001
```

Use the adapter with Transformers:

```sh
.venv/bin/python nram_infer.py $'Explain the ENCODE operation in NRA-M.\nAnswer:' \
  --max-new-tokens 24 --min-new-tokens 6
```

## First real result

The accepted `models/distilgpt2-nra-qa.nram` bundle is version 3. On the small
included held-out set, its completion NLL fell from **5.660 to 3.111** and
general guardrail NLL from **3.604 to 2.410**. The rank of the expected first
answer token moved from mean rank **6.0 to 1.67**, with top-1 hits improving
from **0/6 to 2/6**. These measurements are limited to this tiny synthetic NRA
Q&A task; they do not show a broad reasoning or language-quality improvement.

Greedy generations can still produce incorrect or generic continuations even
when token-level validation improves. Treat this as a working parameter-update
and rollback experiment, not as a reliable assistant model. The evaluator now
tracks first-token ranking to catch candidates that improve average loss while
making the next answer token worse.

## GGUF output

Yes, after a candidate passes, merge the adapter into a fresh copy of the base
weights and export that checkpoint as a new GGUF. The current evolved version
has these files:

- `models/onram-nram-v3-f16.gguf` — merged higher-precision checkpoint.
- `models/onram-nram-v3-Q4_K_M.gguf` — quantized copy for smaller local
  inference.
- `models/distilgpt2-nra-qa.nram` — adapter, graph, provenance, and evolution
  journal used to reproduce the merged checkpoint.

The `.gguf` is the inference checkpoint. The `.nram` holds the change process
and reversible state. GGUF is a model storage format, not an implementation of
the mutation runtime. See the [GGUF format specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).

To merge and export a later accepted `.nram` version:

```sh
./export_evolved.sh models/distilgpt2-nra-qa.nram distilgpt2-nra-next
```

Each training run starts a fresh AdamW optimizer; the accepted adapter and
version history persist, while optimizer moments are not stored yet.

The local llama.cpp checkout is pinned to commit
`1b70350c92563449576bcbbf552994b1d1ed847d`. Its converter needed a one-line
local patch to omit GPT-2's stored causal-mask buffers; `export_gguf.sh` applies
that patch when setting up a fresh checkout.

## Verification

```sh
.venv/bin/python verify_local.py
.venv/bin/python verify_evolved.py
.venv/bin/python -m unittest -v
```

`verify_evolved.py` checks that the adapter and merged checkpoint produce
matching logits, the held-out metrics have not regressed, and llama.cpp loads
and generates from the evolved F16 GGUF. The core unit tests cover transaction
rejection, checksums, `.nram` tensor persistence, and that a zero-initialized
LoRA adapter leaves the original model output unchanged.

## Onram alpha: local inference plus supervised evolution

`onram.py` is a small Ollama-shaped HTTP service around `llama-server`. It
serves a local GGUF through `/api/generate` and `/api/chat`, and includes
`/api/tags`, `/api/show`, `/api/ps`, and `/api/version`. It streams generation
results in newline-delimited JSON. It does not include Ollama's model
registry/pull, Modelfile build, embeddings, cloud, or platform packaging
features. Coding-agent API compatibility is described below.

The local llama.cpp checkout can build its server without downloading UI
assets:

```sh
.venv/bin/cmake -S third_party/llama.cpp -B build \
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_CURL=OFF
.venv/bin/cmake --build build --target llama-server -j2
./onram
```

The default HTTP port is `11434`, so existing clients that call Ollama's
`/api/generate` or `/api/chat` can point at Onram. Example:

```sh
curl http://127.0.0.1:11434/api/generate -d '{
  "model":"distilgpt2-nram:latest",
  "prompt":"ENCODE means",
  "stream":false,
  "options":{"num_predict":32,"temperature":0.2}
}'
```

An accepted training run can be started while the current GGUF keeps serving:

```sh
curl http://127.0.0.1:11434/api/evolve -d '{
  "train":"nra_qa_train.jsonl",
  "validation":"nra_qa_validation.jsonl",
  "guardrail":"general_guardrail.jsonl",
  "epochs":2,
  "learning_rate":0.0005
}'
curl http://127.0.0.1:11434/api/evolve/status
```

The evolution endpoint only accepts the project datasets under `data/`. The
trainer validates the candidate before committing `.nram`; after acceptance,
Onram exports a fresh GGUF and restarts the inference worker on that file. This
short reload can interrupt requests. A local `models/onram-current.gguf`
pointer records the active version across restarts. Inference prompts are not
training data: the model does not silently learn from user conversations.

The original evolution endpoint supports the pinned DistilGPT2 experiment. The
generic Hugging Face causal-LM conversion and evolution workflow is documented
below. Passing a different llama.cpp-compatible GGUF with `--model` changes
inference only; an evolvable NRAM bundle must be created from trainable source
weights. Binding to a non-local address requires `--api-key` or `ONRAM_API_KEY`.

## Coding-agent API proxy

Onram also exposes the local model through the API formats used by Claude Code,
Codex, and Pi. Requests and streaming responses are relayed to llama.cpp without
rewriting their JSON or SSE events, preserving tool declarations and tool
results. The supported routes are:

- `POST /v1/messages` and `/v1/messages/count_tokens` for Anthropic Messages.
- `POST /v1/responses` and `/v1/responses/input_tokens` for OpenAI Responses.
- `POST /v1/chat/completions`, `/v1/chat/completions/control`, and
  `/v1/chat/completions/input_tokens` for OpenAI Chat Completions.
- `GET /v1/models` returns the active Onram model alias.

Start the service with the GGUF and a stable ID for the coding agents:

```sh
./onram --model /path/to/coder-model.gguf --model-name local-coder
```

The model ID reported by `/v1/models` is then `local-coder:latest`. A bearer
token or Anthropic `x-api-key` is accepted when `--api-key`/`ONRAM_API_KEY` is
set. Without a key the server only binds to localhost by default.
By default llama.cpp uses the GGUF's context setting; pass `--ctx-size N` to
override it (larger contexts use more memory).

### Claude Code

Claude Code uses Anthropic Messages and supports a custom base URL. For a
key-protected Onram instance:

```sh
export ANTHROPIC_BASE_URL=http://127.0.0.1:11434
export ANTHROPIC_API_KEY="$ONRAM_API_KEY"
export ANTHROPIC_MODEL=local-coder:latest
claude
```

For a localhost-only instance without `--api-key`, set a dummy
`ANTHROPIC_API_KEY=local-onram`; Onram ignores it when authentication is
disabled. Anthropic's `ANTHROPIC_AUTH_TOKEN` can be used instead and is sent as
a bearer token.

### OpenAI Codex

Codex uses the Responses API. Add a custom provider to `~/.codex/config.toml`:

```toml
model = "local-coder:latest"
model_provider = "onram"

[model_providers.onram]
name = "Onram local"
base_url = "http://127.0.0.1:11434/v1"
wire_api = "responses"
requires_openai_auth = false
supports_websockets = false

[model_providers.onram.capabilities]
remote_compaction = "unsupported"
```

For a key-protected server, add `env_key = "ONRAM_API_KEY"` to that provider
table and set the variable before starting Codex. Codex's HTTP Responses API works through this
proxy; WebSocket transport and remote compaction are not provided. Codex's
Responses API configuration is documented in the [Codex config
reference](https://developers.openai.com/codex/config-reference/).

### Pi

Add a provider to `~/.pi/agent/models.json`:

```json
{
  "$schema": "https://pi.dev/schemas/models.schema.json",
  "providers": {
    "onram": {
      "baseUrl": "http://127.0.0.1:11434/v1",
      "api": "openai-completions",
      "apiKey": "${ONRAM_API_KEY}",
      "models": [{ "id": "local-coder:latest" }]
    }
  }
}
```

Pi requires an API key entry to make the custom model available. Use
`"apiKey": "local-onram"` for an unauthenticated local server, or use
`${ONRAM_API_KEY}` when Onram authentication is enabled. Pi's provider format
is documented in [Choose a model](https://pi.dev/docs/latest/models).

Claude Code's endpoint variables are documented in [LLM gateway
configuration](https://code.claude.com/docs/en/llm-gateway).

These routes provide protocol compatibility, not a guarantee that every model
can act as a coding agent. Tool use depends on the GGUF's chat template and
training; llama.cpp documents that tool calling requires a compatible Jinja
template. Codex web search, remote compaction, and WebSocket transport are not
implemented by this local proxy. Check the model's context limit before using
large coding-agent conversations.

## Generic conversion for Hugging Face causal LMs

The generic CLI adds an architecture-aware path for dense causal language
models that load with `AutoModelForCausalLM`. It recognizes common attention
and MLP projection names, supports `nn.Linear` storage and GPT-2 `Conv1D`
storage, and defaults to LoRA on the last transformer block. It records the
exact module paths and tensor layouts in the `.nram` manifest. If auto-detection
does not fit a model, specify projection suffixes with `--targets`.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python download_model.py
./setup_onram.sh

.venv/bin/python nram_convert.py \
  --model /path/to/huggingface-checkpoint \
  --output models/my-model.nram \
  --rank 8 --last-k-blocks 2

.venv/bin/python nram_evolve_generic.py \
  --bundle models/my-model.nram \
  --train data/nra_qa_train.jsonl \
  --validation data/nra_qa_validation.jsonl \
  --guardrail data/general_guardrail.jsonl \
  --epochs 2 --local-files-only

.venv/bin/python nram_export_generic.py \
  --bundle models/my-model.nram \
  --output-prefix models/my-model-evolved
```

The `.nram` bundle stores the trainable adapter and the base checkpoint's
source/revision/fingerprint; it does not duplicate the full base weights. Use a
Hugging Face checkpoint or repository with trainable weights and a tokenizer.
Inference-only GGUF files cannot be converted into trainable weights by this
path. GGUF export also depends on llama.cpp having a converter for the model's
architecture. Architectures with custom projections may need explicit target
names, and custom model code must be opted into with `--trust-remote-code`.

Generic evolution automatically selects CUDA, then Apple MPS, then CPU. CPU
training uses up to 10 PyTorch threads and batches four examples to reduce
per-example model calls; override with `--device`, `--threads`, and
`--batch-size`. The Onram status endpoint exposes the current stage, batch
count, device, and elapsed seconds while training.

After each run, the `.nram` manifest stores an `inference_attribution` profile.
It replays captured prompt/answer pairs through the trainable Transformers
model, records adapter-input/output magnitudes and per-layer LoRA gradient
strength, and uses the strongest 60% of layers as the next run's update targets
once a prior profile exists. Validation and the general guardrail still decide
whether changed adapter weights are accepted. This is gradient-based attribution
of captured responses; llama.cpp's GGUF runtime does not identify a unique set
of causal weights for generated text. A rejected candidate saves the diagnostic
profile and validation report while retaining the previous adapter tensors.

For an instrumented Transformers generation, the `nram-debug` command runs the
prompt through the bundle and records per-layer hidden-state RMS/top channels
plus each LoRA adapter's contribution magnitude. It prints the answer but
stores only prompt/answer hashes in the JSON report. Add
`--gradient-attribution` to run a slower teacher-forced gradient pass over the
generated answer and rank adapter targets for investigation:

```sh
./build/onram-cli nram-debug "Explain this function and its edge cases." \
  --bundle models/llama3.2-1b-instruct.nram \
  --local-files-only --gradient-attribution \
  --output results/inference-debug.json
```

This is a Transformers replay using the `.nram` bundle, not a trace from the
live quantized `llama-server` GGUF process. Activation magnitudes and
gradient-based adapter scores are diagnostic signals, not proof that a specific
base-model weight caused an answer. The debugger never changes weights; a
candidate still needs separate validation and promotion gates. The additional
gradient pass can use substantially more memory.

Normal agent requests through Onram also use the patched GGUF runtime debugger
by default. Build Onram's local `llama-server` with `./setup_onram.sh`, then
inspect its state at `GET /api/debug/status`. It writes one compact JSONL entry
per request under the model's ignored learning directory, with prompt/answer
hashes and per-layer final-residual RMS/top-channel summaries. Raw text and full
activation vectors are not written. Disable runtime tracing with
`--no-inference-debug` or `ONRAM_INFERENCE_DEBUG=0`.

The GGUF trace comes from the actual quantized server request. It exposes
activation magnitudes, not gradients through quantized weights; use the
Transformers `nram-debug --gradient-attribution` replay for adapter-gradient
salience. Neither score alone proves that a weight caused an answer, and neither
automatically mutates the model.

To keep an arbitrary converted model in Onram's inference loop, first export
the initial (zero-adapter) checkpoint and link its GGUF to a model-specific
current pointer, then start Onram with that bundle. Its `/api/evolve` endpoint
detects the generic bundle, trains/validates it, and reloads a new GGUF after an
accepted candidate:

```sh
.venv/bin/python nram_export_generic.py \
  --bundle models/my-model.nram \
  --output-prefix models/my-model-bootstrap
ln -s my-model-bootstrap-Q4_K_M.gguf models/my-model-current.gguf
./onram --model models/my-model-current.gguf \
  --bundle models/my-model.nram --model-name my-model --port 11434
```

The generic server mode still uses the bundled example datasets by default;
replace them with curated examples and separate held-out validation/guardrail
sets before relying on training results.

## Learning from local API interactions

The server enables local interaction learning by default, including when
launched through `./build/onram-cli serve` or `./onram serve`. To disable it for
one run, pass `--no-auto-learn` or set `ONRAM_AUTO_LEARN=0`. Start the configured
Llama bundle with:

```sh
./start-onram-learning.sh
```

The same launcher enables the per-layer GGUF trace. Recent reports are stored
in `models/llama3.2-1b-instruct-learning/inference-traces.jsonl`; the temporary
callback stream is cleared after each request. The server uses one parallel
slot so each trace remains associated with the request that produced it.
`./onram-cli runtime-setup` applies the llama.cpp callback patch from
`patches/llama-cpp-inference-trace.patch` before rebuilding the server. The
runtime GGUF is downloaded separately because model files are local artifacts:

```sh
hf download bartowski/Llama-3.2-1B-Instruct-GGUF \
  Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --local-dir models/llama3.2-1b-instruct-runtime-gguf
ln -s llama3.2-1b-instruct-runtime-gguf/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  models/llama3.2-1b-instruct-current.gguf
```

The trace stores prompt/response hashes and per-layer residual activation
summaries; it does not persist prompt or response text or full activations.

### Feed answer quality back into selective training

When Onram is running, list recent traces and copy the `trace_id` for the
answer you want to rate. Then submit `good` to keep that generated pair as a
positive example, or `bad` with a corrected answer to train on your correction:

```sh
curl -s http://127.0.0.1:11534/api/debug/traces
curl -s -X POST http://127.0.0.1:11534/api/learning/feedback \
  -H 'Content-Type: application/json' \
  -d '{"trace_id":"<trace-id>","rating":"bad","corrected_completion":"Respuesta corregida por ti."}'
```

Use `"rating":"good"` to approve a response; `corrected_completion` is only
accepted with a bad rating. A bad answer without a correction is excluded from
the training split. The correction replaces that answer as the supervised
target and is stored in the feedback ledger because it is text explicitly
provided by you. Feedback is matched to the actual GGUF inference using its
trace ID and text hashes; the trace ledger itself continues to store hashes,
not conversation text.

Three distinct bad traces with your corrections are required before their GGUF activation
profiles can guide initial layer selection. The profile compares per-layer
residual RMS on corrected bad answers with approved good answers when those are
available. It is a selection hint, not a causal explanation: LoRA gradients,
held-out validation, personal validation, and general guardrails still gate
the candidate. Automatic evolution also requires the configured interaction
batch plus at least 10 training and 3 validation examples. Check
`GET /api/learning/status` and `GET /api/evolve/status` to see whether feedback
has accumulated or an evolution run has started.

Claude Code, Codex, Pi, or another client must send inference requests to this
Onram instance for those turns to be captured. The listener is local-only at
`http://127.0.0.1:11534`; the current setup uses `/v1/messages` (Anthropic),
`/v1/responses` (OpenAI Responses), `/v1/chat/completions`, and `/api/chat` or
`/api/generate`. The current Codex conversation is not captured unless Codex is
configured to use this provider.

On this computer, the clients are configured as opt-in profiles:

```sh
./claude-local.sh
codex --profile onram
pi --provider onram --model llama3.2-1b-instruct:latest
```

For another OpenAI-compatible client, set its base URL to
`http://127.0.0.1:11534/v1`, its API key to `local-onram`, and its model to
`llama3.2-1b-instruct:latest`. Start Onram first with `./start-onram-learning.sh`.
The existing Codex default model and Pi's Ollama provider remain available.

Llama 3.2 1B supports up to 128K tokens. The launchers leave llama.cpp's
`--ctx-size` at `0`, which loads the GGUF's native maximum (131072 tokens for
this model); Pi and the Onram Codex profile advertise the same limit. Long
prompts use more unified memory, so actual usable context can still depend on
available memory.

Onram stores unique, completed text pairs in the ignored file
`models/llama3.2-1b-instruct-learning/interactions.jsonl`. It omits system
prompts and tool payloads, skips tool-call/partial responses, and keeps only
recent user text plus the final assistant text. It also records explicitly
stated names and preferences in the ignored `personal_facts.json` file and
builds dedicated personal question/answer examples for evolution. Every 24 new
conversation pairs, it trains a candidate and validates both generated fact
recall and general behavior. A new or changed personal fact gets 50 focused
epochs; later micro-updates use 8 epochs. Personal examples use the model's
chat template and include held-out paraphrases. A personal-fact candidate is
promoted only when held-out generation answers at least 75% of those questions
with the expected fact, personal loss improves, and general validation and the
guardrail stay within limits.
Accepted candidates reload automatically. An interrupted attempt is resumed
when the server starts again. Check progress and the number of
captured personal facts at `GET /api/learning/status`; `/api/evolve/status`
shows the current candidate run. This is still local self-distillation from
the model's own final answers, not a signal that the user approved each answer.
Inspect the captured files and periodically review or remove entries that
should not be retained. Training uses the BF16 Hugging Face base and can put
memory pressure on a 16 GB machine.
