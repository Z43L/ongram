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
./atenhea --bundle models/distilgpt2-nra-qa.nram
```

## Unified Onram command

The adapted `onram` command is the single entry point for the upstream
utilities and the workspace's Gemma NRA-M tools. Build its small native
launcher once, then use subcommands instead of invoking utility scripts
directly:

```sh
./build_onram.sh
./build/atenhea --help
./build/atenhea list
./build/atenhea test
./build/atenhea convert --help
./build/atenhea nram info models/Gemma4-E2B-NRA.nram
./build/atenhea nram-eval --help
./build/atenhea launch --help
```

List downloaded models with `./atenhea list` (alias: `./atenhea models`). This
scans `models/` recursively without starting the server, showing format, size
in GiB and absolute path. Split GGUF files are grouped into one model; incomplete
downloads are excluded. Use `--json` for structured output or `--dir /path/to/models`
to inspect another directory. Local GGUF, NRAM and Hugging Face weight directories
are supported.

In one terminal, start the Onram API with a coding model:

```sh
./atenhea serve --model /path/to/coder-model.gguf --model-name local-coder
```

In another terminal, launch an installed agent through its native protocol
adapter (choose one):

```sh
./atenhea launch claude
./atenhea launch codex --model local-coder:latest
./atenhea launch pi -- --print "Review this repository"
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

For compatibility, `./atenhea --model ...` and `./atenhea --bundle ...` still
start the HTTP server; explicit `./atenhea serve ...` does the same. The launcher
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
./atenhea convert --model ../../models/Gemma4-E2B-NRA.nram \
  --dtype bfloat16 --rank 4 --output models/gemma4-adapter.nram
./atenhea infer-generic "Answer briefly: what is a recurrent neural state?" \
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
./atenhea
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
./atenhea --model /path/to/coder-model.gguf --model-name local-coder
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
./build/atenhea nram-debug "Explain this function and its edge cases." \
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
./atenhea --model models/my-model-current.gguf \
  --bundle models/my-model.nram --model-name my-model --port 11434
```

The generic server mode still uses the bundled example datasets by default;
replace them with curated examples and separate held-out validation/guardrail
sets before relying on training results.

## Learning from local API interactions

The server enables local interaction learning by default, including when
launched through `./build/atenhea serve` or `./atenhea serve`. To disable it for
one run, pass `--no-auto-learn` or set `ONRAM_AUTO_LEARN=0`. Start the configured
Llama bundle with:

```sh
./start-onram-learning.sh
```

The same launcher enables the per-layer GGUF trace. Recent reports are stored
in `models/llama3.2-1b-instruct-learning/inference-traces.jsonl`; the temporary
callback stream is cleared after each request. The server uses one parallel
slot so each trace remains associated with the request that produced it.
`./build/atenhea runtime-setup` applies the llama.cpp callback patch from
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

## Strata backend on Apple Silicon

Install Strata, its private Python environment, Metal engine and model from Onram:

```sh
./atenhea strata-setup --no-start
./start-strata.sh
```

The installer copies the existing `../strata` checkout (override with `--source`);
it never clones it or modifies that installation. If no local source exists,
it downloads the GitHub source archive automatically; `--archive-url` can pin
a specific commit archive. No separate Strata installation is required. Runtime code and build tools
live in `third_party/strata`, and weights/packs live in `models/strata`.
Existing downloaded weights and partial downloads are copied independently,
using APFS clones where supported. Later runs resume downloads. Source provenance is recorded in `third_party/strata/onram-source.json` (revision when available, or archive SHA-256).
Strata's own installer installs dependencies, compiles Metal, downloads missing
model files and prepares its expert streaming layout. Internet access and the
Apple Command Line Tools are required. The default is Q2_0, text only, 2048
context tokens. Setup needs roughly 100 GB available for weights and preparation.
16 GB Macs are experimental: successful setup does not establish usable inference.

`strata-setup` starts Onram after installation unless `--no-start` is supplied.
Installation output can be saved with `./atenhea strata-setup --no-start > strata-install.log 2>&1`.
To use an already prepared installation explicitly:

```sh
./atenhea serve --backend strata \
  --strata-dir /absolute/path/to/strata \
  --strata-config /absolute/path/to/strata/strata-q2_0.json
```

Onram starts Strata on localhost port 11436 and exposes its usual port 11434.
OpenAI Chat Completions/Responses and Anthropic requests are relayed to Strata;
Ollama generate/chat requests are translated to Chat Completions, including SSE
streaming and chat roles. `/api/generate` treats its prompt as a user message.
Model size metadata is reported as unknown (zero), not as the default GPT-2 size.
Strata controls model context through its generated JSON; Onram's `--ctx-size`
only applies to llama.cpp. Override the interpreter with `--strata-python`.
Automatic evolution and llama.cpp activation traces are disabled for Strata;
manual evolution returns an explicit unsupported error. The existing llama.cpp
backend remains the default. Installation does not change client provider settings.

`./atenhea launch pi` also starts the installed Strata backend automatically when
Onram is down; Q2_0 is preferred if multiple configurations exist. No Llama 3.2
files are required in that case. On Macs with at most 20 GiB of RAM the installer
disables the large prompt GEMM buffers and uses decode windows for prefill.
This trades prompt speed for lower memory use. Avoid running a second model
server alongside it on a 16 GB Mac.

## Gemma 4 26B-A4B with native maximum context

```sh
./atenhea gemma-setup --no-start
./start-gemma.sh
# in another terminal:
./atenhea launch pi
```

This installs the latest stable llama.cpp release as a separate Metal runtime
under `third_party/llama-current`, builds `build/llama-current/bin/llama-server`,
and downloads Unsloth's UD-IQ3_S GGUF (~11.3 GB), pinned to a model revision and
verified by SHA-256. The setup is resumable. Existing research runtime patches
remain separate. CMake must be installed (the Strata environment's CMake can
also be reused).

The generated `models/onram-launch.json` selects Gemma for agent autostart.
Its context is explicitly 262144 (the native maximum), with a single sequence,
Q4_0 KV cache, Flash Attention and small prompt batches. Model weights use mmap
and may be paged from SSD. Automatic fitting may adjust GPU offload, but the
minimum fit context is also 262144. No wired-memory system settings are changed.
Pi receives a 262144-token context window when launching this profile.

On a 16 GB Mac, configured capacity does not prove that a fully occupied 262K
conversation fits or performs acceptably. Paging and context processing can be
slow. Stop other model servers before starting Gemma. Auto learning and runtime
activation tracing are disabled for this unpatched inference runtime.

## Apple Silicon SSD / Metal performance work

The optional Metal IO expert-reader patch and a reproducible native benchmark
are described in [docs/apple-ssd-performance.md](docs/apple-ssd-performance.md).
`./atenhea ssd-bench` measures disk-to-shared-memory reads plus a checksum kernel,
without downloading models. `tools/bench_strata_ssd.py` compares real generations
and logs memory pressure / swap. The original pread reader stays the default;
Metal IO is selected explicitly in the Strata config. Gemma downloads are not
resumed by these tools.

## CHL episodic memory for direct `.nram` inference

The `infer` and `infer-generic` commands use `models/athenea.memory` by default
(resolved relative to this project, regardless of the working directory).
Override it with `--chl-memory PATH`; the memory utility also defaults to this
archive. Before
inference they retrieve up to four CHL episodes (at most 4,000 characters of
serialized episode data); after successful generation they persist the original
prompt and answer as an **unverified** episode. Reuse the same path across calls.
The original `.nram` is not modified. The HTTP server and `onram launch` also use this archive by default.

Install the actual upstream engine (the checkout is ignored by Git):

```sh
git clone https://github.com/Z43L/CHL-episodic-agent-memory.git third_party/chl
git -C third_party/chl checkout 3d5d7700384b55c2c29c84d8a7becb91eaf210ef
# Optional native acceleration; requires Node headers and clang++:
node third_party/chl/scripts/build-native.js
```

Node.js must be available on PATH, or set `ONRAM_NODE` to its executable.
`ONRAM_CHL_REPO` can point to another checkout. No cloud model or API key is
needed. CHL uses its JavaScript fallback when the native addon is unavailable;
`recall` reports the active engine. The bridge locks the archive across processes
and waits for CHL hydration before each operation. Use one archive per user and
project; archives store conversation text. Keep archives under ignored `models/`.

```sh
./atenhea infer-generic '¿Qué decidimos sobre el almacenamiento?' \
  --bundle models/my-model.nram --chl-memory models/my-model.memory

# Add a reviewed question/answer, or a corrected answer, as verified evidence:
python3 episodic_memory.py --memory models/my-model.memory remember \
  '¿Dónde guardamos los pedidos?' 'En PostgreSQL.' --verified
python3 episodic_memory.py --memory models/my-model.memory recall 'pedidos'
python3 episodic_memory.py --memory models/my-model.memory export \
  --output data/chl-reviewed.jsonl
```

Export includes only explicitly verified episodes, deduplicated with provenance
hashes, in the `prompt`/`completion` format consumed by `nram_evolve_generic.py`.
Unreviewed generations remain retrievable historical data and are labelled as
potentially incorrect in the context. Retrieved content is labelled as data,
with the current request taking priority; this is a prompt instruction, not a
security boundary. The character budget is not a tokenizer context-window check.

Memory retrieval changes the input context; it does not train model weights.
To evolve weights, use reviewed exports as training input in the existing
`onram evolve` workflow, with a separate, independent validation dataset and its
candidate acceptance checks. This bridge does not automatically validate its own
answers, schedule training, or promote a new model. Improvements in recall or model
quality must be measured; persistence alone does not establish an improvement.

Validate the bridge without loading model weights:

```sh
python3 -m unittest test_episodic_memory -v
```


### Shared Athenea memory for coding agents

Start agents through the existing launch commands:

```sh
./atenhea launch pi
./atenhea launch claude
./atenhea launch codex
```

All three use the Onram server, which now defaults to
`models/athenea.memory`. It retrieves episodes for Anthropic Messages, OpenAI
Chat Completions, OpenAI Responses, and Ollama chat/generate requests, including
streaming responses. Only completed text answers are stored; tool calls and
partial answers are omitted by the existing extractor. Capture works even with
`--no-auto-learn` and with the Strata backend. The agent tool protocol and
original request text are preserved; injected context is not written back as a
new user episode. CHL errors appear in the server log and at
`/api/health` under `episodic_memory.last_error`; requests continue without memory
when retrieval fails.

Override the server archive with `onram serve --chl-memory PATH`.
Restart an already running Onram server to load these changes. Agents started
against other providers do not use this local memory.

## Athenea: documented and evaluated self-improvement

`docs/athenea-capabilities.json` describes the system's identity, available
mechanisms, known limitations, treatment of user facts and world knowledge,
and criteria for accepting improvements. It deliberately separates documented
mechanisms from capabilities demonstrated by benchmarks.

```sh
onram rsi-prepare --ingest-docs
onram rsi-prepare --analyze
```

Preparation writes `models/athenea-rsi/self-knowledge.json` (full allowlisted
source snapshots with SHA-256 hashes), `analysis-prompt.txt`, and, when analysis
completes, `unverified-proposal.json`. Documentation ingestion writes eight
verified, typed `knowledge` records to `models/athenea.memory`, tagged by area
(`self`, `user`, `world`, `code`). These are verified statements about the
implementation, not evidence of model accuracy. Model proposals remain
unverified and are never interpreted as executable code or optimizer settings.
`--analyze` calls the local model through Onram and requires a complete text
response; a truncated response is not accepted as a completed analysis.

The generic LoRA trainer now defaults to embedding this documentation and the
allowlisted source code in the `.nram` manifest under `self_knowledge`.
This is inert documentation; it is never imported or executed from the bundle.
It adds a bounded number of grounded documentation examples to training and
at most 32 verified CHL replay examples. More recent verified corrections take
priority for the same question. Unverified episodes and model-generated
proposals are excluded. Exact normalized evaluation questions are excluded from
replay; overlapping original train/evaluation prompts fail before training.
Semantic overlap still requires dataset review.

Before optimization, the trainer records shapes, parameter counts and numerical
RMS for adapter tensors and their frozen base projections. It records target
selection, baseline validation and a completed local-model analysis proposal.
The proposal is advisory: the existing gradient/activation selector controls
which LoRA modules are trainable. There is no claim that activation magnitudes
explain the causal meaning of individual weights. Inspecting projections does
not inspect every parameter of the full model.

After training, existing validation and regression gates decide acceptance.
Non-finite loss, gradient norms, metrics or inspected tensors prevent accepting
weights. Rejected runs retain original tensors; diagnostic metadata describes
the candidate and records its rejection. Analysis files are written beside the
bundle in `<bundle-stem>-analysis/`. The effective augmented training dataset
has its own hash in `self_knowledge.training_context`.

To select an area, use `onram evolve --specialization self|user|world|code`.
This filters supplemental documentation and CHL replay by area; the explicitly
provided training dataset remains the primary task data. Verified memories can
be tagged with `python3 episodic_memory.py remember QUESTION ANSWER --verified
--area code`. `--no-self-analysis` runs offline with numerical preflight only;
`--no-self-knowledge` disables this entire optional layer. Neither changes the
existing model architecture.

A bounded recursive candidate workflow is available:

```sh
onram rsi-train --bundle models/athenea.nram \
  --train data/specialized-train.jsonl \
  --validation data/independent-validation.jsonl \
  --guardrail data/general_guardrail.jsonl \
  --specialization code --rounds 2
```

The named bundle and datasets must actually exist. This command requires the
Python training dependencies and a trainable generic NRAM v2 bundle with its
Transformers base checkpoint, plus a running local Onram model for the analysis
step. A GGUF alone cannot satisfy these requirements. It copies each candidate
to a new run directory, continues only from accepted candidates, stops on the
first rejection, and caps work at three rounds. The original bundle and running
GGUF remain unchanged. The final candidate still needs a separate untouched
benchmark before deployment: repeatedly consulting one validation set can
adaptively overfit it. This implements bounded evaluated improvement, not a
claim of unrestricted RSI or an established intelligence gain.

```sh
python3 -m unittest test_self_improvement test_episodic_memory test_onram_memory -v
```

### Trainable Athenea E2B checkpoint and NRAM

`onram e2b-trainable-setup` downloads only `google/gemma-4-E2B-it` into
`models/gemma4-e2b-trainable/`, pins its Hub commit, resumes completed byte ranges,
and verifies each LFS SHA-256. It then creates `models/athenea-e2b.nram` using
`onram convert --low-memory --dtype bfloat16`. An existing bundle is preserved.

The low-memory converter builds the architecture on PyTorch's meta device,
creates only CPU LoRA tensors, and compares each selected projection shape with
the downloaded Safetensors headers. It never materializes the full base in RAM.
The v2 `.nram` contains adapters, source identity and fingerprint, code/document
snapshots and the `athenea.memory` path. The base weights remain in their local
checkpoint directory; keep that directory alongside the bundle. Zero-initialized
LoRA residuals preserve the original model until training is performed.

This provides trainable source files, not proof that this Mac has enough free
memory for a complete training or inference pass. The full multimodal E2B
checkpoint needs more memory than its existing quantized GGUF. Stop other model
servers or use a machine with enough memory before loading it for training.

```sh
onram e2b-trainable-setup
# Equivalent conversion after download, with an explicit alternative output:
onram convert --model models/gemma4-e2b-trainable \
  --output models/athenea-e2b-copy.nram --dtype bfloat16 --low-memory
```

## Direct Onram inference and online micro-training (E2B)

The Athenea launch profile now uses `--backend nram`: Onram loads the downloaded
BF16 Transformers weights and `models/athenea-e2b.nram` directly. It keeps the
model resident on Apple MPS when available and attaches
`models/athenea-e2b.extensions.nram`. No GGUF conversion, merge or llama.cpp
process is involved in this backend. Start it with `onram`, `onram serve`,
`./start-athenea.sh`, or `onram launch pi|claude|codex`. Explicit backend/model
arguments still override the saved launch profile.

The base checkpoint stays frozen. Three external residual micro-MLPs are seeded
from actual rows of the final E2B MLP gate projection. Each expert has 24,576
parameters; the activation router adds 4,608, so one expert plus the router has
29,184 trainable parameters. Initial residual outputs are zero. These extensions
need the E2B's activations; they are not standalone language models. Their
channel partitions have no established factual or semantic meaning.

For offline preparation and candidate training:

```sh
onram micro-train cache --bundle models/athenea-e2b.nram \
  --train data/reviewed-train.jsonl --validation data/heldout.jsonl \
  --output models/athenea-cache --max-length 64
onram micro-train train --extension models/athenea-e2b.extensions.nram \
  --cache models/athenea-cache --output models/micro-candidate.nram --expert 0
```

Caching runs the frozen E2B prefix once. Subsequent training uses cached
features, the exact final-layer tail (including E2B's per-layer inputs), and one
small expert. Updates use 2,048 sampled vocabulary candidates by default;
validation normalizes over all 262,144 vocabulary tokens using bounded FP32
projection chunks. This is not a full-model speed benchmark. Tests compare
cached-tail logits with a complete small Gemma forward pass. Rebuild the cache
when upstream base or LoRA weights change.

The running backend also exposes `POST /api/train`. Send JSON arrays `train`,
`validation` and `guardrail`, each with 1..128 examples containing `prompt`,
`completion` and explicit `verified: true`. Optional settings include `expert`
(default 0), `epochs` (1..20), `max_length` (16..256), and `learning_rate`
(positive, at most 0.01). The endpoint serializes with generation, captures the
current frozen prefix, trains a separate candidate, and accepts only improved
held-out NLL with at most 2% guardrail NLL drift. It leaves the active extension
unchanged on rejection. Acceptance updates its in-memory tensors and atomically
saves the sidecar, without restarting the model. `GET /api/runtime/status`
reports the current version and last result.

```sh
curl http://127.0.0.1:11434/api/runtime/status
curl http://127.0.0.1:11434/api/train -H 'Content-Type: application/json' \
  --data-binary @models/athenea-online/smoke-request.json
```

Do not blindly repeat the smoke dataset to claim further improvement. The smoke
run used two documentation examples, two differently worded validation examples,
and two regression examples. A version accepted by this small test is not proof
of general model improvement. Unreviewed chat generations are collected in CHL
but are not automatically treated as training truth. Full-model updates remain
separate from these micro-updates.

The direct backend supports text and tool calls through Chat Completions,
Responses and Anthropic Messages. Its SSE implementation transfers visible text incrementally during generation,
with stable response IDs and completion/usage events for each protocol. Structured
tool calls are parsed before their arguments are emitted. The native default output
budget uses the remaining model context (E2B: 131072 tokens shared by input and
output); explicit client limits
(`max_output_tokens`, `max_completion_tokens`, `max_tokens`) take precedence.
Runtime status includes the applied limit, its source and the finish reason. Disconnects cancel
generation; weight updates wait until the active generation has stopped. Multimodal inputs are rejected explicitly. Agent protocol
formatting is tested; successful model tool use still depends on model quality.

External experts can grow without changing the active model:

```sh
onram micro grow --extension models/athenea-e2b.extensions.nram \
  --output models/athenea-e2b-grown.extensions.nram
```

Growth preserves existing outputs and leaves the new branch inactive. It must
be separately trained and evaluated before being enabled. Candidate creation
does not automatically replace the active bank.

### Explicit world memory

`models/athenea.world.sqlite` stores subject/relation/object edges, positive or
negative polarity, facts or dated events, provenance, and explicit verification.
Opposite verified claims remain visible as unresolved contradictions and are
excluded from supervised export until resolved. This symbolic graph is separate
from the micro-expert weight map; no claim is made that a fact has been located
in a specific neuron. Relevant verified records augment CHL context within the
same character budget.

```sh
onram world-memory add --subject Atlas --relation usa --object PostgreSQL \
  --source 'decisión revisada del usuario' --verified
onram world-memory recall 'Atlas almacenamiento'
onram world-memory export --output data/reviewed-world.jsonl
```

Generated or inferred statements should remain unverified until supported by
an appropriate source. Exported world records still need an independent
validation dataset before training.


The native Atenhea agent launchers share one local server and checkpoint:

```sh
atenhea launch pi
atenhea launch claude
atenhea launch codex
```

Before launching against the default local address, Atenhea verifies the active
backend is `nram` and that the model alias and bundle match `models/onram-launch.json`.
A different backend is rejected. The server loads the original E2B safetensors,
the NRAM adapters and the external micro-extension bank, and retrieves CHL context
from `models/athenea.memory`. No GGUF conversion is involved in this path.

The Codex launcher uses `models/atenhea-agents/codex` as its local `CODEX_HOME`,
with the model catalog in `docs/atenhea-codex-models.json`, so unrelated cloud
plugins and MCP schemas do not consume the E2B context. Standard coding tools
remain available. Hosted web search is disabled for this local profile; the
native runtime does not implement that provider-side service. These scoped
settings follow the [official OpenAI configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference).
Claude Code must be installed separately; its launcher uses the same local server
through the Anthropic Messages protocol. Invoking an agent directly outside
`atenhea launch` uses that agent's own configuration.

Agent launches reuse the resident native server and its loaded weights; they do
not create a model per Pi session. Startup is serialized with
`build/atenhea-startup.lock`, and the public port is reserved before a new server
loads any checkpoint. A persistent inference worker serves successive requests.
`/api/runtime/status` reports the worker PID, load timestamp, current activity and
one weight load per worker lifetime. SSE keep-alive comments also cover waits
between visible text fragments. Closing Pi does not stop the shared model server.

The Metal micro-extension path uses fixed-size masked residuals instead of
per-expert GPU scalar reads and variable-size boolean gathers. Its output is
tested against the original sparse routing; trained extension weights are unchanged.

Brief public explanations can be toggled live:

```sh
atenhea reasoning on
atenhea reasoning off
atenhea reasoning status
```

The current default is on, persisted in `models/athenea-inference.json` and read
for each request. For explanatory questions, the model is instructed to write
`Enfoque:` with one short sentence about its method, then `Respuesta:` with the
answer. Both are ordinary visible text and stream through Pi, Messages and
Responses clients. This is a brief public explanation, not a raw hidden thinking
trace; private parser fields remain excluded. Exact-output requests, structured
formats and tool calls are instructed to omit the preamble. Adherence depends on
the model. Requests can override it with `show_reasoning_summary: false` or
`reasoning: {"summary": null}`. No second generation pass is added.

Native runtime status also reports the `prefill`/`decoding` phase and the current
visible-fragment counter while a request runs. Changing the explanation preference
does not restart the worker or reload weights.
