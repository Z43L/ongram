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
results in newline-delimited JSON. It is a prototype compatibility layer; it
does not include Ollama's model registry/pull, Modelfile build, embeddings,
vision, tools, cloud, or platform packaging features.

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
