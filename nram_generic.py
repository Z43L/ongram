"""Architecture-aware LoRA helpers for dense Hugging Face causal LMs.

The base weights stay in their original Transformers checkpoint. A .nram file
stores the adapter, its target/layout map, source identity, and evolution record.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM

from nram_container import load_bundle, save_bundle

LAYER_INDEX = re.compile(r"(?:layers|layer|h|blocks|block)\.(\d+)\.")
PROJECTION_LEAVES = {
    "q_proj", "k_proj", "v_proj", "o_proj", "query", "key", "value",
    "q", "k", "v", "o", "query_key_value", "qkv_proj", "out_proj",
    "proj_in", "proj_out", "dense", "c_attn", "c_proj", "c_fc", "fc1", "fc2", "fc_in", "fc_out", "dense_h_to_4h",
    "dense_4h_to_h", "gate_proj", "up_proj", "down_proj", "w1", "w2",
    "w3", "wi", "wi_0", "wi_1", "wo",
}
EXCLUDED_PARTS = {"lm_head", "embed_tokens", "word_embeddings", "wte", "wpe", "classifier"}


def module_layout(module: nn.Module) -> str | None:
    """Return supported projection storage layout, else None."""
    if isinstance(module, nn.Linear) and module.weight.ndim == 2:
        return "linear_out_in"
    # GPT-2 Transformers Conv1D stores its weight as [input, output]. Avoid
    # importing a private Transformers class and detect its public shape API.
    if module.__class__.__name__ == "Conv1D" and getattr(module, "weight", None) is not None:
        if module.weight.ndim == 2 and hasattr(module, "nf"):
            return "conv1d_in_out"
    return None


def discover_targets(model: nn.Module, *, last_k_blocks: int = 1,
                     explicit: list[str] | None = None) -> dict[str, str]:
    if last_k_blocks < 1:
        raise ValueError("last_k_blocks must be positive")
    candidates: list[tuple[str, str, int | None]] = []
    for name, module in model.named_modules():
        if not name or module_layout(module) is None:
            continue
        parts = set(name.lower().split("."))
        if parts & EXCLUDED_PARTS:
            continue
        layout = module_layout(module)
        candidates.append((name, layout or "", _layer_number(name)))
    if explicit:
        selected = {name: layout for name, layout, _ in candidates
                    if any(name == target or name.endswith("." + target) for target in explicit)}
        missing = [target for target in explicit if not any(name == target or name.endswith("." + target) for name in selected)]
        if missing:
            raise ValueError("Requested adapter target(s) not found among supported dense projections: " + ", ".join(missing))
        return selected

    named = [(name, layout, index) for name, layout, index in candidates
             if name.rsplit(".", 1)[-1].lower() in PROJECTION_LEAVES]
    if not named:
        raise ValueError("No recognized dense attention/MLP projections found; pass --targets explicitly")
    indexed = [index for _, _, index in named if index is not None]
    if indexed:
        cutoff = max(indexed) - last_k_blocks + 1
        named = [(name, layout, index) for name, layout, index in named if index is not None and index >= cutoff]
    elif last_k_blocks == 1:
        # Some architectures do not encode block indices in module names.
        # Keep the final attention/MLP projection group rather than adapting
        # every arbitrary linear layer in the model.
        named = named[-min(8, len(named)):]
    return {name: layout for name, layout, _ in named}


def _layer_number(name: str) -> int | None:
    match = LAYER_INDEX.search(name)
    return int(match.group(1)) if match else None


class LoRAProjection(nn.Module):
    def __init__(self, base: nn.Module, rank: int, alpha: float, layout: str):
        super().__init__()
        self.base = base
        self.layout = layout
        if layout == "linear_out_in":
            out_features, in_features = base.weight.shape
        elif layout == "conv1d_in_out":
            in_features, out_features = base.weight.shape
        else:
            raise ValueError(f"Unsupported weight layout: {layout}")
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, dtype=base.weight.dtype,
                                               device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=base.weight.dtype,
                                               device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        self.scale = float(alpha) / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.linear's weight convention gives the same low-rank residual for
        # nn.Linear [out,in] and GPT-2 Conv1D [in,out].
        residual = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        return self.base(x) + residual


def _parent_and_leaf(root: nn.Module, path: str) -> tuple[nn.Module, str]:
    parent_path, _, leaf = path.rpartition(".")
    return (root.get_submodule(parent_path) if parent_path else root), leaf


def attach_adapters(model: nn.Module, target_layouts: dict[str, str], rank: int,
                    alpha: float) -> dict[str, LoRAProjection]:
    adapters: dict[str, LoRAProjection] = {}
    for name, layout in target_layouts.items():
        parent, leaf = _parent_and_leaf(model, name)
        base = getattr(parent, leaf)
        if module_layout(base) != layout:
            raise ValueError(f"Projection layout changed for {name}: expected {layout}, got {module_layout(base)}")
        wrapped = LoRAProjection(base, rank, alpha, layout)
        setattr(parent, leaf, wrapped)
        adapters[name] = wrapped
    return adapters


def adapter_tensors(adapters: dict[str, LoRAProjection]) -> dict[str, torch.Tensor]:
    return {f"{name}.{part}": getattr(adapter, part).detach().cpu().clone()
            for name, adapter in adapters.items() for part in ("lora_A", "lora_B")}


def restore_adapters(adapters: dict[str, LoRAProjection], tensors: dict[str, torch.Tensor]) -> None:
    expected = {f"{name}.{part}" for name in adapters for part in ("lora_A", "lora_B")}
    if set(tensors) != expected:
        raise ValueError("Adapter tensors do not match the NRAM target graph")
    with torch.no_grad():
        for name, adapter in adapters.items():
            for part in ("lora_A", "lora_B"):
                param = getattr(adapter, part)
                param.copy_(tensors[f"{name}.{part}"].to(device=param.device, dtype=param.dtype))


def merge_adapters(model: nn.Module, adapters: dict[str, LoRAProjection]) -> None:
    with torch.no_grad():
        for name, adapter in adapters.items():
            delta = adapter.lora_B @ adapter.lora_A
            if adapter.layout == "conv1d_in_out":
                delta = delta.T
            adapter.base.weight.add_(delta.to(adapter.base.weight.dtype), alpha=adapter.scale)
            parent, leaf = _parent_and_leaf(model, name)
            setattr(parent, leaf, adapter.base)


def load_base_model(source: str | Path, *, revision: str | None = None, local_files_only: bool = False,
                    trust_remote_code: bool = False) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        str(source), local_files_only=local_files_only, trust_remote_code=trust_remote_code,
        dtype=torch.float32, revision=revision,
    )
    return model


def weight_fingerprint(source: str | Path) -> str | None:
    path = Path(source)
    if not path.is_dir():
        return None
    files = sorted(p for p in path.rglob("*") if p.is_file() and
                   (p.name.endswith(".safetensors") or
                    (p.name.startswith("pytorch_model") and p.suffix == ".bin")))
    if not files:
        return None
    digest = hashlib.sha256()
    for file in files:
        digest.update(str(file.relative_to(path)).encode())
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def load_nram_model(bundle_path: str | Path, source: str | Path | None = None, *,
                    local_files_only: bool = False, trust_remote_code: bool = False):
    manifest, tensors = load_bundle(bundle_path)
    if manifest.get("model_family") != "hf-causal-lm-dense-lora":
        raise ValueError("This is not a generic Hugging Face causal-LM NRAM bundle")
    spec = manifest["adapter"]
    base = manifest["base_model"]
    source = str(source or base["source"])
    model = load_base_model(source, revision=base.get("revision"), local_files_only=local_files_only,
                            trust_remote_code=trust_remote_code)
    expected_hash = base.get("weights_sha256")
    actual_hash = weight_fingerprint(source)
    if expected_hash and actual_hash and expected_hash != actual_hash:
        raise ValueError("Base weight fingerprint does not match this NRAM bundle")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adapters = attach_adapters(model, spec["target_layouts"], spec["rank"], spec["alpha"])
    restore_adapters(adapters, tensors)
    return manifest, model, adapters


def save_accepted_bundle(path: str | Path, manifest: dict[str, Any],
                         adapters: dict[str, LoRAProjection]) -> None:
    save_bundle(path, manifest, adapter_tensors(adapters))
