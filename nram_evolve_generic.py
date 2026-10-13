"""Train and safely accept a LoRA update for any compatible HF causal LM."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoTokenizer

from nram_generic import load_nram_model, save_accepted_bundle


def read_jsonl(path: Path) -> list[dict[str, str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No examples found in {path}")
    for index, row in enumerate(rows):
        if not isinstance(row.get("prompt"), str) or not isinstance(row.get("completion"), str):
            raise ValueError(f"Example {index} needs string fields 'prompt' and 'completion'")
        if not row["prompt"].strip() or not row["completion"].strip():
            raise ValueError(f"Example {index} has an empty prompt or completion")
    return rows


def encode(tokenizer: Any, row: dict[str, str], max_length: int, device: torch.device):
    prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(row["completion"], add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        completion_ids = completion_ids + [tokenizer.eos_token_id]
    if not prompt_ids or not completion_ids:
        raise ValueError("Prompt and completion must each tokenize to at least one token")
    ids = (prompt_ids + completion_ids)[:max_length]
    prompt_len = min(len(prompt_ids), len(ids))
    labels = [-100] * prompt_len + ids[prompt_len:]
    if not any(token != -100 for token in labels[1:]):
        raise ValueError("max_length leaves no completion token to score")
    return (torch.tensor([ids], dtype=torch.long, device=device),
            torch.tensor([labels], dtype=torch.long, device=device),
            sum(token != -100 for token in labels[1:]))


@torch.no_grad()
def score(model: nn.Module, tokenizer: Any, rows: list[dict[str, str]], max_length: int,
          device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for row in rows:
        ids, labels, count = encode(tokenizer, row, max_length, device)
        loss = model(input_ids=ids, labels=labels).loss
        total_loss += float(loss) * count
        total_tokens += count
    return total_loss / total_tokens


@torch.no_grad()
def first_token_rank(model: nn.Module, tokenizer: Any, rows: list[dict[str, str]],
                     device: torch.device) -> dict[str, float]:
    model.eval()
    ranks: list[int] = []
    for row in rows:
        ids = tokenizer(row["prompt"], add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        target = tokenizer(row["completion"], add_special_tokens=False)["input_ids"][0]
        logits = model(input_ids=ids).logits[0, -1]
        ranks.append(int((logits > logits[target]).sum()) + 1)
    return {"mean_rank": sum(ranks) / len(ranks),
            "top1_hits": float(sum(rank == 1 for rank in ranks)), "examples": float(len(ranks))}


def train_epoch(model: nn.Module, tokenizer: Any, rows: list[dict[str, str]],
                optimizer: torch.optim.Optimizer, max_length: int, device: torch.device) -> float:
    model.eval()
    random.shuffle(rows)
    total_loss = 0.0
    total_tokens = 0
    for row in rows:
        ids, labels, count = encode(tokenizer, row, max_length, device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, labels=labels).loss
        loss.backward()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        total_loss += float(loss.detach()) * count
        total_tokens += count
    return total_loss / total_tokens


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.threads)
    train_rows = read_jsonl(args.train)
    validation_rows = read_jsonl(args.validation)
    guardrail_rows = read_jsonl(args.guardrail) if args.guardrail else None
    device = torch.device(args.device)

    manifest, model, adapters = load_nram_model(
        args.bundle, source=args.model, local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    base = manifest["base_model"]
    tokenizer_source = args.tokenizer or base.get("tokenizer_source") or base["source"]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, revision=base.get("tokenizer_revision", base.get("revision")),
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(device)
    model.config.use_cache = False

    validation_before = score(model, tokenizer, validation_rows, args.max_length, device)
    rank_before = first_token_rank(model, tokenizer, validation_rows, device)
    guard_before = score(model, tokenizer, guardrail_rows, args.max_length, device) if guardrail_rows else None
    params = [parameter for adapter in adapters.values()
              for parameter in (adapter.lora_A, adapter.lora_B)]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    train_losses = []
    for epoch in range(args.epochs):
        loss = train_epoch(model, tokenizer, list(train_rows), optimizer, args.max_length, device)
        train_losses.append(loss)
        print(f"epoch {epoch + 1}/{args.epochs}: train_nll={loss:.5f}", flush=True)

    validation_after = score(model, tokenizer, validation_rows, args.max_length, device)
    rank_after = first_token_rank(model, tokenizer, validation_rows, device)
    guard_after = score(model, tokenizer, guardrail_rows, args.max_length, device) if guardrail_rows else None
    guard_ok = guard_before is None or guard_after <= guard_before * (1.0 + args.max_guardrail_drift)
    rank_ok = rank_after["mean_rank"] <= rank_before["mean_rank"] - args.min_first_token_rank_improvement
    accepted = validation_after + args.min_improvement < validation_before and guard_ok and rank_ok
    old_version = int(manifest["version"])
    report: dict[str, Any] = {
        "accepted": accepted,
        "version": old_version + 1 if accepted else old_version,
        "model_type": base["model_type"],
        "validation_nll_before": validation_before,
        "validation_nll_after": validation_after,
        "first_token_rank_before": rank_before["mean_rank"],
        "first_token_rank_after": rank_after["mean_rank"],
        "first_token_top1_before": rank_before["top1_hits"],
        "first_token_top1_after": rank_after["top1_hits"],
        "guardrail_nll_before": guard_before,
        "guardrail_nll_after": guard_after,
        "guardrail_ok": guard_ok,
        "trainable_parameters": sum(parameter.numel() for parameter in params),
        "train_nll_by_epoch": train_losses,
    }
    if not accepted:
        return report

    manifest["version"] = report["version"]
    manifest["dataset"] = {
        "train_file": str(args.train), "train_sha256": sha256(args.train),
        "train_examples": len(train_rows), "validation_file": str(args.validation),
        "validation_sha256": sha256(args.validation), "validation_examples": len(validation_rows),
        "guardrail_file": str(args.guardrail) if args.guardrail else None,
        "guardrail_sha256": sha256(args.guardrail) if args.guardrail else None,
    }
    manifest["last_validation"] = report
    manifest.setdefault("journal", []).append({**report, "epochs": args.epochs,
                                                "learning_rate": args.learning_rate})
    save_accepted_bundle(args.bundle, manifest, adapters)
    report["output"] = str(args.bundle)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--guardrail", type=Path)
    parser.add_argument("--model", help="Override the HF base source recorded in the bundle")
    parser.add_argument("--tokenizer", help="Override tokenizer source")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--min-improvement", type=float, default=1e-4)
    parser.add_argument("--min-first-token-rank-improvement", type=float, default=0.0)
    parser.add_argument("--max-guardrail-drift", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.learning_rate <= 0 or args.max_length < 2:
        parser.error("epochs, learning-rate, and max-length must be positive (max-length >= 2)")
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
