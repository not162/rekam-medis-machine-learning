"""Phase 2: dependency validation, model verification/download, memory estimates, LoRA wrapping.

CLI:
  python -m src.model_loader --check-deps
  python -m src.model_loader --models distilgpt2            # metadata + config-only estimate (no weights)
  python -m src.model_loader --models distilgpt2 --download --smoke-test
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata as md
import platform
import sys
import time

import psutil
import torch

from .common import ROOT, load_config, write_json
from .hardware_monitor import GB, MB

# Licences as stated on the Hugging Face model cards (re-checked online by verify_model when reachable).
MODEL_REGISTRY = {
    "distilgpt2": {"license": "apache-2.0"},
    "gpt2": {"license": "mit"},
    "EleutherAI/pythia-14m": {"license": "apache-2.0"},
    "EleutherAI/pythia-31m": {"license": "apache-2.0"},
    "EleutherAI/pythia-70m": {"license": "apache-2.0"},
}

# LoRA targets by architecture: attention input projection only, for a like-for-like comparison.
LORA_TARGETS = {
    "gpt2": {"target_modules": ["c_attn"], "fan_in_fan_out": True},
    "gpt_neox": {"target_modules": ["query_key_value"], "fan_in_fan_out": False},
}

REQUIRED_PACKAGES = ["torch", "transformers", "peft", "psutil", "pyyaml", "pandas", "matplotlib"]
OPTIONAL_PACKAGES = ["wandb", "datasets", "pytest", "huggingface_hub"]


def check_dependencies() -> dict:
    out = {"python": platform.python_version(), "required": {}, "optional": {}, "problems": []}
    for group, pkgs in (("required", REQUIRED_PACKAGES), ("optional", OPTIONAL_PACKAGES)):
        for p in pkgs:
            try:
                out[group][p] = md.version(p)
            except md.PackageNotFoundError:
                out[group][p] = None
                if group == "required":
                    out["problems"].append(f"missing required package: {p}")
    if out["required"].get("torch") and "+cpu" not in out["required"]["torch"] and torch.cuda.is_available():
        out["problems"].append("CUDA build of torch detected; project is CPU-only (harmless, but wasteful)")
    if sys.version_info >= (3, 13):
        out["problems"].append(
            "Python >= 3.13: some wheels (wandb, tokenizers) may lag. If installation fails, create an "
            "isolated Python 3.11/3.12 venv instead of modifying the global environment."
        )
    if out["optional"].get("wandb") is None:
        out["problems"].append("wandb not installed: runs will log locally only (pip install wandb)")
    return out


def verify_model(name: str) -> dict:
    info = {"name": name, "registry_license": MODEL_REGISTRY.get(name, {}).get("license", "unknown"), "hub": None}
    try:
        from huggingface_hub import HfApi

        mi = HfApi().model_info(name, timeout=10)
        info["hub"] = {
            "reachable": True,
            "license": [t.split(":", 1)[1] for t in (mi.tags or []) if t.startswith("license:")],
            "gated": bool(mi.gated),
            "private": bool(mi.private),
        }
    except Exception as e:  # offline, not installed, or missing repo
        info["hub"] = {"reachable": False, "error": f"{type(e).__name__}: {str(e)[:150]}"}
    return info


def pick_device(requested: str = "cpu") -> torch.device:
    """'cpu' (default, laptop), 'cuda' (e.g. Colab T4) or 'auto' (cuda if available). Fails loudly if cuda is requested but absent."""
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("hardware.device=cuda but no CUDA device is available (Colab: Runtime > Change runtime type > T4 GPU)")
    return torch.device(requested)


def load_model_and_tokenizer(name: str, local_files_only: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(name, local_files_only=local_files_only).float()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def check_tokenizer(tok, model, sample="Pasien laki-laki 45 tahun, keluhan demam 3 hari.") -> dict:
    ids = tok(sample, add_special_tokens=False)["input_ids"]
    vocab_model = model.get_input_embeddings().weight.shape[0]
    return {
        "vocab_size_tokenizer": len(tok),
        "vocab_size_model": vocab_model,
        "compatible": len(tok) <= vocab_model and max(ids) < vocab_model,
        "eos_token_id": tok.eos_token_id,
        "sample_tokens": len(ids),
    }


def count_parameters(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    base = total - trainable  # with LoRA, everything not trainable is the frozen base model
    return {
        "total": total,
        "trainable": trainable,
        "frozen_base": base,
        "trainable_pct_of_base": round(100 * trainable / base, 4) if base else None,
        "trainable_pct_of_total": round(100 * trainable / total, 4) if total else None,
    }


def _cfg_dims(config) -> dict:
    g = lambda *names, d=0: next((getattr(config, n) for n in names if hasattr(config, n)), d)
    return {
        "layers": g("n_layer", "num_hidden_layers"),
        "hidden": g("n_embd", "hidden_size"),
        "heads": g("n_head", "num_attention_heads", d=1),
        "vocab": g("vocab_size"),
    }


def estimate_memory(n_params: int, n_trainable: int, config, seq_len: int, batch_size: int,
                    overhead_gb: float = 0.5) -> dict:
    """Rough FP32 training-memory estimate (GB). An upper-ish heuristic, validated by the smoke test."""
    d = _cfg_dims(config)
    weights = 4 * n_params
    optimizer = 12 * n_trainable  # grads + Adam m + v
    act_per_layer = batch_size * seq_len * d["hidden"] * 34 * 2  # ~34 fp32 tensors of (b, s, h)
    attn = batch_size * d["heads"] * seq_len ** 2 * 4 * 3
    activations = d["layers"] * (act_per_layer + attn)
    logits = batch_size * seq_len * d["vocab"] * 4 * 3
    parts = {
        "weights_gb": weights / GB,
        "optimizer_gb": optimizer / GB,
        "activations_gb": activations / GB,
        "logits_gb": logits / GB,
        "runtime_overhead_gb": overhead_gb,
    }
    parts["total_gb"] = sum(parts.values())
    return {k: round(v, 3) for k, v in parts.items()}


def check_budget(n_params: int, est_total_gb: float, cfg: dict) -> tuple[bool, list[str]]:
    reasons = []
    max_m = cfg["model"]["max_params_millions"]
    if n_params > max_m * 1e6:
        reasons.append(f"{n_params / 1e6:.1f}M parameters exceeds the {max_m}M limit")
    if est_total_gb > cfg["hardware"]["max_train_ram_gb"]:
        reasons.append(f"estimated {est_total_gb:.2f} GB exceeds {cfg['hardware']['max_train_ram_gb']} GB budget")
    return (not reasons), reasons


def apply_lora(model, cfg: dict):
    from peft import LoraConfig, get_peft_model

    mt = model.config.model_type
    if mt not in LORA_TARGETS:
        raise ValueError(f"No LoRA target mapping for model_type '{mt}'; add it to LORA_TARGETS")
    t = LORA_TARGETS[mt]
    lc = LoraConfig(
        r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"]["dropout"],
        target_modules=t["target_modules"], fan_in_fan_out=t["fan_in_fan_out"],
        bias="none", task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lc)  # freezes the base model


def smoke_test(model, vocab_size: int, seq_len: int, batch_size: int) -> dict:
    """One forward/backward pass on random tokens through the (LoRA-wrapped) model."""
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=next(model.parameters()).device)
    model.train()
    t0 = time.time()
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    dt = time.time() - t0
    finite = bool(torch.isfinite(out.loss))
    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    model.zero_grad(set_to_none=True)
    return {
        "loss": round(float(out.loss.detach()), 4), "finite": finite, "lora_grads_present": has_grad,
        "fwd_bwd_seconds": round(dt, 3), "rss_delta_mb": round((proc.memory_info().rss - rss0) / MB, 1),
        "ok": finite and has_grad,
    }


def _estimate_from_config_only(name: str, cfg: dict) -> dict:
    """Count parameters on the meta device (no weight download) and estimate memory."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(name, local_files_only=cfg["model"]["local_files_only"])
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(config)
    n = sum(p.numel() for p in m.parameters())
    tr = 2 * cfg["lora"]["r"] * _cfg_dims(config)["hidden"] * 4 * _cfg_dims(config)["layers"]  # rough
    est = estimate_memory(n, tr, config, cfg["data"]["max_seq_len"], cfg["train"]["batch_size"])
    ok, why = check_budget(n, est["total_gb"], cfg)
    return {"params": n, "estimate": est, "within_budget": ok, "reject_reasons": why}


def main():
    ap = argparse.ArgumentParser(description="Phase 2: dependency validation and model checks")
    ap.add_argument("--config", default=None)
    ap.add_argument("--check-deps", action="store_true")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--download", action="store_true", help="download weights and load one model at a time")
    ap.add_argument("--smoke-test", action="store_true", help="fwd/bwd test (implies --download)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    report = {}

    if a.check_deps:
        report["dependencies"] = check_dependencies()
        print(report["dependencies"])
    if a.models is not None:
        names = a.models or cfg["experiments"]["A_models"]
        report["models"] = {}
        for n in names:
            entry = {"verify": verify_model(n)}
            print(f"\n== {n}\n  licence: registry={entry['verify']['registry_license']} hub={entry['verify']['hub']}")
            try:
                entry["config_estimate"] = _estimate_from_config_only(n, cfg)
                print("  config-only estimate:", entry["config_estimate"])
            except Exception as e:
                entry["config_estimate"] = {"error": f"{type(e).__name__}: {e}"[:200]}
                print("  could not fetch config:", entry["config_estimate"]["error"])
            if (a.download or a.smoke_test) and "error" not in entry["config_estimate"]:
                if not entry["config_estimate"]["within_budget"]:
                    print("  REJECTED:", entry["config_estimate"]["reject_reasons"])
                else:
                    model, tok = load_model_and_tokenizer(n, cfg["model"]["local_files_only"])
                    entry["tokenizer"] = check_tokenizer(tok, model)
                    entry["params_base"] = count_parameters(model)
                    pm = apply_lora(model, cfg)
                    entry["params_lora"] = count_parameters(pm)
                    print("  tokenizer:", entry["tokenizer"])
                    print("  params:", entry["params_lora"])
                    if a.smoke_test:
                        entry["smoke_test"] = smoke_test(
                            pm, len(tok), cfg["data"]["max_seq_len"], cfg["train"]["batch_size"])
                        print("  smoke test:", entry["smoke_test"])
                    del pm, model, tok
                    gc.collect()  # never hold two models at once
            report["models"][n] = entry
    if report:
        write_json(ROOT / "reports" / "model_check.json", report)
        print("\nSaved reports/model_check.json")


if __name__ == "__main__":
    main()
