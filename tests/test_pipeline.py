import copy
import math

import pytest
import torch

from src import dataset_processor as dp
from src.common import load_config
from src import model_loader as ml
from src.hardware_monitor import MemoryGuard, ResourceSampler, preflight, system_snapshot
from src.lora_finetuning import run_finetuning
from src.scaling_experiments import build_plan
from src.wandb_logger import NullLogger, resolve_mode, sanitize


# ---- hardware -------------------------------------------------------------------------------
def test_snapshot_and_preflight():
    s = system_snapshot()
    assert s["logical_cores"] >= 1 and s["ram_total_gb"] > 0
    ok, problems, _ = preflight(min_available_ram_gb=10_000, min_free_disk_gb=0)
    assert not ok and problems
    assert preflight(0, 0)[0]


def test_memory_guard_trips():
    assert not MemoryGuard(min_available_gb=10_000, max_process_gb=100).check()[0]
    assert not MemoryGuard(min_available_gb=0, max_process_gb=0.0001).check()[0]
    assert MemoryGuard(0, 1000).check()[0]


def test_sampler_summary():
    with ResourceSampler(0.05) as s:
        import time
        time.sleep(0.2)
    sm = s.summary()
    assert sm["peak_rss_mb"] > 0 and sm["n_samples"] >= 2


# ---- data -----------------------------------------------------------------------------------
def test_synthetic_deterministic_and_split():
    a = dp.generate_synthetic_records(50, 42)
    assert a == dp.generate_synthetic_records(50, 42) and a != dp.generate_synthetic_records(50, 7)
    tr, va = dp.split_records(a, 0.2, 42)
    assert len(va) == 10 and not set(tr) & set(va)
    assert (tr, va) == dp.split_records(a, 0.2, 42)


def test_blocks_and_repeat(tok):
    blocks, stats = dp.tokenize_to_blocks(dp.generate_synthetic_records(20), tok, 64)
    assert blocks.shape[1] == 64 and stats["n_tokens"] == blocks.numel()
    assert dp.repeat_blocks(blocks, 4).shape[0] == 4 * blocks.shape[0]


def test_scrub_pii():
    t = dp.scrub_pii("hubungi budi@mail.com atau 081234567890, NIK 3171234567890001")
    assert "@" not in t and "081234567890" not in t and "3171234567890001" not in t


def test_build_datasets_no_leak(tiny_cfg, tok, tmp_path):
    d = dp.build_datasets(tiny_cfg, tok, tmp_path)
    assert d["train"].shape[1] == 64 and len(d["val"]) >= 1
    assert (tmp_path / "data" / "processed").glob("*.pt")


# ---- model utilities ------------------------------------------------------------------------
def test_lora_param_counts(tiny_model, tiny_cfg):
    base = ml.count_parameters(tiny_model)["total"]
    pm = ml.apply_lora(tiny_model, tiny_cfg)
    c = ml.count_parameters(pm)
    assert 0 < c["trainable"] < base and c["frozen_base"] == base
    assert ml.smoke_test(pm, 257, 64, 1)["ok"]


def test_memory_estimate_and_budget(tiny_model, tiny_cfg):
    est = ml.estimate_memory(82_000_000, 300_000, tiny_model.config, 128, 1)
    assert est["total_gb"] > est["weights_gb"] > 0
    assert not ml.check_budget(124_000_000, 1.0, tiny_cfg)[0]  # gpt2-size rejected by the 100M limit
    assert ml.check_budget(82_000_000, 1.0, tiny_cfg)[0]


# ---- training: completion, interruption, resume -----------------------------------------------
def _run(cfg, tmp, tok, model, **kw):
    return run_finetuning(cfg, "t", model_and_tok=(model, tok), guard=False, root=tmp, **kw)


def test_train_and_resume_matches_uninterrupted(tiny_cfg, tok, tmp_path):
    from transformers import GPT2Config, GPT2LMHeadModel

    def fresh():
        torch.manual_seed(0)
        return GPT2LMHeadModel(GPT2Config(vocab_size=257, n_positions=128, n_embd=32, n_layer=2, n_head=2))

    full = run_finetuning(tiny_cfg, "full", model_and_tok=(fresh(), tok), guard=False, root=tmp_path)
    assert full["status"] == "completed" and math.isfinite(full["final_val_loss"])
    assert full["steps_completed"] == full["total_steps"] > 3

    part = run_finetuning(tiny_cfg, "split", model_and_tok=(fresh(), tok), guard=False, root=tmp_path, stop_after_steps=3)
    assert part["status"] == "interrupted" and part["steps_completed"] == 3
    done = run_finetuning(tiny_cfg, "split", model_and_tok=(fresh(), tok), guard=False, root=tmp_path)
    assert done["status"] == "completed" and done["resumed_from_step"] == 3
    assert done["final_val_loss"] == pytest.approx(full["final_val_loss"], abs=1e-4)

    again = run_finetuning(tiny_cfg, "split", model_and_tok=(fresh(), tok), guard=False, root=tmp_path)
    assert again["status"] == "completed" and again["finished_at"] == done["finished_at"]  # skipped


def test_memory_guard_stops_gracefully(tiny_cfg, tok, tiny_model, tmp_path):
    cfg = copy.deepcopy(tiny_cfg)
    cfg["hardware"]["min_available_ram_gb"] = 0  # preflight passes...
    cfg["hardware"]["max_train_ram_gb"] = 0.0001  # ...but the runtime RSS ceiling trips immediately
    r = run_finetuning(cfg, "mem", model_and_tok=(tiny_model, tok), guard=True, root=tmp_path)
    assert r["status"] == "stopped_memory_pressure" or r["status"] == "rejected"


def test_model_over_budget_rejected(tiny_cfg, tok, tiny_model, tmp_path):
    cfg = copy.deepcopy(tiny_cfg)
    cfg["model"]["max_params_millions"] = 0.001
    r = run_finetuning(cfg, "big", model_and_tok=(tiny_model, tok), guard=False, root=tmp_path)
    assert r["status"] == "rejected"


# ---- experiments plan and logger ---------------------------------------------------------------
def test_plan_dedup():
    tiny_cfg = load_config()  # default config: baseline lr/wd/r are members of the sweeps
    plan = build_plan(tiny_cfg, ["A", "B", "C", "D", "E"])
    assert len(plan) == 12  # 3 + 2 + 2 + 2 + 2 sweep runs + 1 baseline shared by all five
    base = [s for s in plan.values() if len(s["experiments"]) == 5]
    assert len(base) == 1
    assert len(build_plan(tiny_cfg, ["B"], seeds=[1, 2, 3])) == 9


def test_logger_safety(monkeypatch):
    assert "api_key" not in sanitize({"api_key": "x", "lr": 1, "nested": {"WANDB_TOKEN": "y", "ok": 2}})
    assert sanitize({"nested": {"WANDB_TOKEN": "y", "ok": 2}}) == {"nested": {"ok": 2}}
    assert resolve_mode("disabled") == "disabled"
    NullLogger().log({"a": 1}, step=1)
