"""Phases 5 and 7: pilot gate and scaling-law experiments A-E.

  python -m src.scaling_experiments --pilot                       # must pass before any full run
  python -m src.scaling_experiments --experiments B --dry-run     # show the run plan
  python -m src.scaling_experiments --estimate                    # wall-clock estimate from the pilot
  python -m src.scaling_experiments --experiments A B C D E       # full study (resumable: re-run to continue)

Runs are de-duplicated: the shared baseline (distilgpt2, lr 1e-4, wd 0.01, 1x data, r=8) is trained once and
tagged with every experiment it belongs to.
"""
from __future__ import annotations

import argparse
import copy
import gc
import math
import shutil
from pathlib import Path

from .common import ROOT, deep_update, load_config, read_json, slug, write_json
from .lora_finetuning import STOP_STATUSES, run_finetuning

EXPERIMENT_ORDER = ["A", "B", "C", "D", "E"]


def make_run_name(model: str, lr: float, wd: float, rep: int, r: int, seed: int) -> str:
    return slug(f"{model.split('/')[-1]}_lr{lr:g}_wd{wd:g}_rep{rep}_r{r}_s{seed}")


def build_plan(cfg: dict, experiments: list[str], seeds: list[int] | None = None) -> dict:
    """Return {run_name: {"overrides": {...}, "experiments": [...]}} with duplicates merged."""
    ex = cfg["experiments"]
    seeds = seeds or ex.get("seeds", [cfg["seed"]])
    base = {"model": cfg["model"]["name"], "lr": cfg["train"]["learning_rate"], "wd": cfg["train"]["weight_decay"],
            "rep": cfg["data"]["repeat"], "r": cfg["lora"]["r"]}
    sweeps = {
        "A": ("model", ex["A_models"]), "B": ("wd", ex["B_weight_decay"]), "C": ("rep", ex["C_repeats"]),
        "D": ("lr", ex["D_learning_rates"]), "E": ("r", ex["E_lora_ranks"]),
    }
    plan: dict = {}
    for seed in seeds:
        for e in [x for x in EXPERIMENT_ORDER if x in experiments]:
            key, values = sweeps[e]
            for v in values:
                p = {**base, key: v}
                name = make_run_name(p["model"], p["lr"], p["wd"], p["rep"], p["r"], seed)
                entry = plan.setdefault(name, {"experiments": [], "overrides": {
                    "seed": seed, "model": {"name": p["model"]},
                    "train": {"learning_rate": p["lr"], "weight_decay": p["wd"]},
                    "data": {"repeat": p["rep"]}, "lora": {"r": p["r"], "alpha": 2 * p["r"]}}})
                entry["experiments"].append(e)
    return dict(sorted(plan.items(), key=lambda kv: (kv[1]["overrides"]["seed"], kv[1]["overrides"]["data"]["repeat"])))


def pilot_path(root: Path) -> Path:
    return Path(root) / "experiments" / "pilot" / "pilot_passed.json"


def pilot_passed(root: Path = ROOT) -> bool:
    return pilot_path(root).exists()


def run_pilot(cfg: dict, root: Path = ROOT, guard: bool = True, **run_kwargs) -> bool:
    """Train a tiny run, interrupt it midway, resume it, and verify the loop is sane."""
    root = Path(root)
    pc = copy.deepcopy(cfg)
    deep_update(pc, {"data": {"n_records": cfg["experiments"]["pilot_records"], "repeat": 1},
                     "train": {"eval_every_steps": 4, "checkpoint_every_steps": 4, "max_eval_blocks": 16,
                               "eval_train_blocks": 8}})
    for d in (root / "experiments" / "pilot", root / "checkpoints" / "pilot", root / "models" / "adapters" / "pilot"):
        shutil.rmtree(d, ignore_errors=True)
    print("[pilot] phase 1: train 8 steps then stop (simulated interruption)")
    r1 = run_finetuning(pc, "pilot", tags=["pilot"], guard=guard, resume=True, stop_after_steps=8, root=root, **run_kwargs)
    if r1["status"] != "interrupted":
        print(f"[pilot] FAILED in phase 1: status={r1['status']} {r1.get('problems') or r1.get('stop_reason') or ''}")
        return False
    print("[pilot] phase 2: resume from checkpoint and finish")
    r2 = run_finetuning(pc, "pilot", tags=["pilot"], guard=guard, resume=True, root=root, **run_kwargs)
    losses = [r2.get("final_train_loss"), r2.get("final_val_loss")]
    checks = {
        "completed": r2["status"] == "completed",
        "resumed_from_step_8": r2.get("resumed_from_step") == 8,
        "finite_losses": all(x is not None and math.isfinite(x) for x in losses),
        "ram_within_budget": (r2.get("peak_rss_mb") or 0) / 1024 <= cfg["hardware"]["max_train_ram_gb"],
        "adapter_saved": bool(r2.get("adapter_dir")) and Path(r2["adapter_dir"]).exists(),
    }
    print("[pilot] checks:", checks)
    if all(checks.values()):
        write_json(pilot_path(root), {"checks": checks, "mean_s_per_step": r2.get("mean_s_per_step"),
                                      "peak_rss_mb": r2.get("peak_rss_mb"), "model": cfg["model"]["name"],
                                      "grad_accum": cfg["train"]["grad_accum"]})
        print(f"[pilot] PASSED  (~{r2.get('mean_s_per_step')} s per optimizer step, peak RSS {r2.get('peak_rss_mb')} MB)")
        return True
    print("[pilot] FAILED")
    return False


def run_plan(cfg: dict, plan: dict, root: Path = ROOT, guard: bool = True) -> list[dict]:
    results = []
    for i, (name, spec) in enumerate(plan.items(), 1):
        print(f"\n===== run {i}/{len(plan)}: {name}  [{','.join(spec['experiments'])}] =====")
        rcfg = deep_update(copy.deepcopy(cfg), copy.deepcopy(spec["overrides"]))
        res = run_finetuning(rcfg, name, tags=spec["experiments"], guard=guard, root=root)
        results.append(res)
        gc.collect()  # one model in memory at a time
        if res["status"] in STOP_STATUSES:
            print(f"[plan] stopping at '{name}' ({res['status']}). Re-run the same command to resume.")
            break
    return results


def estimate(cfg: dict, plan: dict, root: Path = ROOT) -> None:
    if not pilot_passed(root):
        print("Run --pilot first: the estimate uses the measured pilot speed.")
        return
    sps = read_json(pilot_path(root))["mean_s_per_step"]
    from transformers import AutoTokenizer

    from . import dataset_processor as dp

    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"], local_files_only=cfg["model"]["local_files_only"])
    blocks = len(dp.build_datasets(cfg, tok, root)["train"])
    t = cfg["train"]
    total = 0.0
    for name, spec in plan.items():
        rep = spec["overrides"]["data"]["repeat"]
        steps = (blocks * rep * t["epochs"] // t["batch_size"]) // t["grad_accum"]
        sec = steps * sps  # distilgpt2 speed: an upper bound for the smaller Pythia models
        total += sec
        print(f"{name:55s} {steps:5d} steps  ~{sec / 60:6.1f} min")
    print(f"\nTotal ~{total / 3600:.1f} h (training only, excludes eval; rough, based on {sps}s/step from the pilot)")


def main():
    ap = argparse.ArgumentParser(description="Scaling-law experiments (CPU, LoRA)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--experiments", nargs="*", default=None, choices=EXPERIMENT_ORDER)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--skip-pilot-check", action="store_true")
    ap.add_argument("--no-guard", action="store_true", help="disable RAM/disk preflight (not recommended)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="config overrides, e.g. hardware.device=cuda train.batch_size=4 train.grad_accum=1")
    ap.add_argument("--wandb-mode", choices=["auto", "online", "offline", "disabled"], default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.wandb_mode:
        cfg["wandb"]["mode"] = a.wandb_mode
    import yaml
    for kv in a.set:
        k, v = kv.split("=", 1)
        node = cfg
        *path, last = k.split(".")
        for p in path:
            node = node[p]
        node[last] = yaml.safe_load(v)

    if a.pilot:
        raise SystemExit(0 if run_pilot(cfg, guard=not a.no_guard) else 1)

    exps = a.experiments or EXPERIMENT_ORDER
    plan = build_plan(cfg, exps, a.seeds)
    if a.dry_run:
        for n, s in plan.items():
            print(f"{n:55s} {','.join(s['experiments'])}")
        print(f"\n{len(plan)} unique runs")
        return
    if a.estimate:
        estimate(cfg, plan)
        return
    if not a.skip_pilot_check and not pilot_passed():
        raise SystemExit("Pilot has not passed. Run:  python -m src.scaling_experiments --pilot")
    results = run_plan(cfg, plan, guard=not a.no_guard)
    done = sum(r["status"] == "completed" for r in results)
    print(f"\n{done}/{len(plan)} runs completed. Next: python -m src.evaluation")


if __name__ == "__main__":
    main()
