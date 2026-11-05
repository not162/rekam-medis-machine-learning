"""Phase 5: interruptible, resumable CPU LoRA fine-tuning loop.

One "step" = one optimizer step (= grad_accum micro-batches). Checkpoints are written at optimizer-step
boundaries, so resuming reproduces the exact data order (per-epoch permutations are seeded).
Ctrl-C (first press) saves a checkpoint and exits cleanly; run the same command again to resume.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from . import dataset_processor as dp
from . import model_loader as ml
from .common import ROOT, read_json, set_seed, write_json
from .hardware_monitor import MB, MemoryGuard, ResourceSampler, preflight
from .wandb_logger import make_logger

STOP_STATUSES = {"interrupted", "stopped_memory_pressure", "preflight_failed", "diverged"}


class _StopFlag:
    requested = False
    reason = ""


def _install_sigint(flag: _StopFlag):
    def handler(signum, frame):
        if flag.requested:
            raise KeyboardInterrupt
        flag.requested, flag.reason = True, "interrupted"
        print("\n[train] Ctrl-C: finishing the current step, saving a checkpoint (press again to force quit)")

    try:
        return signal.signal(signal.SIGINT, handler)
    except ValueError:  # not in the main thread
        return None


def config_hash(cfg: dict) -> str:
    keep = {"model": cfg["model"]["name"], "lora": cfg["lora"], "train": cfg["train"],
            "data": {k: cfg["data"].get(k) for k in ("source", "n_records", "val_fraction", "max_seq_len", "repeat")},
            "seed": cfg["seed"]}
    return hashlib.sha1(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:10]


@torch.no_grad()
def evaluate(model, blocks: torch.Tensor, batch_size: int) -> float:
    """Token-weighted mean cross-entropy over ``blocks`` in eval mode."""
    was_training = model.training
    dev = next(model.parameters()).device
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(blocks), batch_size):
        b = blocks[i:i + batch_size].to(dev)
        loss = model(input_ids=b, labels=b).loss
        n = b.shape[0] * (b.shape[1] - 1)
        total += float(loss) * n
        count += n
    if was_training:
        model.train()
    return total / max(count, 1)


def _lr_factor(step: int, total: int, warmup_ratio: float, min_ratio: float) -> float:
    warm = max(1, int(warmup_ratio * total))
    if step < warm:
        return (step + 1) / warm
    progress = (step - warm) / max(1, total - warm)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def _save_checkpoint(model, opt, ckpt_dir: Path, state: dict) -> float:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir / "adapter"))
    tmp = ckpt_dir / "trainer_state.pt.tmp"
    torch.save({**state, "optimizer": opt.state_dict(), "rng": torch.get_rng_state()}, tmp)
    os.replace(tmp, ckpt_dir / "trainer_state.pt")
    return round(sum(f.stat().st_size for f in ckpt_dir.rglob("*") if f.is_file()) / MB, 2)


def run_finetuning(cfg: dict, run_name: str, *, tags: list[str] | None = None, logger=None,
                   resume: bool = True, stop_after_steps: int | None = None, guard: bool = True,
                   model_and_tok=None, data: dict | None = None, root: Path = ROOT) -> dict:
    """Train one configuration. Returns the result dict (also written to experiments/<run>/result.json).

    ``model_and_tok``/``data`` allow injecting a model, tokenizer and pre-tokenized data (used by tests).
    """
    root = Path(root)
    t, hw = cfg["train"], cfg["hardware"]
    exp_dir, ckpt_dir = root / "experiments" / run_name, root / "checkpoints" / run_name
    adapter_dir = root / "models" / "adapters" / run_name
    result_path = exp_dir / "result.json"
    exp_dir.mkdir(parents=True, exist_ok=True)
    cfg_h = config_hash(cfg)

    if resume and result_path.exists():
        prev = read_json(result_path)
        if prev.get("status") == "completed" and prev.get("config_hash") == cfg_h:
            merged = sorted(set(prev.get("experiments", [])) | set(tags or []))
            if merged != prev.get("experiments"):
                prev["experiments"] = merged
                write_json(result_path, prev)
            print(f"[train] {run_name}: already completed, skipping")
            return prev

    result = {"run_name": run_name, "experiments": sorted(tags or []), "config_hash": cfg_h,
              "model": cfg["model"]["name"], "seed": cfg["seed"], "lora_r": cfg["lora"]["r"],
              "learning_rate": t["learning_rate"], "weight_decay": t["weight_decay"],
              "repeat": cfg["data"]["repeat"], "batch_size": t["batch_size"], "grad_accum": t["grad_accum"],
              "seq_len": cfg["data"]["max_seq_len"], "epochs": t["epochs"], "num_threads": hw["num_threads"],
              "started_at": datetime.now(timezone.utc).isoformat(), "status": "running"}

    if guard:
        ok, problems, snap = preflight(hw["min_available_ram_gb"], hw["min_free_disk_gb"], root, hw["num_threads"],
                                     hw.get("device", "cpu"))
        if not ok:
            result.update(status="preflight_failed", problems=problems)
            write_json(result_path, result)
            print(f"[train] preflight failed: {problems}")
            return result

    set_seed(cfg["seed"])
    torch.set_num_threads(hw["num_threads"])
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    sampler = ResourceSampler(hw.get("sample_interval_s", 1.0)).start()
    flag = _StopFlag()
    prev_handler = _install_sigint(flag)
    mem_guard = MemoryGuard(hw["min_available_ram_gb"], hw["max_train_ram_gb"]) if guard else None
    own_logger = logger is None
    try:
        model, tok = model_and_tok or ml.load_model_and_tokenizer(cfg["model"]["name"], cfg["model"]["local_files_only"])
        base_counts = ml.count_parameters(model)
        est = ml.estimate_memory(base_counts["total"], 0, model.config, result["seq_len"], t["batch_size"])
        ok, why = ml.check_budget(base_counts["total"], est["total_gb"], cfg)
        if not ok:
            result.update(status="rejected", reject_reasons=why, estimated_memory=est)
            write_json(result_path, result)
            print(f"[train] {cfg['model']['name']} rejected: {why}")
            return result
        device = ml.pick_device(hw.get("device", "cpu"))
        model = ml.apply_lora(model, cfg).to(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        counts = ml.count_parameters(model)
        data = data or dp.build_datasets(cfg, tok, root)
        train_unique, val = data["train"], data["val"]
        train = dp.repeat_blocks(train_unique, cfg["data"]["repeat"])
        val_eval = val[: t["max_eval_blocks"]]
        train_probe = train_unique[: t["eval_train_blocks"]]
        L = result["seq_len"]

        bs, accum = t["batch_size"], t["grad_accum"]
        n = len(train)
        mpe = n // bs  # micro-batches per epoch
        total_steps = (mpe * t["epochs"]) // accum
        if t.get("max_steps"):
            total_steps = min(total_steps, int(t["max_steps"]))
        if total_steps < 1:
            raise ValueError("Zero optimizer steps; dataset too small for batch_size*grad_accum")

        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=t["learning_rate"], weight_decay=t["weight_decay"])

        step, elapsed_prev, resumed_from = 0, 0.0, 0
        hist_train, hist_eval, epoch_durations = [], [], []
        state_path = ckpt_dir / "trainer_state.pt"
        if resume and state_path.exists():
            st = torch.load(state_path)
            if st.get("config_hash") == cfg_h:
                from peft import load_peft_weights, set_peft_model_state_dict

                set_peft_model_state_dict(model, load_peft_weights(str(ckpt_dir / "adapter")))
                opt.load_state_dict(st["optimizer"])
                torch.set_rng_state(st["rng"])
                step = resumed_from = st["step"]
                elapsed_prev = st["elapsed_s"]
                hist_train, hist_eval, epoch_durations = st["train"], st["eval"], st["epoch_durations"]
                print(f"[train] {run_name}: resumed from step {step}/{total_steps}")
            else:
                print("[train] checkpoint config differs from current config; starting fresh")
                shutil.rmtree(ckpt_dir, ignore_errors=True)

        logger = logger or make_logger(
            cfg, run_name,
            {"experiment_name": ",".join(tags or ["adhoc"]), "model_name": cfg["model"]["name"],
             "model_params": counts["total"], "frozen_base_params": counts["frozen_base"],
             "trainable_params": counts["trainable"], "trainable_pct_of_base": counts["trainable_pct_of_base"],
             "dataset_records_train": data["train_stats"]["n_records"], "dataset_blocks_train": n,
             "train_tokens_unique": data["train_stats"]["n_tokens"],
             "train_tokens_processed": mpe * bs * L * t["epochs"] if not t.get("max_steps") else total_steps * accum * bs * L,
             "learning_rate": t["learning_rate"], "weight_decay": t["weight_decay"], "batch_size": bs,
             "grad_accum": accum, "seq_len": L, "lora_r": cfg["lora"]["r"], "repeat": cfg["data"]["repeat"],
             "seed": cfg["seed"], "epochs": t["epochs"], "total_steps": total_steps, "config_hash": cfg_h},
            group=",".join(tags or []) or None, tags=tags, root=root)

        perms: dict[int, torch.Tensor] = {}

        def get_batch(g: int) -> torch.Tensor:
            e, j = divmod(g, mpe)
            if e not in perms:
                perms.clear()
                perms[e] = torch.randperm(n, generator=torch.Generator().manual_seed(cfg["seed"] + e))
            return train[perms[e][j * bs:(j + 1) * bs]].to(device)

        def snapshot_state() -> dict:
            return {"step": step, "elapsed_s": elapsed_prev + time.time() - t_start, "config_hash": cfg_h,
                    "train": hist_train, "eval": hist_eval, "epoch_durations": epoch_durations}

        def do_eval():
            v = evaluate(model, val_eval, t["eval_batch_size"])
            tr = evaluate(model, train_probe, t["eval_batch_size"])
            rec = {"step": step, "val_loss": v, "train_eval_loss": tr, "gap": v - tr,
                   "val_ppl": math.exp(min(v, 20))}
            hist_eval.append(rec)
            logger.log({"val/loss": v, "val/ppl": rec["val_ppl"], "train_eval/loss": tr, "gap/val_minus_train": v - tr}, step=step)
            print(f"[eval] step {step}/{total_steps} val={v:.4f} train_eval={tr:.4f}")

        status = "completed"
        t_start = time.time()
        epoch_t0 = t_start
        model.train()
        opt.zero_grad(set_to_none=True)
        stop_reason = ""
        while step < total_steps:
            acc_loss, g = 0.0, step * accum
            interrupted = False
            for _ in range(accum):
                if flag.requested:
                    interrupted = True
                    break
                if mem_guard:
                    safe, why = mem_guard.check()
                    if not safe:
                        flag.requested, flag.reason, stop_reason = True, "stopped_memory_pressure", why
                        interrupted = True
                        break
                b = get_batch(g)
                loss = model(input_ids=b, labels=b).loss
                if not torch.isfinite(loss):
                    flag.requested, flag.reason, stop_reason = True, "diverged", f"non-finite loss at step {step}"
                    interrupted = True
                    break
                (loss / accum).backward()
                acc_loss += loss.item() / accum
                g += 1
            if interrupted:
                opt.zero_grad(set_to_none=True)  # discard partial accumulation; state stays at last full step
                status, stop_reason = flag.reason, stop_reason or "user interrupt"
                break

            lr_now = t["learning_rate"] * _lr_factor(step, total_steps, t["warmup_ratio"], t["min_lr_ratio"])
            for grp in opt.param_groups:
                grp["lr"] = lr_now
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, t["max_grad_norm"]))
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            res = sampler.latest()
            rec = {"step": step, "loss": acc_loss, "lr": lr_now, "grad_norm": gnorm, "tokens_seen": g * bs * L,
                   "elapsed_s": elapsed_prev + time.time() - t_start,
                   "cpu_percent": res.get("cpu_percent"), "rss_mb": res.get("rss_mb")}
            hist_train.append(rec)
            logger.log({"train/loss": acc_loss, "train/lr": lr_now, "train/grad_norm": gnorm,
                        "train/tokens_seen": rec["tokens_seen"], "system/cpu_percent": rec["cpu_percent"],
                        "system/rss_mb": rec["rss_mb"], "system/ram_available_gb": res.get("ram_available_gb")}, step=step)
            if step % 5 == 0 or step == 1:
                print(f"[train] step {step}/{total_steps} loss={acc_loss:.4f} lr={lr_now:.2e} gnorm={gnorm:.2f} "
                      f"rss={rec['rss_mb']}MB")

            if g // mpe > (g - accum) // mpe:  # an epoch boundary was crossed
                epoch_durations.append(round(time.time() - epoch_t0, 2))
                epoch_t0 = time.time()
                logger.log({"epoch/duration_s": epoch_durations[-1], "epoch/index": len(epoch_durations)}, step=step)
            if step % t["eval_every_steps"] == 0 or step == total_steps:
                do_eval()
            if step % t["checkpoint_every_steps"] == 0 and step < total_steps:
                size = _save_checkpoint(model, opt, ckpt_dir, snapshot_state())
                logger.log({"checkpoint/step": step, "checkpoint/size_mb": size}, step=step)
                write_json(exp_dir / "metrics.json", {"train": hist_train, "eval": hist_eval})
            if stop_after_steps is not None and step >= stop_after_steps and step < total_steps:
                status, stop_reason = "interrupted", f"stop_after_steps={stop_after_steps}"
                break

        elapsed = elapsed_prev + time.time() - t_start
        ckpt_size = _save_checkpoint(model, opt, ckpt_dir, snapshot_state())
        if status == "completed":
            if not hist_eval or hist_eval[-1]["step"] != step:
                do_eval()
            adapter_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(adapter_dir))
        logger.log({"checkpoint/step": step, "checkpoint/size_mb": ckpt_size}, step=step)
        sampler.stop()

        last = hist_train[-5:] or [{"loss": float("nan")}]
        fe = hist_eval[-1] if hist_eval else {}
        summ = sampler.summary()
        result.update({
            "status": status, "stop_reason": stop_reason, "resumed_from_step": resumed_from,
            "steps_completed": step, "total_steps": total_steps,
            "params_total": counts["total"], "params_frozen_base": counts["frozen_base"],
            "params_trainable": counts["trainable"], "trainable_pct_of_base": counts["trainable_pct_of_base"],
            "trainable_pct_of_total": counts["trainable_pct_of_total"], "estimated_memory": est,
            "dataset_records_train": data["train_stats"]["n_records"], "dataset_records_val": data["val_stats"]["n_records"],
            "train_blocks_unique": len(train_unique), "train_blocks_total": n, "val_blocks": len(val),
            "train_tokens_unique": data["train_stats"]["n_tokens"], "train_tokens_processed": step * accum * bs * L,
            "chars_per_token_val": data["val_stats"]["chars_per_token"],
            "final_train_loss": sum(r["loss"] for r in last) / len(last),
            "final_train_eval_loss": fe.get("train_eval_loss"), "final_val_loss": fe.get("val_loss"),
            "best_val_loss": min((e["val_loss"] for e in hist_eval), default=None),
            "val_ppl": fe.get("val_ppl"), "gap": fe.get("gap"),
            "train_duration_s": round(elapsed, 2), "mean_s_per_step": round(elapsed / max(step - resumed_from, 1), 3)
            if step > resumed_from else None,
            "epoch_durations_s": epoch_durations, "cpu_percent_mean": summ.get("cpu_percent_mean"),
            "cpu_percent_max": summ.get("cpu_percent_max"), "peak_rss_mb": summ.get("peak_rss_mb"),
            "min_ram_available_gb": summ.get("min_ram_available_gb"), "checkpoint_dir": str(ckpt_dir),
            "checkpoint_size_mb": ckpt_size, "adapter_dir": str(adapter_dir) if status == "completed" else None,
            "device": str(device), "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / MB, 1) if device.type == "cuda" else None,
            "wandb_mode": getattr(logger, "mode", "disabled"), "finished_at": datetime.now(timezone.utc).isoformat(),
        })
        write_json(exp_dir / "metrics.json", {"train": hist_train, "eval": hist_eval})
        write_json(exp_dir / "resources.json", sampler.timeline())
        write_json(result_path, result)
        logger.summary({k: result[k] for k in ("final_val_loss", "final_train_loss", "train_duration_s", "peak_rss_mb",
                                                "cpu_percent_mean", "status") if result.get(k) is not None})
        print(f"[train] {run_name}: {status} step={step}/{total_steps} val={result['final_val_loss']} "
              f"time={elapsed:.0f}s peak_rss={result['peak_rss_mb']}MB")
        return result
    finally:
        sampler.stop() if sampler._thread and sampler._thread.is_alive() else None
        if prev_handler is not None:
            signal.signal(signal.SIGINT, prev_handler)
        if own_logger and logger is not None:
            logger.finish()
        gc.collect()
