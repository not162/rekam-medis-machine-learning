"""Phase 8: aggregate results, make figures, write the experimental report.

CLI: python -m src.evaluation
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .common import ROOT, load_config, read_json, write_json
from .hardware_monitor import diagnose

PARAM = {"A": ("model", "model"), "B": ("weight_decay", "weight decay"), "C": ("repeat", "data repetition"),
         "D": ("learning_rate", "learning rate"), "E": ("lora_r", "LoRA rank")}
NAMES = {"A": "A: model capacity", "B": "B: weight decay", "C": "C: repeated data", "D": "D: learning rate",
         "E": "E: PEFT (LoRA rank)"}


def load_results(root: Path = ROOT) -> pd.DataFrame:
    rows = []
    for p in sorted((Path(root) / "experiments").glob("*/result.json")):
        r = read_json(p)
        if r.get("status") == "completed" and r["run_name"] != "pilot":
            rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # bits per character: comparable across tokenizers (nats/token / (chars/token * ln 2))
    df["val_bpc"] = df["final_val_loss"] / (df["chars_per_token_val"] * math.log(2))
    return df


def subset(df: pd.DataFrame, exp: str) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["experiments"].apply(lambda t: exp in t)]


def _label(row, exp):
    v = row[PARAM[exp][0]]
    return str(v).split("/")[-1] if exp == "A" else f"{v:g}"


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / name, dpi=110)
    plt.close(fig)


def plot_curves(df, root: Path, out: Path, kind: str):
    exps = [e for e in PARAM if not subset(df, e).empty]
    if not exps:
        return
    fig, axes = plt.subplots(1, len(exps), figsize=(4.2 * len(exps), 3.6), squeeze=False)
    for ax, e in zip(axes[0], exps):
        for _, row in subset(df, e).sort_values(PARAM[e][0]).iterrows():
            m = read_json(Path(root) / "experiments" / row["run_name"] / "metrics.json")
            label = _label(row, e) + (f" s{row['seed']}" if df["seed"].nunique() > 1 else "")
            if kind == "train":
                s = pd.DataFrame(m["train"])
                ax.plot(s["step"], s["loss"].rolling(5, min_periods=1).mean(), label=label)
            else:
                s = pd.DataFrame(m["eval"])
                ax.plot(s["step"], s["val_loss"], marker="o", ms=3, label=label)
        ax.set_title(NAMES[e], fontsize=9)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("train loss (5-step mean)" if kind == "train" else "validation loss")
        ax.legend(fontsize=7, title=PARAM[e][1], title_fontsize=7)
        ax.grid(alpha=.3)
    _save(fig, out, f"{kind}_loss_vs_steps.png")


def plot_sweep(df, exp: str, ycols: dict, out: Path, name: str, logx=False):
    d = subset(df, exp)
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(5, 3.6))
    col = PARAM[exp][0]
    for ycol, lab in ycols.items():
        g = d.groupby(col)[ycol].agg(["mean", "std"]).reset_index().sort_values(col)
        ax.errorbar(g[col], g["mean"], yerr=g["std"].fillna(0) if d["seed"].nunique() > 1 else None,
                    marker="o", capsize=3, label=lab)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(PARAM[exp][1])
    ax.set_ylabel("loss (nats/token)")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    _save(fig, out, name)


def plot_capacity(df, out: Path):
    d = subset(df, "A")
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(5, 3.6))
    d = d.sort_values("params_total")
    ax.plot(d["params_total"] / 1e6, d["val_bpc"], marker="o", label="val bits/char")
    for _, r in d.iterrows():
        ax.annotate(r["model"].split("/")[-1], (r["params_total"] / 1e6, r["val_bpc"]), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("total parameters (M, log)")
    ax.set_ylabel("validation bits per character")
    ax.grid(alpha=.3)
    _save(fig, out, "capacity_vs_val_bpc.png")


def plot_resources(df, root: Path, out: Path):
    if df.empty:
        return
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    for _, row in df.head(12).iterrows():
        p = Path(root) / "experiments" / row["run_name"] / "resources.json"
        if not p.exists():
            continue
        s = pd.DataFrame(read_json(p))
        a1.plot(s["t"], s["cpu_percent"], lw=.8, label=row["run_name"][:28])
        a2.plot(s["t"], s["rss_mb"], lw=.8)
    a1.set_ylabel("system CPU %")
    a2.set_ylabel("process RSS (MB)")
    a2.set_xlabel("seconds since run start")
    a1.legend(fontsize=6)
    a1.grid(alpha=.3)
    a2.grid(alpha=.3)
    _save(fig, out, "cpu_ram_usage.png")


def make_figures(df, root: Path, out: Path):
    plot_curves(df, root, out, "train")
    plot_curves(df, root, out, "val")
    plot_sweep(df, "B", {"final_val_loss": "validation", "final_train_eval_loss": "train (eval mode)"}, out, "weight_decay_vs_val_loss.png")
    plot_sweep(df, "D", {"final_val_loss": "validation", "final_train_eval_loss": "train (eval mode)"}, out, "learning_rate_vs_val_loss.png", logx=True)
    plot_sweep(df, "C", {"final_val_loss": "validation", "final_train_eval_loss": "train (eval mode)"}, out, "repetition_vs_val_loss.png")
    plot_capacity(df, out)
    plot_resources(df, root, out)


def observations(df) -> list[str]:
    obs = []
    multi = not df.empty and df["seed"].nunique() > 1
    noise = "" if multi else " Single seed: differences of this size cannot be separated from run-to-run noise."
    for e in ("B", "D", "E"):
        d = subset(df, e)
        if len(d) < 2:
            continue
        g = d.groupby(PARAM[e][0])["final_val_loss"].mean()
        spread = g.max() - g.min()
        obs.append(f"{NAMES[e]}: lowest mean validation loss at {PARAM[e][1]} = {g.idxmin():g} ({g.min():.4f}); "
                   f"spread across settings {spread:.4f} nats/token.{noise}")
    d = subset(df, "C")
    g = d.groupby("repeat")[["final_train_eval_loss", "final_val_loss"]].mean().sort_index() if len(d) else d
    if len(g) >= 2:
        a, b = g.iloc[0], g.iloc[-1]
        dt, dv = b.final_train_eval_loss - a.final_train_eval_loss, b.final_val_loss - a.final_val_loss
        obs.append(f"C: going from {g.index[0]}x to {g.index[-1]}x repetition changed train(eval-mode) loss by {dt:+.4f} and "
                   f"validation loss by {dv:+.4f}. "
                   + ("Train improved while validation worsened (overfitting signature)." if dt < 0 < dv else
                      "This is not an overfitting signature at this scale." if dv <= 0 else "") + noise)
    d = subset(df, "A")
    if len(d) >= 2:
        s = d.sort_values("params_total")
        obs.append("A: validation bits/char by size: " + ", ".join(
            f"{r.model.split('/')[-1]} ({r.params_total / 1e6:.0f}M) = {r.val_bpc:.3f}" for r in s.itertuples()) +
            ". Models differ in architecture/tokenizer/pre-training data, so this is not a clean parameter-count law.")
    return obs


def _table(df: pd.DataFrame, cols: dict) -> str:
    if df.empty:
        return "_no completed runs_\n"
    lines = ["| " + " | ".join(cols.values()) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:.4g}" if isinstance(v, float) else str(v).split("/")[-1] if c == "model" else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def build_report(root: Path = ROOT, cfg: dict | None = None) -> Path:
    root = Path(root)
    cfg = cfg or load_config()
    df = load_results(root)
    figs = root / "reports" / "figures"
    hw_path = root / "reports" / "hardware.json"
    hw = read_json(hw_path) if hw_path.exists() else diagnose(root, save=False)
    if not df.empty:
        make_figures(df, root, figs)
        df.drop(columns=["experiments"]).assign(experiments=df["experiments"].apply(",".join)).to_csv(
            root / "reports" / "results.csv", index=False)
    deps = {}
    for p in ("torch", "transformers", "peft", "wandb", "datasets", "psutil", "pandas", "matplotlib"):
        try:
            deps[p] = md.version(p)
        except md.PackageNotFoundError:
            deps[p] = "not installed"
    write_json(root / "reports" / "metadata.json", {"hardware": hw, "dependencies": deps, "config": cfg,
                                                    "results": df.to_dict("records") if not df.empty else []})

    fig = lambda n, t: f"![{t}](figures/{n})\n" if (figs / n).exists() else f"_{t}: no data_\n"
    common = {"model": "model", "learning_rate": "lr", "weight_decay": "wd", "repeat": "rep", "lora_r": "r", "seed": "seed",
              "final_train_eval_loss": "train loss (eval)", "final_val_loss": "val loss", "val_bpc": "val bpc",
              "train_duration_s": "time (s)", "peak_rss_mb": "peak RSS (MB)"}
    md_ = [f"# CPU-only LoRA scaling experiment: report\n",
           "> Small-scale, CPU-only measurements. They do **not** validate or extrapolate to quadrillion-parameter "
           "scaling laws; they only show how these knobs behave for sub-100M models under tight resources.\n",
           f"Completed runs: **{len(df)}**\n"]
    md_ += ["## 1. Hardware configuration\n", "| item | value |\n|---|---|"]
    for k in ("cpu_model", "gpu_name", "physical_cores", "logical_cores", "ram_total_gb", "ram_available_gb", "disk_free_gb", "os"):
        md_.append(f"| {k} | {hw.get(k)} |")
    md_ += [f"\nDesign target: Lenovo ThinkPad T470s, i5-6300U (2C/4T), 8 GB RAM, CPU-only. "
            f"Notes: {'; '.join(hw.get('notes', [])) or 'none'}\n", "## 2. Software dependencies\n"]
    md_ += [f"- {k}: {v}" for k, v in deps.items()] + [f"- python: {hw.get('python')}\n"]
    md_ += ["## 3. Model specifications\n",
            _table(df.drop_duplicates("model") if not df.empty else df,
                   {"model": "model", "params_total": "total params", "params_frozen_base": "frozen base",
                    "params_trainable": "trainable (LoRA)", "trainable_pct_of_base": "trainable % of base"}),
            "## 4. Fine-tuning methodology\n",
            "Frozen pretrained causal LM + LoRA adapters on the attention input projection, FP32 on CPU, AdamW with "
            "cosine schedule, gradient clipping 1.0, record-level train/validation split (seeded) performed before "
            "repetition and packing, so repeated data never leaks into validation. Corpus: synthetic Indonesian "
            "medical-record notes unless `data.source` points to local files. Validation loss is token-weighted "
            "cross-entropy; bits/char (bpc) normalises across tokenizers.\n",
            "## 5. Training configuration\n", f"```\n{_cfg_text(cfg)}\n```\n",
            "## 6. Training and validation loss curves\n", fig("train_loss_vs_steps.png", "Training loss vs steps"),
            fig("val_loss_vs_steps.png", "Validation loss vs steps"),
            "## 7. Weight decay comparison (B)\n", fig("weight_decay_vs_val_loss.png", "Weight decay vs loss"),
            _table(subset(df, "B"), common),
            "## 8. Repeated data comparison (C)\n", fig("repetition_vs_val_loss.png", "Repetition vs loss"),
            _table(subset(df, "C"), common),
            "## 9. Learning-rate comparison (D)\n", fig("learning_rate_vs_val_loss.png", "Learning rate vs loss"),
            _table(subset(df, "D"), common),
            "## Model capacity (A) and PEFT (E)\n", fig("capacity_vs_val_bpc.png", "Capacity vs validation bpc"),
            _table(subset(df, "A"), common), _table(subset(df, "E"), {**common, "trainable_pct_of_base": "trainable % of base"}),
            "## 10. CPU and RAM utilisation\n", fig("cpu_ram_usage.png", "CPU and RAM over time"),
            _table(df, {"run_name": "run", "cpu_percent_mean": "mean CPU %", "cpu_percent_max": "max CPU %",
                        "peak_rss_mb": "peak RSS (MB)", "min_ram_available_gb": "min free RAM (GB)"}),
            "## 11. Training time\n", _table(df, {"run_name": "run", "train_duration_s": "total (s)",
                                                  "mean_s_per_step": "s/optimizer step", "steps_completed": "steps",
                                                  "train_tokens_processed": "tokens processed"}),
            "## Observations (auto-generated from the numbers above)\n"]
    md_ += [f"- {o}" for o in observations(df)] or ["- no completed runs yet"]
    md_ += ["\n## 12. Limitations and reproducibility\n",
            "- Single seed unless `--seeds` was given; treat small differences as noise.",
            "- Validation split is drawn from the same synthetic generator, so it is easy and templated; real "
            "records will behave differently and need a clinically meaningful held-out split.",
            "- Experiment C repeats data within a fixed 1 epoch, so more repetitions also means more optimizer steps.",
            "- Experiment A compares different pre-trained models (architecture, tokenizer, pre-training data), "
            "not a controlled parameter-count sweep; bpc corrects only for tokenization.",
            "- Weight decay on LoRA matrices over very short schedules has weak effects; absence of a trend is a valid result.",
            "- Wall-clock figures depend on background load; see the hardware notes above.",
            "- Reproduce: same config, seed 42, same package versions (metadata.json), CPU thread count fixed. "
            "Bitwise-identical results across machines are not guaranteed.\n"]
    out = root / "reports" / "report.md"
    out.write_text("\n".join(md_), encoding="utf-8")
    return out


def _cfg_text(cfg: dict) -> str:
    import yaml

    return yaml.safe_dump({k: cfg[k] for k in ("seed", "hardware", "model", "data", "lora", "train")}, sort_keys=False)


def main():
    ap = argparse.ArgumentParser(description="Phase 8: evaluation and report")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    out = build_report(ROOT, load_config(a.config))
    print(f"Report written to {out}\nAlso: reports/results.csv, reports/metadata.json, reports/figures/")


if __name__ == "__main__":
    main()
