# CPU-only LoRA fine-tuning and scaling-law study (rekam medis)

Small, reproducible experiments on how model size, data repetition, learning rate, weight decay and
parameter-efficient fine-tuning behave when training on a laptop CPU (designed for a ThinkPad T470s:
i5-6300U 2C/4T, 8 GB RAM, no usable GPU). Inspired by the "Quadrillion-Parameter Transformers" idea, but it
does **not** test or validate any quadrillion-parameter extrapolation.

This repository root is the project root (the `cpu-llm-research/` tree from the brief).
The walkthrough notebook is `notebooks/rekam_medis_cpu_llm.ipynb`.

## Data and privacy
- Default corpus is **synthetic** Indonesian medical-record text (`data.source: synthetic`). No real patients.
- Real records: put `.txt` (blank-line separated), `.jsonl` (`{"text": ...}`) or `.csv` (`text` column) in `data/raw/`
  and set `data.source: data/raw`. `data/raw/` is git-ignored. A best-effort scrubber removes emails, 16-digit IDs and
  mobile numbers, but it does **not** detect names or addresses, so de-identify real data properly first.
- W&B receives numeric metrics and hyper-parameters only (credential-like keys are stripped). Never dataset text.

## Setup (Windows PowerShell, isolated environment)
```powershell
py -3.12 -m venv .venv            # 3.11/3.12 recommended; 3.13 can lag on some wheels
.\.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
Do not install into the global Python. For W&B: `wandb login` (online) or nothing (auto falls back to offline;
sync later with `wandb sync experiments\wandb\offline-run-*`).

## Phases
```powershell
python -m src.hardware_monitor                                   # 1  diagnostics -> reports/hardware.json
python -m src.model_loader --check-deps                          # 2  dependency validation
python -m src.model_loader --models distilgpt2 EleutherAI/pythia-70m --download --smoke-test   # 2  verify, download, fwd/bwd test
python -m src.dataset_processor                                  # 3  data + tokenization summary
python -m src.scaling_experiments --pilot                        # 4/5 pilot (includes interrupt + resume check)
python -m src.scaling_experiments --estimate                     # wall-clock estimate from the pilot speed
python -m src.scaling_experiments --experiments A B C D E        # 7  full study (needs a passed pilot)
python -m src.evaluation                                         # 8  figures + reports/report.md
python -m pytest tests -q                                        # offline tests (no downloads)
```
`--wandb-mode offline|online|disabled` overrides the config. `--seeds 42 43 44` repeats runs for error bars.

## Interrupt and resume
Press **Ctrl-C once**: the current step finishes, a checkpoint is saved, and the process exits. Run the same command
again to resume (completed runs are skipped). Checkpoints (adapter, optimizer, scheduler state, RNG, history) are
written every `checkpoint_every_steps` to `checkpoints/<run>/`. A changed config invalidates the checkpoint.

## Resource safety
Preflight requires `min_available_ram_gb` and `min_free_disk_gb`. During training, available system RAM and process RSS
are checked before every micro-batch; if unsafe the run saves a checkpoint and stops with `stopped_memory_pressure`.
Defaults: batch 1, grad-accum 4, seq len 128, 2 threads, FP32. Models above 100M parameters are rejected (e.g. `gpt2`).
If RAM is tight, close the browser/IDE first; lower `max_seq_len` or `lora.r` before touching anything else.

## Experiments
| ID | Varies | Values (configs/training.yaml) |
|---|---|---|
| A | model | pythia-14m / 31m / 70m, distilgpt2 |
| B | weight decay | 0.01, 0.05, 0.1 |
| C | train-set repetition (fixed 1 epoch) | 1x, 2x, 4x |
| D | learning rate | 5e-5, 1e-4, 2e-4 |
| E | LoRA rank (alpha = 2r) | 4, 8, 16 (also reports trainable % of base) |

The shared baseline (distilgpt2, lr 1e-4, wd 0.01, 1x, r=8) is trained once and used by all five sweeps: 12 unique runs.

## Limitations
- Synthetic, templated text: validation loss is optimistic and says little about real clinical notes.
- A compares different pretrained models, not a controlled size sweep; use bits/char, not raw loss, across tokenizers.
- C increases optimizer steps along with repetition. One seed by default: small gaps are noise.
- Pythia/GPT-2 were pretrained on English; Indonesian medical text is out-of-distribution for them.
- Timings depend on background load. Run the pilot, then `--estimate`, before committing to the full study.
