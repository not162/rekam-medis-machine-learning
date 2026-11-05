"""Phase 3: dataset preparation, reproducible splitting, tokenization and block packing.

The default corpus is SYNTHETIC Indonesian medical-record text (no real patients). Real records can be
placed under data/raw/ (.txt blank-line separated, .jsonl with "text", or .csv with "text" column);
they stay local (git-ignored) and are never sent to W&B.

CLI: python -m src.dataset_processor [--config configs/training.yaml]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from pathlib import Path

import torch

from .common import ROOT, load_config, write_json

CONDITIONS = [
    {"dx": "Hipertensi esensial", "icd": "I10", "age": (40, 80), "sys": (145, 180), "dia": (90, 110), "temp": (36.2, 36.9),
     "symptoms": ["sakit kepala", "tengkuk terasa berat", "pusing", "mudah lelah"],
     "drugs": ["Amlodipin 5 mg 1x1", "Captopril 25 mg 2x1", "Candesartan 8 mg 1x1"],
     "plan": ["kurangi garam", "kontrol tekanan darah 2 minggu", "olahraga ringan teratur"]},
    {"dx": "Diabetes melitus tipe 2", "icd": "E11.9", "age": (35, 80), "sys": (120, 150), "dia": (75, 95), "temp": (36.2, 36.9),
     "symptoms": ["sering haus", "sering buang air kecil", "lemas", "penurunan berat badan", "kesemutan"],
     "drugs": ["Metformin 500 mg 2x1", "Glimepirid 2 mg 1x1", "Metformin 850 mg 2x1"],
     "plan": ["diet rendah gula", "cek gula darah puasa", "kontrol HbA1c 3 bulan"]},
    {"dx": "Infeksi saluran pernapasan akut", "icd": "J06.9", "age": (1, 70), "sys": (100, 125), "dia": (65, 85), "temp": (37.5, 38.8),
     "symptoms": ["batuk", "pilek", "nyeri tenggorokan", "demam", "hidung tersumbat"],
     "drugs": ["Paracetamol 500 mg 3x1", "Ambroksol 30 mg 3x1", "Cetirizin 10 mg 1x1"],
     "plan": ["perbanyak minum air putih", "istirahat cukup", "kontrol bila demam > 3 hari"]},
    {"dx": "Demam tifoid", "icd": "A01.0", "age": (5, 55), "sys": (100, 120), "dia": (60, 80), "temp": (38.2, 39.8),
     "symptoms": ["demam naik turun", "mual", "nyeri perut", "lidah kotor", "tidak nafsu makan"],
     "drugs": ["Kloramfenikol 500 mg 4x1", "Ciprofloxacin 500 mg 2x1", "Paracetamol 500 mg 3x1"],
     "plan": ["tirah baring", "diet lunak", "cek darah lengkap ulang"]},
    {"dx": "Gastritis", "icd": "K29.7", "age": (18, 65), "sys": (105, 130), "dia": (65, 85), "temp": (36.3, 37.0),
     "symptoms": ["nyeri ulu hati", "mual", "kembung", "sering sendawa"],
     "drugs": ["Omeprazol 20 mg 1x1", "Antasida sirup 3x1", "Sukralfat 500 mg 3x1"],
     "plan": ["makan teratur", "hindari makanan pedas dan kopi", "kontrol 1 minggu"]},
    {"dx": "Demam berdarah dengue", "icd": "A91", "age": (3, 45), "sys": (95, 115), "dia": (55, 75), "temp": (38.5, 40.0),
     "symptoms": ["demam tinggi mendadak", "nyeri otot", "nyeri belakang mata", "bintik merah di kulit"],
     "drugs": ["Paracetamol 500 mg 3x1", "Cairan oral 2-3 liter/hari"],
     "plan": ["pantau trombosit setiap hari", "rawat inap bila trombosit < 100.000", "waspada tanda syok"]},
    {"dx": "Asma bronkial", "icd": "J45.9", "age": (5, 60), "sys": (105, 130), "dia": (65, 85), "temp": (36.3, 37.2),
     "symptoms": ["sesak napas", "mengi", "batuk malam hari", "dada terasa berat"],
     "drugs": ["Salbutamol inhaler 2 puff bila perlu", "Budesonid inhaler 2x1", "Salbutamol 2 mg 3x1"],
     "plan": ["hindari pencetus debu", "peak flow meter", "kontrol 2 minggu"]},
    {"dx": "Diare akut", "icd": "A09", "age": (1, 60), "sys": (95, 120), "dia": (60, 80), "temp": (36.8, 38.0),
     "symptoms": ["buang air besar cair", "mulas", "mual", "muntah"],
     "drugs": ["Oralit setiap diare", "Zinc 20 mg 1x1 selama 10 hari", "Probiotik 2x1"],
     "plan": ["rehidrasi oral", "jaga kebersihan makanan", "kontrol bila ada darah pada tinja"]},
    {"dx": "Infeksi saluran kemih", "icd": "N39.0", "age": (18, 75), "sys": (105, 135), "dia": (65, 88), "temp": (36.8, 38.3),
     "symptoms": ["nyeri saat berkemih", "sering berkemih", "nyeri perut bawah", "urin keruh"],
     "drugs": ["Kotrimoksazol 480 mg 2x2", "Ciprofloxacin 500 mg 2x1", "Fenazopiridin 100 mg 3x1"],
     "plan": ["perbanyak minum air putih", "periksa urinalisis", "kontrol 3 hari"]},
    {"dx": "Migrain", "icd": "G43.9", "age": (15, 55), "sys": (105, 135), "dia": (65, 88), "temp": (36.3, 36.9),
     "symptoms": ["sakit kepala sebelah", "mual", "sensitif terhadap cahaya", "berdenyut"],
     "drugs": ["Ibuprofen 400 mg 3x1", "Sumatriptan 50 mg bila perlu", "Parasetamol 500 mg 3x1"],
     "plan": ["catat pemicu migrain", "tidur teratur", "kontrol 1 bulan"]},
    {"dx": "Dermatitis", "icd": "L30.9", "age": (2, 70), "sys": (100, 130), "dia": (65, 85), "temp": (36.3, 37.0),
     "symptoms": ["gatal", "ruam kemerahan", "kulit kering", "bersisik"],
     "drugs": ["Hidrokortison krim 1% 2x1", "Cetirizin 10 mg 1x1", "Emolien 3x1"],
     "plan": ["hindari pemicu alergi", "gunakan pelembap", "kontrol 1 minggu"]},
    {"dx": "Osteoartritis lutut", "icd": "M17.9", "age": (50, 85), "sys": (120, 155), "dia": (75, 95), "temp": (36.2, 36.9),
     "symptoms": ["nyeri lutut", "kaku pagi hari", "nyeri saat naik tangga", "bengkak ringan"],
     "drugs": ["Meloksikam 7,5 mg 1x1", "Glukosamin 500 mg 3x1", "Paracetamol 500 mg 3x1"],
     "plan": ["turunkan berat badan", "fisioterapi", "kontrol 1 bulan"]},
]


def generate_synthetic_records(n: int, seed: int = 42) -> list[str]:
    """Deterministic synthetic medical-record notes (Bahasa Indonesia). Contains no real patient data."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        c = rng.choice(CONDITIONS)
        sym = rng.sample(c["symptoms"], k=rng.randint(2, min(3, len(c["symptoms"]))))
        sex = rng.choice(["Laki-laki", "Perempuan"])
        hr = rng.randint(68, 108)
        out.append(
            "REKAM MEDIS (SINTETIS)\n"
            f"No: RM-{i:05d}\n"
            f"Usia: {rng.randint(*c['age'])} tahun | Jenis kelamin: {sex}\n"
            f"Keluhan utama: {', '.join(sym)} sejak {rng.randint(1, 10)} hari.\n"
            f"Pemeriksaan: TD {rng.randint(*c['sys'])}/{rng.randint(*c['dia'])} mmHg, nadi {hr} x/menit, "
            f"suhu {rng.uniform(*c['temp']):.1f} C.\n"
            f"Diagnosis: {c['dx']} ({c['icd']}).\n"
            f"Terapi: {rng.choice(c['drugs'])}; {rng.choice(c['drugs'])}.\n"
            f"Rencana: {rng.choice(c['plan'])}; {rng.choice(c['plan'])}."
        )
    return out


_PII = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"\b\d{16}\b"), "[NIK]"),
    (re.compile(r"(?<!\d)(?:\+62|62|0)8\d{8,11}\b"), "[TELEPON]"),
]


def scrub_pii(text: str) -> str:
    """Best-effort removal of emails, 16-digit national IDs and Indonesian mobile numbers.
    NOT a de-identification guarantee: names and addresses are not detected."""
    for rx, tag in _PII:
        text = rx.sub(tag, text)
    return text


def load_raw_records(path: str | Path) -> list[str]:
    path = Path(path)
    files = sorted(p for p in path.rglob("*") if p.suffix in {".txt", ".jsonl", ".csv"}) if path.is_dir() else [path]
    records: list[str] = []
    for f in files:
        if f.suffix == ".txt":
            blocks = re.split(r"\n\s*\n", f.read_text(encoding="utf-8"))
            records += [b.strip() for b in blocks if b.strip()]
        elif f.suffix == ".jsonl":
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    records.append(json.loads(line)["text"])
        elif f.suffix == ".csv":
            with open(f, encoding="utf-8", newline="") as fh:
                records += [r["text"] for r in csv.DictReader(fh) if r.get("text")]
    if not records:
        raise ValueError(f"No records found under {path}")
    return records


def split_records(records: list[str], val_fraction: float, seed: int):
    """Reproducible record-level split (done BEFORE repetition/packing, so no train/val leakage)."""
    idx = list(range(len(records)))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(round(len(records) * val_fraction)))
    return [records[i] for i in idx[n_val:]], [records[i] for i in idx[:n_val]]


def tokenize_to_blocks(records: list[str], tokenizer, seq_len: int):
    """Concatenate EOS-terminated records and cut into fixed (seq_len) blocks; remainder dropped."""
    ids = tokenizer(records, add_special_tokens=False)["input_ids"]
    stream: list[int] = []
    for r in ids:
        stream.extend(r)
        stream.append(tokenizer.eos_token_id)
    n_blocks = len(stream) // seq_len
    if n_blocks == 0:
        raise ValueError("Too little data for a single block; increase n_records or lower max_seq_len")
    blocks = torch.tensor(stream[: n_blocks * seq_len], dtype=torch.long).view(n_blocks, seq_len)
    chars = sum(len(r) for r in records)
    return blocks, {
        "n_records": len(records), "n_blocks": n_blocks, "n_tokens": n_blocks * seq_len,
        "dropped_tokens": len(stream) - n_blocks * seq_len,
        "chars_per_token": round(chars / len(stream), 4),
    }


def repeat_blocks(blocks: torch.Tensor, k: int) -> torch.Tensor:
    return blocks.repeat(k, 1) if k > 1 else blocks


def build_datasets(cfg: dict, tokenizer, root: Path = ROOT, use_cache: bool = True) -> dict:
    d = cfg["data"]
    seed = cfg["seed"]
    tok_name = getattr(tokenizer, "name_or_path", "unknown")
    key_src = json.dumps({k: d.get(k) for k in ("source", "n_records", "val_fraction", "max_seq_len", "scrub_pii")}
                         | {"seed": seed, "tok": tok_name}, sort_keys=True)
    key = hashlib.sha1(key_src.encode()).hexdigest()[:12]
    cache = Path(root) / "data" / "processed" / f"{key}.pt"
    if use_cache and cache.exists():
        return torch.load(cache)

    if d["source"] == "synthetic":
        records = generate_synthetic_records(d["n_records"], seed)
        raw = Path(root) / "data" / "raw" / "synthetic_rekam_medis.jsonl"
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text("\n".join(json.dumps({"text": r}, ensure_ascii=False) for r in records), encoding="utf-8")
    else:
        p = Path(d["source"])
        records = load_raw_records(p if p.is_absolute() else Path(root) / p)
    if d.get("scrub_pii", True):
        records = [scrub_pii(r) for r in records]

    train_rec, val_rec = split_records(records, d["val_fraction"], seed)
    train, tstats = tokenize_to_blocks(train_rec, tokenizer, d["max_seq_len"])
    val, vstats = tokenize_to_blocks(val_rec, tokenizer, d["max_seq_len"])
    out = {"train": train, "val": val, "train_stats": tstats, "val_stats": vstats,
           "meta": {"tokenizer": tok_name, "seq_len": d["max_seq_len"], "source": d["source"], "key": key}}
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, cache)
    return out


def main():
    ap = argparse.ArgumentParser(description="Phase 3: prepare and tokenize the dataset")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    from .model_loader import load_model_and_tokenizer  # tokenizer only is needed; reuse the loader's setup

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"], local_files_only=cfg["model"]["local_files_only"])
    data = build_datasets(cfg, tok, use_cache=False)
    summary = {"train": data["train_stats"], "val": data["val_stats"], "meta": data["meta"],
               "repeat": cfg["data"]["repeat"]}
    write_json(ROOT / "reports" / "dataset_summary.json", summary)
    print(json.dumps(summary, indent=2))
    print("\nExample record:\n" + generate_synthetic_records(1, cfg["seed"])[0])


if __name__ == "__main__":
    main()
