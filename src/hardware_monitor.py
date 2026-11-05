"""Phase 1: hardware diagnostics, preflight checks, runtime memory guard and resource sampling.

CLI:  python -m src.hardware_monitor
"""
from __future__ import annotations

import argparse
import platform
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

from .common import ROOT, write_json

GB = 1024 ** 3
MB = 1024 ** 2

# Hardware the project is designed for (informational comparison only).
TARGET = {"cpu": "Intel Core i5-6300U", "physical_cores": 2, "logical_cores": 4, "ram_gb": 8}


def cpu_model() -> str:
    try:
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        )
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except Exception:
        return platform.processor() or "unknown"


def system_snapshot(path: str | Path = ".") -> dict:
    vm = psutil.virtual_memory()
    du = shutil.disk_usage(str(path))
    freq = psutil.cpu_freq()
    snap = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(),
        "cpu_model": cpu_model(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "cpu_freq_mhz": round(freq.current) if freq else None,
        "ram_total_gb": round(vm.total / GB, 2),
        "ram_available_gb": round(vm.available / GB, 2),
        "ram_percent_used": vm.percent,
        "disk_free_gb": round(du.free / GB, 2),
        "disk_total_gb": round(du.total / GB, 2),
        "python": platform.python_version(),
        "process_rss_mb": round(psutil.Process().memory_info().rss / MB, 1),
    }
    try:
        import torch

        snap.update(
            torch_version=torch.__version__,
            torch_threads=torch.get_num_threads(),
            cuda_available=torch.cuda.is_available(),
            gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        )
    except ImportError:
        snap.update(torch_version=None, torch_threads=None, cuda_available=None)
    return snap


def preflight(min_available_ram_gb: float, min_free_disk_gb: float,
              path: str | Path = ".", num_threads: int | None = None, device: str = "cpu"):
    """Return (ok, problems, snapshot). Problems are blocking; warnings are in snapshot['warnings']."""
    snap = system_snapshot(path)
    problems, warnings = [], []
    if snap["ram_available_gb"] < min_available_ram_gb:
        problems.append(
            f"available RAM {snap['ram_available_gb']} GB < required {min_available_ram_gb} GB "
            "(close other applications)"
        )
    if snap["disk_free_gb"] < min_free_disk_gb:
        problems.append(f"free disk {snap['disk_free_gb']} GB < required {min_free_disk_gb} GB")
    if num_threads and num_threads > snap["logical_cores"]:
        warnings.append(f"num_threads={num_threads} exceeds logical cores={snap['logical_cores']}")
    if snap.get("cuda_available") and device == "cpu":
        warnings.append("CUDA is visible but hardware.device=cpu, so the GPU is ignored")
    snap["warnings"] = warnings
    return (not problems), problems, snap


class MemoryGuard:
    """Runtime check used inside the training loop to stop gracefully under memory pressure."""

    def __init__(self, min_available_gb: float, max_process_gb: float):
        self.min_available_gb = min_available_gb
        self.max_process_gb = max_process_gb
        self._proc = psutil.Process()

    def check(self):
        avail = psutil.virtual_memory().available / GB
        rss = self._proc.memory_info().rss / GB
        if avail < self.min_available_gb:
            return False, f"system available RAM {avail:.2f} GB < {self.min_available_gb} GB"
        if rss > self.max_process_gb:
            return False, f"process RSS {rss:.2f} GB > {self.max_process_gb} GB"
        return True, ""


class ResourceSampler:
    """Background thread sampling CPU and RAM. Use as a context manager or start()/stop()."""

    def __init__(self, interval_s: float = 1.0):
        self.interval_s = interval_s
        self.samples: list[dict] = []
        self._proc = psutil.Process()
        self._ncpu = psutil.cpu_count(logical=True) or 1
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    def _sample(self) -> dict:
        vm = psutil.virtual_memory()
        s = {
            "t": round(time.time() - self._t0, 2),
            "cpu_percent": psutil.cpu_percent(None),
            "proc_cpu_percent": round(self._proc.cpu_percent(None) / self._ncpu, 1),
            "rss_mb": round(self._proc.memory_info().rss / MB, 1),
            "ram_available_gb": round(vm.available / GB, 3),
        }
        self.samples.append(s)
        return s

    def _run(self):
        while not self._stop.wait(self.interval_s):
            self._sample()

    def start(self):
        psutil.cpu_percent(None)
        self._proc.cpu_percent(None)
        self._t0 = time.time()
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._sample()

    __enter__ = start

    def __exit__(self, *exc):
        self.stop()

    def latest(self) -> dict:
        return self.samples[-1] if self.samples else {}

    def peak_rss_mb(self) -> float:
        peak = max((s["rss_mb"] for s in self.samples), default=0.0)
        try:  # Windows reports the true OS-level peak working set
            peak = max(peak, self._proc.memory_info().peak_wset / MB)
        except AttributeError:
            pass
        return round(peak, 1)

    def summary(self) -> dict:
        if not self.samples:
            return {}
        cpu = [s["cpu_percent"] for s in self.samples[1:]] or [0.0]
        return {
            "cpu_percent_mean": round(sum(cpu) / len(cpu), 1),
            "cpu_percent_max": max(cpu),
            "peak_rss_mb": self.peak_rss_mb(),
            "min_ram_available_gb": min(s["ram_available_gb"] for s in self.samples),
            "n_samples": len(self.samples),
        }

    def timeline(self, max_points: int = 2000) -> list[dict]:
        step = max(1, len(self.samples) // max_points)
        return self.samples[::step]


def diagnose(path: str | Path = ROOT, save: bool = True) -> dict:
    snap = system_snapshot(path)
    notes = []
    if snap["ram_total_gb"] < TARGET["ram_gb"] - 0.5:
        notes.append(f"RAM {snap['ram_total_gb']} GB is below the {TARGET['ram_gb']} GB design target")
    if snap["logical_cores"] != TARGET["logical_cores"]:
        notes.append(
            f"{snap['logical_cores']} logical cores (design target {TARGET['logical_cores']}); "
            "results/timings are not directly comparable with the ThinkPad T470s"
        )
    if "6300U" not in snap["cpu_model"]:
        notes.append(f"CPU is '{snap['cpu_model']}', not the {TARGET['cpu']} target")
    snap["notes"] = notes
    if save:
        write_json(Path(ROOT) / "reports" / "hardware.json", snap)
    return snap


def main():
    argparse.ArgumentParser(description="Phase 1: hardware and environment diagnostics").parse_args()
    snap = diagnose()
    for k, v in snap.items():
        if k != "notes":
            print(f"{k:>20}: {v}")
    for n in snap["notes"]:
        print("NOTE:", n)
    print("\nSaved reports/hardware.json")


if __name__ == "__main__":
    main()
