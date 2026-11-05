"""Phase 6: Weights & Biases wrapper with offline fallback and a no-op mode.

Only numeric metrics and a sanitised hyper-parameter config are logged. Dataset text, raw records,
and anything whose key looks like a credential are never sent.
"""
from __future__ import annotations

import os
import re
import socket
from pathlib import Path

from .common import ROOT

try:
    import wandb  # type: ignore
except ImportError:  # wandb is optional; training still works with local logs only
    wandb = None

_SENSITIVE = re.compile(r"(key|token|secret|password|credential)", re.I)


def sanitize(obj):
    """Recursively drop credential-looking keys from a config."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items() if not _SENSITIVE.search(str(k))}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    return obj


def _has_credentials() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    for name in (".netrc", "_netrc"):
        p = Path.home() / name
        if p.exists() and "api.wandb.ai" in p.read_text(errors="ignore"):
            return True
    return False


def _network_up(host="api.wandb.ai", port=443, timeout=2.0) -> bool:
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


def resolve_mode(requested: str = "auto") -> str:
    if wandb is None or requested == "disabled":
        return "disabled"
    if requested in ("online", "offline"):
        return requested
    return "online" if (_has_credentials() and _network_up()) else "offline"


class NullLogger:
    mode = "disabled"

    def log(self, metrics: dict, step: int | None = None): pass
    def summary(self, metrics: dict): pass
    def finish(self): pass


class WandbLogger:
    def __init__(self, run_name: str, config: dict, project: str, entity: str | None = None,
                 mode: str = "auto", group: str | None = None, tags: list[str] | None = None,
                 root: Path = ROOT):
        self.mode = resolve_mode(mode)
        self.run = None
        self._broken = False
        if self.mode == "disabled":
            return
        wdir = Path(root) / "experiments" / "wandb"
        wdir.mkdir(parents=True, exist_ok=True)
        kwargs = dict(project=project, entity=entity, name=run_name, id=re.sub(r"[^A-Za-z0-9_-]", "-", run_name)[:64],
                      group=group, tags=tags or [], config=sanitize(config), dir=str(wdir), resume="allow",
                      settings=wandb.Settings(save_code=False, silent=True))
        for mode_try in dict.fromkeys([self.mode, "offline"]):  # online failure falls back to offline
            try:
                self.run = wandb.init(mode=mode_try, **kwargs)
                self.mode = mode_try
                return
            except Exception as e:
                print(f"[wandb] init failed in '{mode_try}' mode: {type(e).__name__}: {str(e)[:120]}")
        self.mode = "disabled"

    def log(self, metrics: dict, step: int | None = None):
        if self.run is None or self._broken:
            return
        try:
            self.run.log(metrics, step=step)
        except Exception as e:
            self._broken = True
            print(f"[wandb] logging disabled after error: {e}")

    def summary(self, metrics: dict):
        if self.run is None or self._broken:
            return
        for k, v in metrics.items():
            self.run.summary[k] = v

    def finish(self):
        if self.run is not None:
            try:
                self.run.finish()
            except Exception:
                pass


def make_logger(cfg: dict, run_name: str, run_config: dict, group: str | None = None,
                tags: list[str] | None = None, root: Path = ROOT):
    w = cfg.get("wandb", {})
    lg = WandbLogger(run_name, run_config, w.get("project", "cpu-llm-research"), w.get("entity"),
                     w.get("mode", "auto"), group, tags, root)
    return NullLogger() if lg.mode == "disabled" else lg
