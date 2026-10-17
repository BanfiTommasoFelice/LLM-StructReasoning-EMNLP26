"""YAML configuration with dotted command-line overrides."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


class Config(dict):
    """A dict whose keys can also be read as attributes (``cfg.gnn.lr``)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as err:
            raise AttributeError(key) from err

    def __setattr__(self, key, value):
        self[key] = value


def _wrap(obj):
    if isinstance(obj, dict):
        return Config({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _set_dotted(cfg: dict, dotted_key: str, value) -> None:
    *parents, leaf = dotted_key.split(".")
    node = cfg
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = value


def load_config(path: str | Path | None = None, overrides: list[str] | tuple = ()) -> Config:
    """Load a YAML config and apply ``key.subkey=value`` overrides."""
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        key, raw = item.split("=", 1)
        _set_dotted(cfg, key, yaml.safe_load(raw))
    return _wrap(cfg)


def resolve(path: str | Path) -> Path:
    """Resolve a path from the config relative to the repository root."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def run_dir(cfg: Config) -> Path:
    """Folder holding every artifact of one (dataset, model) pair."""
    return resolve(cfg.output_dir) / cfg.dataset / cfg.model


def stage_file(cfg: Config, stage: str, run: int, ext: str = "json") -> Path:
    """Path of the per-run file written by a pipeline stage."""
    return run_dir(cfg) / stage / f"run{run}.{ext}"
