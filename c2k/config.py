"""Config loading. A single frozen Config object is the source of truth for
every run; `Config.model_dump_json()` is what gets written to `runs.config_json`
so runs are reproducible from that column plus the LLM cache (invariant I6)."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    name: str
    temperature: float = 0.0


class ChunkingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    tokens: int = 900
    overlap: int = 150
    snap_window: int = 50


class ExtractionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    constrained: bool = True
    max_retries: int = 3
    min_confidence: float = 0.0


class ResolutionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    top_k: int = 20
    tau_lo: float = 0.72
    tau_hi: float = 0.88
    adjudication_batch: int = 20


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    entity_match_threshold: float = 0.85
    span_iou_threshold: float = 0.5
    repeats: int = 3
    judge_sample: int = 100


class PathsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    pdfs: Path = Path("data/pdfs")
    gold: Path = Path("data/gold")
    db: Path = Path("data/db/c2k.sqlite")


class Config(BaseModel):
    model_config = ConfigDict(frozen=True)

    arm: str
    model: ModelConfig
    chunking: ChunkingConfig
    extraction: ExtractionConfig
    resolution: ResolutionConfig
    evaluation: EvaluationConfig
    paths: PathsConfig


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
RELATIONS_PATH = Path(__file__).resolve().parent.parent / "config" / "relations.yaml"
RELATION_PHRASES_PATH = Path(__file__).resolve().parent.parent / "config" / "relation_phrases.yaml"
SPLITS_PATH = Path(__file__).resolve().parent.parent / "config" / "splits.yaml"


def load_config(path: Path | str | None = None) -> Config:
    """`C2K_CONFIG` lets an operator point the CLI at an alternate config
    file (e.g. an isolated scratch environment for a demo run) without
    adding an undocumented CLI flag beyond §11's fixed command surface."""
    resolved = path or os.environ.get("C2K_CONFIG") or DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(Path(resolved).read_text())
    return Config.model_validate(raw)


ARM_TABLE: dict[str, dict] = {
    # §8: arm is a config value, never a code branch. a0-a2 keep whatever
    # model config.yaml already names (the "claude" default); a3 is the only
    # arm that switches provider/model, to isolate capability vs. constraint.
    "a0": {"constrained": False, "resolution_enabled": False},
    "a1": {"constrained": True, "resolution_enabled": False},
    "a2": {"constrained": True, "resolution_enabled": True},
    "a3": {"constrained": True, "resolution_enabled": True, "provider": "ollama", "name": "qwen2.5:7b"},
}


def apply_arm(config: Config, arm: str) -> Config:
    if arm not in ARM_TABLE:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARM_TABLE)}")
    overrides = ARM_TABLE[arm]
    model = config.model
    if "provider" in overrides or "name" in overrides:
        model = model.model_copy(
            update={
                "provider": overrides.get("provider", model.provider),
                "name": overrides.get("name", model.name),
            }
        )
    extraction = config.extraction.model_copy(update={"constrained": overrides["constrained"]})
    resolution = config.resolution.model_copy(update={"enabled": overrides["resolution_enabled"]})
    return config.model_copy(update={"arm": arm, "model": model, "extraction": extraction, "resolution": resolution})


def load_splits(path: Path | str = SPLITS_PATH) -> dict[str, str]:
    """doc_id -> 'tune' | 'test', inverted from the checked-in tune/test lists."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    out: dict[str, str] = {}
    for doc_id in raw.get("tune", []) or []:
        out[doc_id] = "tune"
    for doc_id in raw.get("test", []) or []:
        out[doc_id] = "test"
    return out
