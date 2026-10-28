"""Pydantic schema layer. `Relation` is the single closed vocabulary (I3): it is
built at import time from config/relations.yaml via the functional Enum API, so
no relation string literal may be constructed anywhere else in the codebase.

`Triple` enforces I2 (no nullable-span path exists) via field validation, not by
convention. I1 (`canonical_text[start:end] == evidence_text`) is enforced where
the canonical text is available, in `store.insert_triple`.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from c2k.config import RELATIONS_PATH


def _load_relation_names(path: Path | str = RELATIONS_PATH) -> list[str]:
    raw = yaml.safe_load(Path(path).read_text())
    return [entry["name"] for entry in raw["relations"]]


def _load_relation_defs(path: Path | str = RELATIONS_PATH) -> dict[str, dict]:
    raw = yaml.safe_load(Path(path).read_text())
    return {entry["name"]: entry for entry in raw["relations"]}


RELATION_DEFS: dict[str, dict] = _load_relation_defs()

# Functional API: Relation.acquired == Relation["acquired"] == Relation("acquired").
Relation = Enum("Relation", {name: name for name in RELATION_DEFS}, type=str)


class ExtractedTriple(BaseModel):
    """What the LLM emits (§6.3). No offsets — the model does not count
    characters; evidence_text is located in post-processing."""

    subject: str
    relation: Relation
    object: str
    evidence_text: str
    confidence: float = Field(ge=0.0, le=1.0)


class Mention(BaseModel):
    mention_id: str
    doc_id: str
    surface: str
    char_start: int
    char_end: int
    entity_id: str | None = None

    @model_validator(mode="after")
    def _span_valid(self) -> "Mention":
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be greater than char_start")
        return self


class Triple(BaseModel):
    """A stored, evidence-grounded fact. Invariant I2: every field required
    for span resolution is mandatory. There is no constructor path that
    produces a Triple with a null or empty span."""

    triple_id: str
    run_id: str
    doc_id: str
    chunk_id: str
    subject_mention_id: str
    subject_entity_id: str | None = None
    relation: Relation
    object_mention_id: str
    object_entity_id: str | None = None
    evidence_start: int
    evidence_end: int
    evidence_text: str
    confidence: float | None = None

    @field_validator("evidence_text")
    @classmethod
    def _evidence_text_nonempty(cls, v: str) -> str:
        if not v:
            raise ValueError("evidence_text must be non-empty (I2)")
        return v

    @model_validator(mode="after")
    def _span_valid(self) -> "Triple":
        if self.evidence_end <= self.evidence_start:
            raise ValueError("evidence_end must be greater than evidence_start (I2)")
        if self.evidence_end - self.evidence_start != len(self.evidence_text):
            raise ValueError(
                "span length must match evidence_text length (I2/I1 precondition)"
            )
        return self


class GoldTriple(BaseModel):
    gold_id: str
    doc_id: str
    subject: str
    relation: Relation | None = None
    object: str
    evidence_start: int | None = None
    evidence_end: int | None = None
    out_of_schema: bool = False
    annotator: str

    @model_validator(mode="after")
    def _relation_xor_out_of_schema(self) -> "GoldTriple":
        if self.out_of_schema and self.relation is not None:
            raise ValueError("out_of_schema gold facts must not carry a relation")
        if not self.out_of_schema and self.relation is None:
            raise ValueError("in-schema gold facts must carry a relation")
        return self
