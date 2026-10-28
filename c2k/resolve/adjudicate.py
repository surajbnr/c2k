"""LLM on the deferred band only (§6.4). One `llm.call()` per batch of
deferred pairs, returning `{same: bool, reason: str}` per pair. Batch 20
pairs per call to control cost (`config/default.yaml`'s
`resolution.adjudication_batch`)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from c2k import llm

PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "adjudicate.md"


class PairVerdict(BaseModel):
    same: bool
    reason: str = ""


class BatchVerdicts(BaseModel):
    verdicts: list[PairVerdict] = Field(default_factory=list)


def build_adjudicate_prompt(pairs: list[tuple[str, str]]) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    pairs_block = "\n".join(f'{i}. "{a}" vs "{b}"' for i, (a, b) in enumerate(pairs))
    return template.format(pairs_block=pairs_block)


def adjudicate_batch(pairs: list[tuple[str, str]], *, run_id: str) -> list[bool]:
    if not pairs:
        return []
    prompt = build_adjudicate_prompt(pairs)
    result = llm.call(prompt, BatchVerdicts, run_id=run_id, purpose="adjudicate")
    if len(result.verdicts) != len(pairs):
        raise RuntimeError(
            f"adjudication batch size mismatch: sent {len(pairs)} pairs, got "
            f"{len(result.verdicts)} verdicts back"
        )
    return [v.same for v in result.verdicts]


def adjudicate_pairs(pairs: list[tuple[str, str]], *, run_id: str, batch_size: int = 20) -> list[bool]:
    verdicts: list[bool] = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        try:
            verdicts.extend(adjudicate_batch(batch, run_id=run_id))
        except RuntimeError as exc:
            # adjudicate_batch() raises on a malformed/mis-sized response by
            # design (mirrors llm.call() itself) — correct at the single-call
            # level, but one bad batch must not discard every other batch's
            # already-adjudicated verdicts in a run with hundreds of deferred
            # pairs. Conservative fallback: an unadjudicated pair is treated
            # as "not the same entity" (False) rather than risking a wrong
            # merge from a response that couldn't be parsed.
            print(f"[adjudication_failure] run={run_id} batch_size={len(batch)} error={exc}")
            verdicts.extend([False] * len(batch))
    return verdicts
