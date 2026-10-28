"""Span-entailment judging (§9.3). Human judgement is the ground truth; the
LLM judge runs over the same sample and is reported alongside with Cohen's
kappa against the human labels — a finding about substitutability, not the
headline number.
"""

from __future__ import annotations

import random
from pydantic import BaseModel, Field

from c2k import llm
from c2k.store import Store


class EntailmentVerdict(BaseModel):
    entails: bool
    reason: str = Field(default="")


ENTAIL_PROMPT_TEMPLATE = """You are checking whether a piece of evidence text \
supports a claimed fact.

Fact: {subject} --[{relation}]--> {object}

Evidence (verbatim span from the source document):
"{evidence_text}"

Does the evidence entail the fact? Answer strictly from the evidence text; do \
not use outside knowledge."""


def sample_triples_for_judging(
    triples: list[dict], n: int, *, seed: int = 0
) -> list[dict]:
    """Stratified-by-arm sample is the caller's responsibility (one call per
    arm, §9.3); this just draws `n` without replacement, deterministically."""
    rng = random.Random(seed)
    pool = list(triples)
    rng.shuffle(pool)
    return pool[:n]


def judge_human(store: Store, run_id: str, triples: list[dict]) -> list[bool]:
    """Presents each triple + its evidence span in the terminal and records a
    verdict. `triples` are dict rows with subject/relation/object/evidence_text/
    triple_id — the caller (`cli.judge`) supplies them via `load_predicted`
    joined back to raw triple rows."""
    verdicts: list[bool] = []
    for t in triples:
        print("\n" + "-" * 72)
        print(f"  {t['subject']}  --[{t['relation']}]-->  {t['object']}")
        print(f"  evidence: \"{t['evidence_text']}\"")
        answer = input("  Does the evidence entail this fact? [y/n]: ").strip().lower()
        verdict = answer.startswith("y")
        verdicts.append(verdict)
        store.insert_entailment_judgement(t["triple_id"], run_id, "human", verdict)
    return verdicts


def judge_llm(store: Store, run_id: str, triples: list[dict]) -> list[bool]:
    verdicts: list[bool] = []
    for t in triples:
        prompt = ENTAIL_PROMPT_TEMPLATE.format(
            subject=t["subject"],
            relation=t["relation"],
            object=t["object"],
            evidence_text=t["evidence_text"],
        )
        result = llm.call(prompt, EntailmentVerdict, run_id=run_id, purpose="entailment_judge")
        verdicts.append(result.entails)
        store.insert_entailment_judgement(t["triple_id"], run_id, "llm", result.entails)
    return verdicts


def cohens_kappa(a: list[bool], b: list[bool]) -> float | None:
    """Cohen's kappa between two equal-length binary verdict lists."""
    n = len(a)
    if n == 0 or n != len(b):
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    p_a_true = sum(a) / n
    p_b_true = sum(b) / n
    pe = p_a_true * p_b_true + (1 - p_a_true) * (1 - p_b_true)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)
