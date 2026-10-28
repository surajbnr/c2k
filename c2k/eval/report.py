"""Results table rendering (§9.2, §8 'report mean +- range'). Plain text —
there is no web UI (§2)."""

from __future__ import annotations

from c2k.eval.metrics import RunMetrics

_NA = "n/a"


def _fmt(x: float | None, pct: bool = True) -> str:
    if x is None:
        return _NA
    return f"{x * 100:.1f}%" if pct else f"{x:.3f}"


def render_run(m: RunMetrics) -> str:
    lines = [
        f"run: {m.run_id}",
        f"  n_predicted={m.n_predicted}  n_gold_in_schema={m.n_gold_in_schema}  "
        f"n_gold_total={m.n_gold_total}",
        "",
        "  metric                          loose      span-strict",
        f"  triple precision                {_fmt(m.triple_loose.precision):<10} "
        f"{_fmt(m.triple_span_strict.precision)}",
        f"  triple recall                   {_fmt(m.triple_loose.recall):<10} "
        f"{_fmt(m.triple_span_strict.recall)}",
        f"  triple F1                       {_fmt(m.triple_loose.f1):<10} "
        f"{_fmt(m.triple_span_strict.f1)}",
        "",
        f"  hallucination rate (human)       {_fmt(m.hallucination)}",
        f"  hallucination rate (LLM judge)   {_fmt(m.hallucination_llm)}",
        f"  entailment judge kappa           {_fmt(m.entailment_kappa, pct=False)}",
        f"  ER pairwise F1                  {_fmt(m.er_pairwise.f1) if m.er_pairwise else _NA}",
        f"  schema recall ceiling           {_fmt(m.schema_recall_ceiling)}",
        f"  span localization failure rate  {_fmt(m.span_localization_failure_rate)}",
        f"  adjudication rate               {_fmt(m.adjudication_rate)}",
        "",
        f"  cost: {m.cost.n_calls} calls, {m.cost.n_input_tokens} in / "
        f"{m.cost.n_output_tokens} out tokens, {m.cost.wall_seconds:.1f}s wall "
        f"over {m.cost.n_documents} doc(s)",
        f"        ({m.cost.calls_per_doc:.2f} calls/doc, "
        f"{m.cost.wall_seconds_per_doc:.2f}s/doc)",
    ]
    return "\n".join(lines)


def mean_range(values: list[float | None]) -> tuple[float, float] | None:
    """(mean, range) across repeats, ignoring None entries. §8: 'report mean
    +- range'. Returns None if nothing to aggregate."""
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present), max(present) - min(present)


def render_success_criterion(a0_repeats: list[RunMetrics], a2_repeats: list[RunMetrics]) -> str:
    """§9.4, declared before any results exist: 'A2 beats A0 on triple F1 by
    a margin larger than the 3-repeat range, at a hallucination rate no worse
    than A0.' Retuning against the test split to force a pass is explicitly
    called out as the one thing that would invalidate the project — this
    function reports whichever way it falls, including failure."""
    a0_f1 = mean_range([m.triple_loose.f1 for m in a0_repeats])
    a2_f1 = mean_range([m.triple_loose.f1 for m in a2_repeats])
    a0_hallu = mean_range([m.hallucination for m in a0_repeats])
    a2_hallu = mean_range([m.hallucination for m in a2_repeats])

    if a0_f1 is None or a2_f1 is None:
        return "success criterion (§9.4): n/a — missing triple F1 for a0 or a2"

    a0_mean, a0_range = a0_f1
    a2_mean, a2_range = a2_f1
    margin = a2_mean - a0_mean
    noise_floor = max(a0_range, a2_range)
    f1_passes = margin > noise_floor

    if a0_hallu is None or a2_hallu is None:
        hallu_passes = None
        hallu_str = "n/a (no entailment judgements recorded for a0 and/or a2)"
    else:
        hallu_passes = a2_hallu[0] <= a0_hallu[0]
        hallu_str = f"a2 {_fmt(a2_hallu[0])} vs a0 {_fmt(a0_hallu[0])} -> {'OK' if hallu_passes else 'WORSE'}"

    overall = f1_passes and (hallu_passes is not False)
    verdict = "PASS" if overall else "FAIL (negative result)"
    return (
        f"success criterion (§9.4): {verdict}\n"
        f"  triple F1: a2 {a2_mean * 100:.1f}% vs a0 {a0_mean * 100:.1f}% "
        f"(margin {margin * 100:+.1f}pp, noise floor {noise_floor * 100:.1f}pp) "
        f"-> {'OK' if f1_passes else 'NOT BEATEN'}\n"
        f"  hallucination: {hallu_str}"
    )


def render_arm_summary(arm: str, per_repeat: list[RunMetrics]) -> str:
    f1s = [m.triple_loose.f1 for m in per_repeat]
    hallu = [m.hallucination for m in per_repeat]
    agg_f1 = mean_range(f1s)
    agg_hallu = mean_range(hallu)
    f1_str = f"{agg_f1[0] * 100:.1f}% +- {agg_f1[1] * 100:.1f}pp" if agg_f1 else _NA
    hallu_str = f"{agg_hallu[0] * 100:.1f}% +- {agg_hallu[1] * 100:.1f}pp" if agg_hallu else _NA
    return f"arm {arm} ({len(per_repeat)} repeat(s)): triple F1 {f1_str}  hallucination {hallu_str}"
