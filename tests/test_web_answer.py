"""Template-written answers: pure functions, no bundle needed."""

from __future__ import annotations

from c2k.schema import Relation
from web.search import ANSWER_TEMPLATES, generate_answer, md_escape


def _fact(subject: str, relation: str, obj: str, n_other_docs: int = 0) -> dict:
    return {"subject": subject, "relation": relation, "object": obj, "also_in": [{}] * n_other_docs}


def test_every_relation_has_an_answer_template() -> None:
    assert {r.value for r in Relation} <= set(ANSWER_TEMPLATES)


def test_single_fact_becomes_one_plain_sentence() -> None:
    sentences, left_out = generate_answer([_fact("NVIDIA", "acquired", "Hugging Face")])
    assert sentences == ["NVIDIA acquired Hugging Face."]
    assert left_out == 0


def test_same_subject_and_relation_are_combined() -> None:
    sentences, _ = generate_answer(
        [_fact("EGF", "evaluated_on", "QM9"), _fact("EGF", "evaluated_on", "ZINC250k"), _fact("EGF", "evaluated_on", "QM9")]
    )
    assert sentences == ["EGF was evaluated on QM9 and ZINC250k."]


def test_three_objects_use_commas_and_and() -> None:
    sentences, _ = generate_answer([_fact("A", "outperforms", o) for o in ("B", "C", "D")])
    assert sentences == ["A outperforms B, C and D."]


def test_repeat_across_documents_is_mentioned() -> None:
    sentences, _ = generate_answer([_fact("Apple Inc.", "headquartered_in", "Cupertino, California", n_other_docs=14)])
    assert sentences[0].endswith("(found in 15 documents).")


def test_names_keep_their_trailing_period() -> None:
    sentences, _ = generate_answer([_fact("NVIDIA", "acquired", "Hugging Face, Inc.")])
    assert sentences == ["NVIDIA acquired Hugging Face, Inc\\."]


def test_extracted_text_cannot_inject_markdown() -> None:
    sentences, _ = generate_answer([_fact("[click](http://evil.example)", "acquired", "![x](http://evil.example/p.png)")])
    text = sentences[0]
    assert "](" not in text.replace("\\]\\(", "")
    assert md_escape("$x$ <b>") == "\\$x\\$ \\<b\\>"


def test_long_phrases_are_cut_and_overflow_is_counted() -> None:
    long_name = "x" * 400
    sentences, left_out = generate_answer([_fact(f"S{i}", "uses_method", long_name) for i in range(15)], max_sentences=12)
    assert len(sentences) == 12 and left_out == 3
    assert all(len(s) < 220 for s in sentences)


def test_leadership_wording_does_not_claim_chief_executive() -> None:
    sentences, _ = generate_answer([_fact("Colette M. Kress", "ceo_of", "NVIDIA Corporation")])
    assert "chief executive" not in sentences[0].lower()


# ---- widening when the kind of fact was misread

from web import search as S  # noqa: E402
from web import upload as U  # noqa: E402


class _FixedMatcher:
    def __init__(self, relations, scores=None):
        self._r, self._s = relations, scores or {}

    def scores(self, question):
        return self._s

    def relations(self, question):
        return list(self._r)


def _acme_db():
    conn = U.new_session_db()
    conn.execute(
        "INSERT INTO facts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("f1", "d", "report.pdf", "upload", "", "Acme Corp", "Acme Corp", "acquired", "Widget Inc", "Widget Inc",
         "Acme Corp acquired Widget Inc.", "", "", 0.9),
    )
    conn.commit()
    return conn


def test_a_misread_kind_of_fact_widens_to_everything_known_about_the_name() -> None:
    conn = _acme_db()
    facts, u = S.ask(conn, "who did Acme snap up?", matcher=_FixedMatcher(["founded_by"], {"acquired": 0.65, "founded_by": 0.7}), lex=S.lexicon(conn))
    assert [f["object"] for f in facts] == ["Widget Inc"] and u.widened


def test_a_correctly_read_question_is_not_flagged_as_widened() -> None:
    conn = _acme_db()
    facts, u = S.ask(conn, "who did Acme buy?", matcher=_FixedMatcher(["acquired"]), lex=S.lexicon(conn))
    assert facts and not u.widened


def test_an_explicit_relation_choice_is_never_widened() -> None:
    conn = _acme_db()
    facts, u = S.ask(conn, "who did Acme snap up?", matcher=_FixedMatcher(["founded_by"]), lex=S.lexicon(conn), relation="founded_by")
    assert facts == [] and not u.widened


def test_an_unknown_name_still_gets_no_answer() -> None:
    conn = _acme_db()
    facts, u = S.ask(conn, "who did Zorblax snap up?", matcher=_FixedMatcher(["founded_by"]), lex=S.lexicon(conn))
    assert facts == [] and not u.widened


def test_stats_summarise_the_facts_table() -> None:
    conn = _acme_db()
    st = S.stats(conn)
    assert st["documents"] == 1 and st["facts"] == 1 and st["fact_types"] == 1
    assert st["entities"] == 2 and st["by_source"] == {"upload": 1}


def test_dollar_signs_and_page_breaks_survive_inside_rendered_html() -> None:
    fact = {
        "ctx_before": "A price of\n\npage two ",
        "evidence_text": "$11.9 billion and up to $1.0 billion",
        "ctx_after": " follows.\n\n",
    }
    out = S.highlight(fact)
    assert "$" not in out and out.count("&#36;") == 2
    assert "\n" not in out and "<mark>&#36;11.9 billion and up to &#36;1.0 billion</mark>" in out
    assert "A price of page two" in out
