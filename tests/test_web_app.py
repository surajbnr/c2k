"""The demo site: search logic over the exported bundle, and headless UI
runs of every page. Skipped when the (gitignored, locally built) bundle is
absent, so a fresh clone still passes."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "web" / "demo_data" / "demo.sqlite"

pytestmark = pytest.mark.skipif(not BUNDLE.exists(), reason="run scripts/export_demo.py first")

from web import search as S  # noqa: E402


@pytest.fixture(scope="module")
def conn():
    return S.connect(BUNDLE)


def test_every_use_case_returns_facts(conn) -> None:
    cases = yaml.safe_load((ROOT / "web" / "queries.yaml").read_text(encoding="utf-8"))
    assert len(cases) >= 5
    for uc in cases:
        facts = S.search(conn, entity=uc.get("entity"), relation=uc.get("relation"), source_type=uc.get("source_type"))
        assert facts, f"use case {uc['id']!r} returns nothing"


def test_free_text_query_uses_relation_and_entity(conn) -> None:
    facts = S.search(conn, text="who acquired Hugging Face")
    assert facts and all(f["relation"] == "acquired" for f in facts)


def test_empty_and_overlong_input_are_safe(conn) -> None:
    assert len(S.search(conn, text="", limit=5)) == 5
    assert S.clean_input("x" * 5000) == "x" * S.MAX_INPUT_CHARS
    assert S.search(conn, entity="'; DROP TABLE facts; --") == []
    assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] > 0


def test_bundle_is_read_only(conn) -> None:
    import sqlite3

    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM facts")


def test_highlight_escapes_document_text(conn) -> None:
    fact = {"ctx_before": "a <script>alert(1)</script> ", "evidence_text": "x & y", "ctx_after": " <b>z</b>"}
    out = S.highlight(fact)
    assert "<script>" not in out and "<b>" not in out
    assert "<mark>x &amp; y</mark>" in out


def test_entity_page_lists_merged_surfaces(conn) -> None:
    page = S.entity_page(conn, "NVIDIA Corporation")
    assert page["n_facts"] > 0 and "NVIDIA" in page["surfaces"]


def test_app_pages_render_without_exception() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "web" / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert any("Knowledge extraction from documents" in m.value for m in at.markdown)

    at.button(key="uc-headquarters").click().run()
    assert not at.exception
    assert any("distinct fact" in c.value for c in at.caption)

    at.text_input(key="q_text").set_value("where is Walmart headquartered").run()
    assert not at.exception
    assert any("Walmart" in m.value for m in at.markdown)

    at.text_input(key="q_text").set_value("zzzzqqqq nothing").run()
    assert not at.exception
    assert any("could not be interpreted" in i.value for i in at.info)

    at.text_input(key="q_text").set_value("where is Acme Corp located").run()
    assert not at.exception
    assert any("No matching facts" in i.value for i in at.info)


def test_uninterpretable_text_returns_nothing_not_everything(conn) -> None:
    assert S.search(conn, text="zzzzqqqq nothing") == []
    assert S.search(conn, text="tell me more") == []


def test_group_facts_collapses_repeats_across_documents(conn) -> None:
    facts = S.search(conn, relation="headquartered_in", source_type="edgar")
    grouped = S.group_facts(facts)
    assert len(grouped) < len(facts)
    assert sum(1 + len(g["also_in"]) for g in grouped) >= len({f["doc_id"] for f in facts})
    keys = {(g["subject_canonical"], g["object_canonical"]) for g in grouped}
    assert len(keys) == len(grouped)


@pytest.mark.parametrize(
    "question",
    ["where is apple located?", "where is apple headquartered?", "Where is Apple based", "apple"],
)
def test_same_meaning_questions_give_the_same_apple_answer(conn, question) -> None:
    answer, _ = S.generate_answer(S.group_facts(S.search(conn, text=question)))
    assert any("Cupertino" in s for s in answer), question


def test_other_paraphrased_questions_find_their_facts(conn) -> None:
    def answer(q: str) -> str:
        return " ".join(S.generate_answer(S.group_facts(S.search(conn, text=q)))[0])

    assert "Hugging Face" in answer("who bought Hugging Face")
    assert "Hugging Face" in answer("which company did NVIDIA acquire")
    assert "Firefly" in answer("what did Johnson & Johnson purchase")
    assert "ZINC250k" in answer("what was EGF tested on")
    assert S.search(conn, text="tell me more") == []


# ---- meaning-based question understanding, end to end (needs the local model)

from c2k.understand import RelationMatcher  # noqa: E402


def _model_is_cached() -> bool:
    import os

    cache = Path(os.environ.get("C2K_EMBED_CACHE") or Path.home() / ".cache" / "c2k_fastembed")
    return cache.exists() and any(cache.iterdir())


needs_model = pytest.mark.skipif(not _model_is_cached(), reason="embedding model not downloaded")


@pytest.fixture(scope="module")
def matcher():
    return RelationMatcher()


@pytest.fixture(scope="module")
def lex(conn):
    return S.lexicon(conn)


def _answer(conn, matcher, lex, question: str) -> str:
    facts, _ = S.ask(conn, question, matcher=matcher, lex=lex)
    return " ".join(S.generate_answer(S.group_facts(facts))[0])


@needs_model
@pytest.mark.parametrize(
    "question, expected",
    [
        ("where is apple located?", "Cupertino"),
        ("what's the home base of walmart", "Bentonville"),
        ("which city does Visa operate out of", "San Francisco"),
        ("where is disney situated", "Burbank"),
        ("tell me the city Netflix calls home", "Los Gatos"),
        ("who snapped up hugging face", "Hugging Face"),
        ("which company did Johnson & Johnson snap up", "Firefly"),
        ("who is NVIDIA working together with", "SB Energy"),
        ("who's in charge of disney", "D'Amaro"),
        ("who is the creator of Armadin", "Mandia"),
        ("what did they test EGF against", "QM9"),
        ("what does EGF do better than", "DiGress"),
        ("who paid for this research", "NSERC"),
        ("which university is Julita Vassileva from", "Saskatchewan"),
    ],
)
def test_differently_worded_questions_reach_the_right_facts(conn, matcher, lex, question, expected) -> None:
    assert expected in _answer(conn, matcher, lex, question)
    assert expected in _answer(conn, matcher, lex, question.lower())


@needs_model
def test_questions_about_nothing_or_unknown_things_get_no_answer(conn, matcher, lex) -> None:
    for q in ("what's the weather like", "hello", "tell me a joke", "where is Acme Corp located"):
        facts, _ = S.ask(conn, q, matcher=matcher, lex=lex)
        assert facts == [], q


@needs_model
def test_a_known_name_with_no_fact_of_the_asked_kind_is_widened_and_flagged(conn, matcher, lex) -> None:
    facts, u = S.ask(conn, "who got taken over by Alphabet", matcher=matcher, lex=lex)
    assert u.widened and facts
    assert all(f["relation"] != "acquired" for f in facts)
    assert any("Alphabet" in f["subject"] + f["object"] for f in facts)


@needs_model
def test_leftover_filler_words_are_reported_not_used(conn, matcher, lex) -> None:
    _, u = S.ask(conn, "who paid for this research", matcher=matcher, lex=lex)
    assert u.entity is None and "paid" in u.ignored


def test_no_placeholder_values_in_the_bundle(conn) -> None:
    rows = conn.execute("SELECT subject, object FROM facts").fetchall()
    assert not [r for r in rows if r[0].startswith("<") or r[1].startswith("<") or r[1].lower() == "unknown"]


# ---- the "Your documents" tab

from streamlit.testing.v1 import AppTest  # noqa: E402

from web import upload as U  # noqa: E402


def _session_with_one_document():
    conn = U.new_session_db()
    conn.execute(
        "INSERT INTO facts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("f1", "upload_x", "report.pdf", "upload", "", "Acme Corp", "Acme Corp", "acquired", "Widget Inc", "Widget Inc",
         "Acme Corp acquired Widget Inc in 2024.", "Before. ", " After.", 0.9),
    )
    conn.commit()
    docs = [{"doc_id": "upload_x", "name": "report.pdf", "n_facts": 1, "cost": 0.012}]
    return conn, docs


def _app(conn=None, docs=None) -> AppTest:
    at = AppTest.from_file(str(ROOT / "web" / "app.py"), default_timeout=90)
    if conn is not None:
        at.session_state["user_conn"] = conn
        at.session_state["user_docs"] = docs
    return at


def test_questions_over_the_visitors_own_document_are_answered(monkeypatch) -> None:
    monkeypatch.setenv("C2K_MEANING_MATCHING", "0")
    conn, docs = _session_with_one_document()
    at = _app(conn, docs).run()
    assert not at.exception
    assert any("report.pdf" in m.value.replace("\\", "") for m in at.markdown)

    at.text_input(key="u_q").set_value("who did Acme buy?").run()
    assert not at.exception
    assert any("Acme Corp acquired Widget Inc" in m.value.replace("\\", "") for m in at.markdown)


def test_uploaded_facts_never_appear_in_the_shared_search(monkeypatch) -> None:
    monkeypatch.setenv("C2K_MEANING_MATCHING", "0")
    conn, docs = _session_with_one_document()
    at = _app(conn, docs).run()
    at.text_input(key="q_text").set_value("who did Acme buy?").run()
    assert not any("Widget Inc" in m.value for m in at.markdown)


def test_removing_a_document_removes_its_facts(monkeypatch) -> None:
    monkeypatch.setenv("C2K_MEANING_MATCHING", "0")
    conn, docs = _session_with_one_document()
    at = _app(conn, docs).run()
    at.button(key="u_rm_upload_x").click().run()
    assert not at.exception
    assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 0
    assert docs == []


def test_without_a_server_key_visitors_must_bring_their_own(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    at = _app().run()
    assert at.radio(key="u_key_mode").options == ["Use my own API key"]
    assert at.text_input(key="u_own_key").proto.type == 1  # 1 = masked (password) input


def test_a_server_key_can_be_protected_by_a_password(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-server-secret")
    monkeypatch.setenv("C2K_UPLOAD_PASSWORD", "open-sesame")
    at = _app().run()
    assert "Use the site's API key" in at.radio(key="u_key_mode").options
    at.radio(key="u_key_mode").set_value("Use the site's API key").run()
    assert at.text_input(key="u_password").proto.type == 1  # masked input
    assert not any("sk-server-secret" in str(m.value) for m in at.markdown)


@needs_model
def test_questions_facts_cannot_answer_get_the_closest_passages() -> None:
    from c2k.understand import FastEmbedder
    from web import passages as P

    text = (
        "NVIDIA entered into a definitive agreement to acquire Hugging Face, Inc. "
        "The transaction includes an approximately $11.9 billion purchase price payable to stockholders. "
        "The closing is subject to customary regulatory approvals."
    )
    conn, docs = _session_with_one_document()
    at = _app(conn, docs)
    at.session_state["user_index"] = {"upload_x": ("report.pdf", P.build_index(text, FastEmbedder()))}
    at.run()
    assert not at.exception

    at.text_input(key="u_q").set_value("how much did NVIDIA pay for Hugging Face").run()
    assert not at.exception
    assert any("11.9 billion" in m.value.replace("\\", "") for m in at.markdown)
    assert any("closest passages" in m.value.lower() for m in at.markdown)

    at.text_input(key="u_q").set_value("give me a pancake recipe").run()
    assert not at.exception
    assert any("Nothing in your documents matches" in i.value for i in at.info)
