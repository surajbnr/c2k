"""Document upload (web/upload.py), with a fake model so nothing is billed."""

from __future__ import annotations

import pymupdf
import pytest

from c2k import llm
from web import search as S
from web import upload as U

SENTENCES = [
    "Acme Corp acquired Widget Inc in 2024.",
    "Acme Corporation is headquartered in Springfield, Illinois.",
    "The deal was approved by the board after a long review of the company finances and its future plans.",
]
BODY = " ".join(SENTENCES) + " " + " ".join(["The report continues with routine details about operations."] * 12)


def make_pdf(text: str = BODY, pages: int = 1) -> bytes:
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 780), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def triple(subject, relation, obj, evidence, confidence=0.9):
    return {"subject": subject, "relation": relation, "object": obj, "evidence_text": evidence, "confidence": confidence}


GOOD = [
    triple("Acme Corp", "acquired", "Widget Inc", SENTENCES[0]),
    triple("Acme Corporation", "headquartered_in", "Springfield, Illinois", SENTENCES[1]),
    triple("Acme Corp", "acquired", "Nowhere Ltd", "This sentence is not in the document at all."),
]


@pytest.fixture
def fake_model(monkeypatch):
    calls: list[dict] = []

    def provider(prompt, response_model, *, model, temperature, max_retries):
        calls.append({"key_seen": llm._api_key, "prompt": prompt})
        return response_model.model_validate({"triples": GOOD}), {"input_tokens": 1000, "output_tokens": 200}

    monkeypatch.setitem(llm._PROVIDERS, "anthropic", provider)
    return calls


# ---------------------------------------------------------------- planning


def test_plan_reads_a_pdf_and_estimates_cost() -> None:
    plan = U.plan_upload("report.pdf", make_pdf())
    assert plan.n_pages == 1 and plan.n_chunks >= 1 and plan.n_chars > U.MIN_CHARS
    assert plan.est_cost_usd == U.estimate_cost(plan.n_chunks) > 0
    assert plan.doc_id.startswith("upload_")


def test_plan_rejects_unreadable_oversized_scanned_and_long_files() -> None:
    with pytest.raises(U.UploadError, match="readable PDF"):
        U.plan_upload("x.pdf", b"this is not a pdf")
    with pytest.raises(U.UploadError, match="larger than"):
        U.plan_upload("x.pdf", b"0" * (U.MAX_BYTES + 1))
    with pytest.raises(U.UploadError, match="readable text"):
        U.plan_upload("blank.pdf", make_pdf("", pages=1))
    with pytest.raises(U.UploadError, match="pages"):
        U.plan_upload("long.pdf", make_pdf(pages=U.MAX_PAGES + 1))


def test_a_document_with_too_many_sections_is_refused_with_a_clear_message(monkeypatch) -> None:
    monkeypatch.setattr(U, "MAX_CHUNKS", 1)
    long_text = (SENTENCES[0] + " The report continues with routine details about operations. ") * 28  # fits on a page; repeated over 3 pages it is more than one section
    with pytest.raises(U.UploadError, match="too long"):
        U.plan_upload("long.pdf", make_pdf(long_text, pages=3))


def test_the_default_limits_are_a_hundred_pages_and_enough_sections_for_them() -> None:
    assert U.MAX_PAGES == 100 and U.MAX_CHUNKS >= 150


def test_the_plan_estimates_time_as_well_as_cost() -> None:
    plan = U.plan_upload("report.pdf", make_pdf())
    assert plan.est_seconds == plan.n_chunks * U.SECONDS_PER_CHUNK > 0


def test_limits_can_be_changed_with_environment_variables(monkeypatch) -> None:
    import importlib

    monkeypatch.setenv("C2K_MAX_PAGES", "7")
    monkeypatch.setenv("C2K_MAX_SECTIONS", "9")
    reloaded = importlib.reload(U)
    try:
        assert reloaded.MAX_PAGES == 7 and reloaded.MAX_CHUNKS == 9
    finally:
        monkeypatch.undo()
        importlib.reload(U)


def test_file_names_are_reduced_to_a_plain_base_name() -> None:
    assert U.plan_upload("../../etc/passwd.pdf", make_pdf()).name == "passwd.pdf"


# ---------------------------------------------------------------- extraction


def test_extraction_keeps_only_facts_whose_quote_is_in_the_document(fake_model) -> None:
    result = U.extract_upload(U.plan_upload("report.pdf", make_pdf()), api_key="sk-test")
    facts = {(r[5], r[7], r[8]) for r in result.rows}
    assert ("Acme Corp", "acquired", "Widget Inc") in facts
    assert ("Acme Corporation", "headquartered_in", "Springfield, Illinois") in facts
    assert not any("Nowhere" in r[8] for r in result.rows)
    assert result.n_span_failures >= 1 and result.n_failed_chunks == 0


def test_names_for_the_same_thing_are_merged(fake_model) -> None:
    result = U.extract_upload(U.plan_upload("report.pdf", make_pdf()), api_key="sk-test")
    canon = {r[6] for r in result.rows if r[5] in ("Acme Corp", "Acme Corporation")}
    assert len(canon) == 1


def test_cost_is_reported_from_the_tokens_actually_used(fake_model) -> None:
    plan = U.plan_upload("report.pdf", make_pdf())
    result = U.extract_upload(plan, api_key="sk-test")
    assert result.cost_usd == pytest.approx(plan.n_chunks * (1000 * 1.0 + 200 * 5.0) / 1_000_000)


def test_a_visitors_key_is_used_for_the_job_then_cleared(fake_model) -> None:
    result = U.extract_upload(U.plan_upload("report.pdf", make_pdf()), api_key="sk-visitor-secret")
    assert {c["key_seen"] for c in fake_model} == {"sk-visitor-secret"}
    assert llm._api_key is None and llm._store is None
    assert "sk-visitor-secret" not in repr(result)


def test_only_one_extraction_runs_at_a_time(fake_model) -> None:
    plan = U.plan_upload("report.pdf", make_pdf())
    assert U._LOCK.acquire(blocking=False)
    try:
        with pytest.raises(U.UploadError, match="Another document"):
            U.extract_upload(plan, api_key="sk-test")
    finally:
        U._LOCK.release()
    U.extract_upload(plan, api_key="sk-test")  # and it is free again afterwards


def test_when_every_request_fails_the_visitor_is_told_and_settings_are_cleared(monkeypatch) -> None:
    def broken(prompt, response_model, *, model, temperature, max_retries):
        raise ValueError("401 invalid x-api-key")

    monkeypatch.setitem(llm._PROVIDERS, "anthropic", broken)
    with pytest.raises(U.UploadError, match="Every request"):
        U.extract_upload(U.plan_upload("report.pdf", make_pdf()), api_key="sk-bad")
    assert llm._api_key is None and not U._LOCK.locked()


def test_progress_is_reported_per_chunk(fake_model) -> None:
    seen: list[tuple[int, int]] = []
    plan = U.plan_upload("report.pdf", make_pdf())
    U.extract_upload(plan, api_key="sk-test", progress=lambda done, total: seen.append((done, total)))
    assert seen and seen[-1] == (plan.n_chunks, plan.n_chunks)


# ---------------------------------------------------------------- the visitor's own table


def test_uploaded_facts_can_be_asked_about_and_removed(fake_model) -> None:
    result = U.extract_upload(U.plan_upload("report.pdf", make_pdf()), api_key="sk-test")
    conn = U.new_session_db()
    U.add_to_session_db(conn, result)

    facts, _ = S.ask(conn, "who did Acme buy?", lex=S.lexicon(conn))
    answer = " ".join(S.generate_answer(S.group_facts(facts))[0])
    assert "Widget Inc" in answer and facts[0]["doc_title"] == "report.pdf"

    U.add_to_session_db(conn, result)  # adding the same document again does not duplicate it
    assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == len(result.rows)

    U.remove_from_session_db(conn, result.doc_id)
    assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 0


@pytest.mark.parametrize(
    "a, b, same",
    [
        ("NVIDIA", "NVIDIA Corporation", True),
        ("Acme Corp", "Acme Corporation", True),
        ("The Walt Disney Company", "Walt Disney", True),
        ("Hugging Face, Inc.", "Hugging Face", True),
        ("Apple", "Apple Watch", False),
        ("Johnson & Johnson", "Procter & Gamble", False),
    ],
)
def test_merge_key_treats_corporate_suffixes_as_the_same_name(a, b, same) -> None:
    assert (U._merge_key(a) == U._merge_key(b)) is same
