"""Offline tests for `scripts/fetch_corpus.py` (SPEC.md §17 decision log):
URL/filename construction, Atom/JSON parsing against checked-in response
fixtures, manifest shape, and that a `Story`-rendered PDF yields extractable
text through the existing `ingest.build_canonical_text`. Nothing here touches
the network — the live HTTP paths (`fetch_arxiv`/`fetch_edgar`) were
hand-verified against the real APIs during planning, not re-verified by CI.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "fetch_corpus.py"


def _load_fetch_corpus():
    spec = importlib.util.spec_from_file_location("fetch_corpus", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["fetch_corpus"] = module
    spec.loader.exec_module(module)
    return module


fetch_corpus = _load_fetch_corpus()


def test_parse_arxiv_feed() -> None:
    xml_text = (FIXTURES / "arxiv_feed_sample.xml").read_text(encoding="utf-8")
    entries = fetch_corpus.parse_arxiv_feed(xml_text)
    assert entries == [
        {
            "id": "2401.01234v1",
            "title": "A Benchmark for Something Interesting",
            "pdf_url": "https://arxiv.org/pdf/2401.01234v1",
        },
        {
            "id": "2401.05678v2",
            "title": "Outperforming Baselines on a New Dataset",
            "pdf_url": "https://arxiv.org/pdf/2401.05678v2",
        },
    ]


def test_filename_for_arxiv() -> None:
    assert fetch_corpus.filename_for_arxiv("2401.01234v1") == "arxiv_2401.01234v1.pdf"


def test_parse_edgar_submissions_filters_by_form() -> None:
    submissions = json.loads((FIXTURES / "edgar_submissions_sample.json").read_text())
    filings = fetch_corpus.parse_edgar_submissions(submissions, filing_types=("8-K",))
    assert [f["accession"] for f in filings] == [
        "0001193125-26-380280",
        "0001193125-26-323632",
        "0001193125-26-258667",
    ]
    assert all(f["form"] == "8-K" for f in filings)
    # 10-K and Form 4 excluded
    assert not any(f["primary_document"] == "msft-20260630.htm" for f in filings)


def test_parse_edgar_submissions_10k() -> None:
    submissions = json.loads((FIXTURES / "edgar_submissions_sample.json").read_text())
    filings = fetch_corpus.parse_edgar_submissions(submissions, filing_types=("10-K",))
    assert len(filings) == 1
    assert filings[0]["primary_document"] == "msft-20260630.htm"


def test_build_edgar_doc_url() -> None:
    url = fetch_corpus.build_edgar_doc_url(789019, "0001193125-26-380280", "d291965d8k.htm")
    assert url == "https://www.sec.gov/Archives/edgar/data/789019/000119312526380280/d291965d8k.htm"


def test_filename_for_edgar() -> None:
    assert fetch_corpus.filename_for_edgar(789019, "8-K", "2026-09-02") == "edgar_0000789019_8-K_2026-09-02.pdf"


def test_render_html_to_pdf_yields_extractable_text(tmp_path: Path) -> None:
    """The whole reason this uses `pymupdf.Story` instead of e.g. saving raw
    HTML: the rest of the pipeline (`c2k ingest`) only reads PDFs. This checks
    the round trip through the real ingest module, not just that a file gets
    written."""
    from c2k.ingest import build_canonical_text

    html = "<html><body><h1>Siemens Corp</h1><p>Siemens acquired Mentor Graphics in 2017.</p></body></html>"
    out_path = tmp_path / "test.pdf"
    fetch_corpus.render_html_to_pdf(html, out_path)
    assert out_path.exists()

    text, spans = build_canonical_text(out_path)
    assert "Siemens" in text
    assert "acquired" in text
    assert "Mentor" in text
    assert spans  # a real page/bbox index was produced, same as any other ingested PDF


def test_manifest_append_and_shape(tmp_path: Path) -> None:
    manifest_path = tmp_path / "MANIFEST.jsonl"
    fetch_corpus._append_manifest(manifest_path, {"filename": "arxiv_1.pdf", "source": "arxiv"})
    fetch_corpus._append_manifest(manifest_path, {"filename": "edgar_1.pdf", "source": "edgar"})
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    rows = [json.loads(line) for line in lines]
    assert rows[0]["filename"] == "arxiv_1.pdf"
    assert rows[1]["source"] == "edgar"


def test_default_ciks_are_ten_digit_ints() -> None:
    assert len(fetch_corpus.DEFAULT_CIKS) >= 20
    assert all(isinstance(c, int) and 0 < c < 10**10 for c in fetch_corpus.DEFAULT_CIKS)
    assert len(set(fetch_corpus.DEFAULT_CIKS)) == len(fetch_corpus.DEFAULT_CIKS)  # no duplicates


def test_main_requires_contact_for_edgar(capsys) -> None:
    import pytest

    with pytest.raises(SystemExit):
        sys.argv = ["fetch_corpus.py", "--source", "edgar", "--n", "1"]
        fetch_corpus.main()
    captured = capsys.readouterr()
    assert "--contact" in captured.err
