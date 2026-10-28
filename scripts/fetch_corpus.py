#!/usr/bin/env python3
"""Assembles the ~120-PDF corpus §10 assumes already exists — a standalone
script, deliberately NOT a `c2k` subcommand, so §11's CLI list stays exactly
as specified (recorded in `SPEC.md` §17's decision log).

Two sources, chosen to exercise both halves of `config/relations.yaml`'s
mixed corporate + scientific ontology:

  --source arxiv  cs.CL/cs.LG papers -> authored_by, affiliated_with,
                  evaluated_on, uses_method, outperforms, funded_by,
                  reports_metric
  --source edgar  SEC 8-K filings from a checked-in list of large-cap
                  companies -> acquired, subsidiary_of, headquartered_in,
                  ceo_of, partnered_with, invested_in

EDGAR filings arrive as HTML, not PDF; `pymupdf.Story` (PyMuPDF is already a
c2k dependency, so this adds nothing new) renders them to PDF. 8-K is the
default filing type deliberately: it's short and event-driven (an
acquisition, an executive change, a new HQ), which is exactly the shape of
fact this project's relation schema targets — a 10-K commonly runs to
multi-megabyte HTML dominated by financial tables, which is both slow to
render and a poor match for the schema.

Usage:
    python scripts/fetch_corpus.py --source arxiv --n 60 --out data/pdfs
    python scripts/fetch_corpus.py --source edgar --n 60 --out data/pdfs \\
        --contact you@example.com
    python scripts/fetch_corpus.py --source arxiv --n 3 --out data/pdfs --dry-run

Idempotent: re-running skips filenames that already exist on disk. Every
fetch (real or dry-run) is recorded in `data/pdfs/MANIFEST.jsonl` — the
corpus's own source-of-truth record, in the same spirit as I6's run
reproducibility.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}
SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SEC_ARCHIVES = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{doc}"

# A fixed list of large-cap CIKs (from data.sec.gov's own ticker->CIK
# mapping), chosen only for filing volume and name recognition — not an
# endorsement, just a stable, reproducible seed list. 30 companies comfortably
# yields 60 recent 8-Ks between them.
DEFAULT_CIKS: list[int] = [
    320193,   # AAPL
    789019,   # MSFT
    1652044,  # GOOGL
    1018724,  # AMZN
    1326801,  # META
    1045810,  # NVDA
    1318605,  # TSLA
    19617,    # JPM
    200406,   # JNJ
    104169,   # WMT
    80424,    # PG
    1403161,  # V
    1141391,  # MA
    354950,   # HD
    1744489,  # DIS
    1065280,  # NFLX
    50863,    # INTC
    858877,   # CSCO
    1341439,  # ORCL
    51143,    # IBM
    796343,   # ADBE
    1108524,  # CRM
    78003,    # PFE
    21344,    # KO
    77476,    # PEP
    63908,    # MCD
    320187,   # NKE
    12927,    # BA
    40545,    # GE
    1097149,  # AMD
]

_MAX_EDGAR_HTML_BYTES = 3_000_000  # skip outsized documents; see module docstring


def _http_get(url: str, *, user_agent: str, accept: str | None = None, timeout: float = 30.0) -> bytes:
    headers = {"User-Agent": user_agent}
    if accept:
        headers["Accept"] = accept
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# ---------------------------------------------------------------- arXiv


def parse_arxiv_feed(xml_text: str) -> list[dict]:
    """Parses an arXiv Atom feed into `[{id, title, pdf_url}]`. Pure/offline
    — the whole reason `fetch_arxiv` below is a thin wrapper around this."""
    root = ET.fromstring(xml_text)
    entries = []
    for entry in root.findall("atom:entry", ARXIV_NS):
        raw_id = entry.findtext("atom:id", default="", namespaces=ARXIV_NS)
        arxiv_id = raw_id.rsplit("/", 1)[-1]  # "http://arxiv.org/abs/2401.01234v1" -> "2401.01234v1"
        title = (entry.findtext("atom:title", default="", namespaces=ARXIV_NS) or "").strip()
        pdf_url = None
        for link in entry.findall("atom:link", ARXIV_NS):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href")
                break
        if arxiv_id and pdf_url:
            entries.append({"id": arxiv_id, "title": title, "pdf_url": pdf_url})
    return entries


def filename_for_arxiv(arxiv_id: str) -> str:
    return f"arxiv_{arxiv_id}.pdf"


def fetch_arxiv(
    n: int,
    out_dir: Path,
    manifest_path: Path,
    *,
    query: str = "cat:cs.CL OR cat:cs.LG",
    dry_run: bool = False,
    sleep: float = 3.0,
    user_agent: str = "c2k-course-project/1.0",
) -> int:
    """Fetches up to `n` recent cs.CL/cs.LG papers. Returns the number of
    files newly written (dry-run: number that *would* be written)."""
    written = 0
    start = 0
    page_size = min(max(n * 2, 20), 100)  # over-fetch a bit; some ids may already exist on disk
    while written < n:
        params = urllib.parse.urlencode(
            {
                "search_query": query,
                "start": start,
                "max_results": page_size,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
        )
        url = f"{ARXIV_API}?{params}"
        try:
            xml_text = _http_get(url, user_agent=user_agent).decode("utf-8")
        except urllib.error.URLError as e:
            print(f"  arxiv: request failed ({e}); stopping.", file=sys.stderr)
            break
        entries = parse_arxiv_feed(xml_text)
        if not entries:
            break
        for entry in entries:
            if written >= n:
                break
            fname = filename_for_arxiv(entry["id"])
            out_path = out_dir / fname
            if out_path.exists():
                continue
            if dry_run:
                print(f"  [dry-run] would fetch {entry['pdf_url']} -> {fname}")
            else:
                try:
                    pdf_bytes = _http_get(entry["pdf_url"], user_agent=user_agent, accept="application/pdf")
                except urllib.error.URLError as e:
                    print(f"  arxiv: failed to fetch {entry['pdf_url']}: {e}", file=sys.stderr)
                    continue
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(pdf_bytes)
                print(f"  fetched {fname} ({len(pdf_bytes)} bytes)")
                time.sleep(sleep)
            _append_manifest(
                manifest_path,
                {
                    "filename": fname,
                    "source": "arxiv",
                    "source_url": entry["pdf_url"],
                    "arxiv_id": entry["id"],
                    "title": entry["title"],
                    "fetched_at": _now(),
                    "dry_run": dry_run,
                },
            )
            written += 1
        start += page_size
    return written


# ---------------------------------------------------------------- EDGAR


def parse_edgar_submissions(submissions: dict, *, filing_types: tuple[str, ...] = ("8-K",)) -> list[dict]:
    """Parses a `data.sec.gov/submissions/CIK##########.json` payload into
    `[{form, accession, primary_document, filing_date}]`, most recent first.
    Pure/offline."""
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    out = []
    for i, form in enumerate(forms):
        if form not in filing_types:
            continue
        out.append(
            {
                "form": form,
                "accession": recent["accessionNumber"][i],
                "primary_document": recent["primaryDocument"][i],
                "filing_date": recent["filingDate"][i],
            }
        )
    return out


def build_edgar_doc_url(cik: int, accession: str, primary_document: str) -> str:
    return SEC_ARCHIVES.format(cik=cik, accession_nodash=accession.replace("-", ""), doc=primary_document)


def filename_for_edgar(cik: int, form: str, filing_date: str) -> str:
    return f"edgar_{cik:010d}_{form}_{filing_date}.pdf"


def render_html_to_pdf(html: str, out_path: Path) -> None:
    """Renders HTML (an EDGAR filing document, XBRL tags and all) to a PDF
    via `pymupdf.Story` — already a c2k dependency (`ingest.py` uses
    PyMuPDF), so this adds nothing new. `Story` tolerates the XBRL namespace
    noise real filings carry; verified against a live 8-K during planning."""
    import pymupdf

    story = pymupdf.Story(html=html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pymupdf.DocumentWriter(str(out_path))
    page_rect = pymupdf.paper_rect("letter")
    content_rect = page_rect + (36, 36, -36, -36)
    more = 1
    while more:
        dev = writer.begin_page(page_rect)
        more, _ = story.place(content_rect)
        story.draw(dev)
        writer.end_page()
    writer.close()


def fetch_edgar(
    n: int,
    out_dir: Path,
    manifest_path: Path,
    *,
    contact: str,
    filing_type: str = "8-K",
    ciks: list[int] | None = None,
    dry_run: bool = False,
    sleep: float = 0.3,
    max_per_cik: int | None = None,
) -> int:
    """Fetches up to `n` filings of `filing_type` across `ciks` (default
    `DEFAULT_CIKS`), rendering each to a PDF via `render_html_to_pdf`.
    `contact` becomes part of the SEC-mandated User-Agent (SEC returns 403
    without a contact — verified during planning) and is never hardcoded;
    pass it explicitly or via `C2K_SEC_CONTACT`."""
    user_agent = f"C2K Course Project ({contact})"
    ciks = ciks if ciks is not None else DEFAULT_CIKS
    written = 0
    for cik in ciks:
        if written >= n:
            break
        try:
            raw = _http_get(SEC_SUBMISSIONS.format(cik=cik), user_agent=user_agent, accept="application/json")
        except urllib.error.URLError as e:
            print(f"  edgar: submissions fetch failed for CIK {cik}: {e}", file=sys.stderr)
            continue
        time.sleep(sleep)
        submissions = json.loads(raw)
        # Counts filings already on disk too, so re-runs stay balanced across companies.
        have = len(list(out_dir.glob(f"edgar_{cik:010d}_*.pdf"))) if out_dir.exists() else 0
        for filing in parse_edgar_submissions(submissions, filing_types=(filing_type,)):
            if written >= n:
                break
            if max_per_cik is not None and have >= max_per_cik:
                break
            fname = filename_for_edgar(cik, filing["form"], filing["filing_date"])
            out_path = out_dir / fname
            if out_path.exists():
                continue
            doc_url = build_edgar_doc_url(cik, filing["accession"], filing["primary_document"])
            if dry_run:
                print(f"  [dry-run] would fetch {doc_url} -> {fname}")
            else:
                try:
                    html_bytes = _http_get(doc_url, user_agent=user_agent, accept="text/html")
                except urllib.error.URLError as e:
                    print(f"  edgar: failed to fetch {doc_url}: {e}", file=sys.stderr)
                    continue
                if len(html_bytes) > _MAX_EDGAR_HTML_BYTES:
                    print(f"  edgar: skipping {fname}, {len(html_bytes)} bytes > cap", file=sys.stderr)
                    time.sleep(sleep)
                    continue
                html_text = html_bytes.decode("utf-8", errors="ignore")
                try:
                    render_html_to_pdf(html_text, out_path)
                except Exception as e:  # pymupdf.Story can choke on malformed markup
                    print(f"  edgar: failed to render {fname}: {e}", file=sys.stderr)
                    time.sleep(sleep)
                    continue
                print(f"  fetched {fname}")
                time.sleep(sleep)
            _append_manifest(
                manifest_path,
                {
                    "filename": fname,
                    "source": "edgar",
                    "source_url": doc_url,
                    "cik": cik,
                    "form": filing["form"],
                    "filing_date": filing["filing_date"],
                    "accession": filing["accession"],
                    "fetched_at": _now(),
                    "dry_run": dry_run,
                },
            )
            written += 1
            have += 1
    return written


# ---------------------------------------------------------------- shared


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_manifest(manifest_path: Path, record: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["arxiv", "edgar", "both"], required=True)
    parser.add_argument("--n", type=int, default=60, help="files to fetch per source (default 60)")
    parser.add_argument("--out", type=Path, default=Path("data/pdfs"))
    parser.add_argument("--contact", default=os.environ.get("C2K_SEC_CONTACT"), help="required for --source edgar; or set C2K_SEC_CONTACT")
    parser.add_argument("--query", default="cat:cs.CL OR cat:cs.LG", help="arXiv search_query (default: cs.CL OR cs.LG)")
    parser.add_argument("--filing-type", default="8-K", help="EDGAR form type (default: 8-K)")
    parser.add_argument("--max-per-cik", type=int, default=None, help="EDGAR: cap filings per company, counting files already on disk")
    parser.add_argument("--dry-run", action="store_true", help="print what would be fetched without downloading")
    args = parser.parse_args()

    out_dir: Path = args.out
    manifest_path = out_dir / "MANIFEST.jsonl"

    if args.source in ("edgar", "both") and not args.contact:
        parser.error("--source edgar requires --contact (an email SEC can reach) or C2K_SEC_CONTACT")

    total = 0
    if args.source in ("arxiv", "both"):
        print(f"fetching up to {args.n} arXiv paper(s) ({args.query})...")
        total += fetch_arxiv(args.n, out_dir, manifest_path, query=args.query, dry_run=args.dry_run)
    if args.source in ("edgar", "both"):
        print(f"fetching up to {args.n} EDGAR {args.filing_type} filing(s)...")
        total += fetch_edgar(
            args.n,
            out_dir,
            manifest_path,
            contact=args.contact,
            filing_type=args.filing_type,
            dry_run=args.dry_run,
            max_per_cik=args.max_per_cik,
        )

    print(f"{'would fetch' if args.dry_run else 'fetched'} {total} file(s) total.")


if __name__ == "__main__":
    main()
