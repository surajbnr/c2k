#!/usr/bin/env python3
"""Builds the self-contained bundle the website serves: one SQLite file with
a `facts` table for a single run. Every fact is re-verified against the
canonical text (I1) via `c2k.query.search_facts`, and only short context
snippets are written, never full text or PDFs.

Usage:
    PYTHONPATH="" uv run python3 scripts/export_demo.py --run-id a2_r0 \\
        --out web/demo_data/demo.sqlite
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path

from c2k import factsdb
from c2k.config import load_config
from c2k.store import Store

ROOT = Path(__file__).resolve().parent.parent

def load_manifest(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["filename"].removesuffix(".pdf"): r for r in rows}


def load_tickers(fetch_script: Path) -> dict[int, str]:
    """CIK -> ticker, from the seed list's own trailing comments."""
    text = fetch_script.read_text(encoding="utf-8")
    return {int(cik): ticker for cik, ticker in re.findall(r"^\s*(\d+),\s*#\s*(\w+)", text, re.MULTILINE)}


def doc_info(doc_id: str, manifest: dict[str, dict], tickers: dict[int, str]) -> tuple[str, str, str]:
    """Returns (title, source_type, url)."""
    row = manifest.get(doc_id)
    if row is None:
        return doc_id, "unknown", ""
    if row["source"] == "arxiv":
        return row["title"], "arxiv", row["source_url"].replace("/pdf/", "/abs/")
    ticker = tickers.get(row["cik"], f"CIK {row['cik']}")
    return f"{ticker} Form {row['form']}, filed {row['filing_date']}", "edgar", row["source_url"]


def export(run_id: str, out: Path) -> dict[str, int]:
    config = load_config()
    store = Store(ROOT / config.paths.db, canonical_dir=ROOT / Path(config.paths.db).parent.parent / "canonical")
    store.initialize()
    manifest = load_manifest(ROOT / "data" / "pdfs" / "MANIFEST.jsonl")
    tickers = load_tickers(ROOT / "scripts" / "fetch_corpus.py")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    conn = sqlite3.connect(out)
    factsdb.create_schema(conn)
    counts = factsdb.add_run_facts(conn, store, run_id, lambda doc_id: doc_info(doc_id, manifest, tickers))

    counts_by_source = dict(conn.execute("SELECT source_type, COUNT(DISTINCT doc_id) FROM facts GROUP BY source_type"))
    n_docs = conn.execute("SELECT COUNT(DISTINCT doc_id) FROM facts").fetchone()[0]
    for key, value in (
        ("run_id", run_id),
        ("model", config.model.name),
        ("n_facts", str(counts["written"])),
        ("n_docs", str(n_docs)),
        ("n_arxiv", str(counts_by_source.get("arxiv", 0))),
        ("n_edgar", str(counts_by_source.get("edgar", 0))),
    ):
        conn.execute("INSERT INTO meta VALUES (?,?)", (key, value))
    conn.commit()
    conn.close()
    return {
        "stored_triples": counts["stored_triples"],
        "exported": counts["written"],
        "dropped": counts["dropped"],
        "placeholders": counts["placeholders"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", default="a2_r0")
    parser.add_argument("--out", type=Path, default=ROOT / "web" / "demo_data" / "demo.sqlite")
    args = parser.parse_args()
    counts = export(args.run_id, args.out)
    print(f"exported {counts['exported']} of {counts['stored_triples']} stored triples "
          f"({counts['dropped']} dropped by re-verification, {counts['placeholders']} dropped as placeholders) -> {args.out}")


if __name__ == "__main__":
    main()
