"""C2K demo site: query structured facts extracted from documents, each linked
to the exact sentence it came from, and optionally add your own PDF.

Run locally:  PYTHONPATH=. streamlit run web/app.py --server.address 127.0.0.1
Uploads need an Anthropic API key (see the runbook); everything else is
read-only and makes no network calls.
"""

from __future__ import annotations

import hmac
import os
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402
import yaml  # noqa: E402

from web import passages as P  # noqa: E402
from web.charts import chart_html  # noqa: E402
from web import search as S  # noqa: E402
from web import theme as T  # noqa: E402
from web import upload as U  # noqa: E402

DB_PATH = Path(os.environ.get("C2K_DEMO_DB", ROOT / "web" / "demo_data" / "demo.sqlite"))
USE_CASES = yaml.safe_load((ROOT / "web" / "queries.yaml").read_text(encoding="utf-8"))
SHOW_FIRST = 8
SOURCE_LABEL = {"arxiv": "arXiv", "edgar": "SEC 8-K", "upload": "Upload"}
OWN_KEY = "Use my own API key"
SITE_KEY = "Use the site's API key"

st.set_page_config(page_title="C2K | Knowledge extraction from documents", layout="wide")
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = st.query_params.get("theme") == "dark"
_mode = "dark" if st.session_state["dark_mode"] else "light"
st.markdown(
    f"<style>{T.tokens_css(_mode)}{(ROOT / 'web' / 'style.css').read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_conn():
    return S.connect(DB_PATH)


@st.cache_resource
def get_lexicon():
    return S.lexicon(get_conn())


@st.cache_resource
def get_stats():
    return S.stats(get_conn())


@st.cache_resource
def get_matcher():
    """The meaning matcher, or None if the embedding library or model is not
    available (the site then uses keyword matching only)."""
    if os.environ.get("C2K_MEANING_MATCHING", "1") == "0":
        return None
    try:
        from c2k.understand import RelationMatcher

        return RelationMatcher()
    except Exception:  # noqa: BLE001 - any failure to load just disables the extra matching
        return None


def safe_url(url: str) -> str:
    return url if url.startswith(("https://", "http://")) else ""


def link_html(title: str, url: str) -> str:
    url = safe_url(url)
    label = S.html_text(title)
    return f'<a href="{escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{label}</a>' if url else label


def render_evidence(f: dict) -> None:
    esc = S.md_escape
    notes = []
    if f["subject_canonical"] != f["subject"]:
        notes.append(f"Subject resolved to “{esc(f['subject_canonical'])}”")
    if f["object_canonical"] != f["object"]:
        notes.append(f"Object resolved to “{esc(f['object_canonical'])}”")
    if notes:
        st.caption("; ".join(notes))
    st.markdown(f'<div class="c2k-quote">{S.highlight(f)}</div>', unsafe_allow_html=True)
    badge = SOURCE_LABEL.get(f.get("source_type", ""), "Source")
    line = f'<span class="c2k-badge">{escape(badge)}</span>{link_html(f["doc_title"], f["source_url"])}'
    others = f.get("also_in") or []
    if others:
        shown = ", ".join(link_html(o["doc_title"], o["source_url"]) for o in others[:4])
        more = f" and {len(others) - 4} more" if len(others) > 4 else ""
        line += f"<br>Also found in {len(others)} other document{'s' if len(others) != 1 else ''}: {shown}{more}"
    st.markdown(f'<div class="c2k-source">{line}</div>', unsafe_allow_html=True)


def render_fact(f: dict, key: str) -> None:
    esc = S.md_escape
    with st.container(border=True):
        st.markdown(f"**{esc(f['subject'])}** &nbsp;·&nbsp; {f['relation'].replace('_', ' ')} &nbsp;·&nbsp; **{esc(f['object'])}**")
        render_evidence(f)


def render_interpretation(u, relation: str = "", entity: str = "") -> None:
    """Shows how the question was understood, so a surprising result can be explained."""
    types = [r.replace("_", " ") for r in ([relation] if relation else u.relations)]
    about = entity or u.entity
    fact_type = ("**" + " / ".join(types) + "**") if types else "not recognised"
    subject = f"“{S.md_escape(about)}”" if about else "none specified"
    st.caption(f"Interpreted as: fact type {fact_type}; name {subject}")
    if u.ignored:
        st.caption("Terms not found in the data (ignored): " + ", ".join(f"“{S.md_escape(w)}”" for w in u.ignored))
    if u.widened:
        st.caption("No facts of that type were found for this name. Showing all available facts for it.")


def render_results(extractions: list[dict], key: str, *, limit: int = SHOW_FIRST) -> None:
    facts = S.group_facts(extractions)
    n_docs = len({f["doc_id"] for f in extractions})
    st.caption(
        f"{len(facts)} distinct fact{'s' if len(facts) != 1 else ''} from {n_docs} "
        f"document{'s' if n_docs != 1 else ''} ({len(extractions)} extractions; duplicates grouped)"
    )

    sentences, left_out = S.generate_answer(facts)
    with st.container(border=True, key=f"summary-{key}"):
        st.markdown('<div class="c2k-h">Summary</div>', unsafe_allow_html=True)
        st.markdown(sentences[0] if len(sentences) == 1 else "\n".join(f"- {t}" for t in sentences))
        if left_out:
            st.caption(f"{left_out} further item{'s' if left_out != 1 else ''} not listed here; see the evidence below.")
        st.caption("Generated from the extracted facts using fixed templates. Expand an item below to view its source sentence.")

    show_all = len(facts) > limit and st.checkbox(f"Show all {len(facts)} evidence items", key=f"all-{key}")
    st.markdown('<div class="c2k-h">Supporting evidence</div>', unsafe_allow_html=True)
    for f in facts[: len(facts) if show_all else limit][:100]:
        label = f"{f['subject']} · {f['relation'].replace('_', ' ')} · {f['object']}"
        with st.expander(S.md_escape(label if len(label) <= 120 else label[:119] + "…")):
            render_evidence(f)


conn = get_conn()
info = S.meta(conn)
figures = get_stats()


def _remember_theme() -> None:
    st.query_params["theme"] = "dark" if st.session_state["dark_mode"] else "light"


# ------------------------------------------------------------------ header
head_left, head_right = st.columns([3, 2])
head_left.markdown(
    '<div class="c2k-title">C2K</div><div class="c2k-sub">Knowledge extraction from documents</div>',
    unsafe_allow_html=True,
)
with head_right:
    st.markdown(
        f'<div class="c2k-meta">Data run <code>{escape(info.get("run_id", "-"))}</code> &nbsp; '
        f'Model <code>{escape(info.get("model", "-"))}</code></div>',
        unsafe_allow_html=True,
    )
    with st.container(key="theme-toggle"):
        st.toggle("Dark mode", key="dark_mode", on_change=_remember_theme)

tab_cases, tab_search, tab_upload, tab_entities, tab_about = st.tabs(
    ["Overview", "Search", "My documents", "Entities", "Methodology"]
)

# ------------------------------------------------------------------ overview
with tab_cases:
    stats = [
        ("Documents", figures["documents"]),
        ("Extracted facts", figures["facts"]),
        ("Distinct entities", figures["entities"]),
        ("Fact types", figures["fact_types"]),
    ]
    st.markdown(
        '<div class="c2k-stats">'
        + "".join(
            f'<div class="c2k-stat"><div class="c2k-stat-label">{label}</div>'
            f'<div class="c2k-stat-value">{value:,}</div></div>'
            for label, value in stats
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    mix = ", ".join(f"{n} {SOURCE_LABEL.get(src, src)}" for src, n in sorted(figures["by_source"].items()))
    if "active_case" not in st.session_state:
        st.session_state["active_case"] = USE_CASES[0]["id"]
    active = next(uc for uc in USE_CASES if uc["id"] == st.session_state["active_case"])
    chart_col, notice_col = st.columns([7, 5], gap="large")
    with chart_col:
        st.markdown(chart_html(S.facts_by_type(conn), active.get("relation")), unsafe_allow_html=True)
    with notice_col:
        st.caption(f"Sources: {mix}. Each fact links to the sentence it was extracted from.")
        st.markdown(
            '<div class="c2k-notice"><span class="c2k-label">Reliability</span>Facts are extracted automatically. '
            "Quoted evidence is verbatim from the source document, but the fact derived from it can still be wrong. "
            "In a sample of 100 facts, an automated review found about 21% not fully supported. "
            "Check results against the cited evidence.</div>",
            unsafe_allow_html=True,
        )

    if "active_case" not in st.session_state:
        st.session_state["active_case"] = USE_CASES[0]["id"]
    list_col, result_col = st.columns([4, 7], gap="large")
    with list_col:
        st.markdown('<div class="c2k-h">Example queries</div>', unsafe_allow_html=True)
        st.caption("Select a query to run it against the extracted facts.")
        for uc in USE_CASES:
            is_active = st.session_state["active_case"] == uc["id"]
            with st.container(key=f"{'egon' if is_active else 'eg'}-{uc['id']}"):
                st.markdown(f"**{uc['title']}**")
                st.caption(uc["blurb"])
                if st.button(
                    "Selected" if is_active else "Run query",
                    key=f"uc-{uc['id']}",
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state["active_case"] = uc["id"]
                    st.rerun()
    with result_col:
        active = next(uc for uc in USE_CASES if uc["id"] == st.session_state["active_case"])
        st.markdown(f'<div class="c2k-h">{escape(active["title"])}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="c2k-note">Note: {escape(active["note"])}</div>', unsafe_allow_html=True)
        facts = S.search(
            conn,
            entity=active.get("entity"),
            relation=active.get("relation"),
            source_type=active.get("source_type"),
        )
        render_results(facts, f"uc-{active['id']}")

# ------------------------------------------------------------------ search
with tab_search:
    matcher = get_matcher()
    text = st.text_input(
        "Question",
        placeholder="Where is Apple headquartered?   Who acquired Hugging Face?   What was EGF evaluated on?",
        max_chars=S.MAX_INPUT_CHARS,
        key="q_text",
    )
    with st.expander("Filters"):
        c1, c2, c3 = st.columns(3)
        entity = c1.text_input("Name contains", max_chars=S.MAX_INPUT_CHARS, key="q_entity")
        relation = c2.selectbox("Fact type", [""] + S.relations(conn), key="q_relation")
        source = c3.selectbox("Source", ["", "arxiv", "edgar"], key="q_source")
    if matcher is None:
        st.caption("Semantic matching is unavailable; questions are interpreted by keywords only.")
    if text or entity or relation or source:
        facts, u = S.ask(
            conn, text, matcher=matcher, lex=get_lexicon(), entity=entity, relation=relation, source_type=source
        )
        if text:
            render_interpretation(u, relation, entity)
        if facts:
            render_results(facts, "search", limit=20)
        elif text and not (u.relations or u.entity):
            st.info(
                "The question could not be interpreted. Include a company, paper or method name and the information "
                "wanted, for example “Where is Apple headquartered?” or “What was EGF evaluated on?”."
            )
        else:
            st.info("No matching facts. Check the spelling of the name or remove a filter.")
    else:
        st.info("Enter a question or select a filter. Examples: “Where is Apple located?”, “Who acquired Hugging Face?”.")

# ------------------------------------------------------------------ uploads
with tab_upload:
    st.markdown('<div class="c2k-h">Add a document</div>', unsafe_allow_html=True)
    st.caption(
        "Upload a PDF to extract its facts and query them. Documents are private to this browser session."
    )
    st.markdown(
        '<div class="c2k-notice"><span class="c2k-label">Privacy</span>The document text is processed on this '
        "server and sent to the Anthropic API for extraction. It is held in memory for this session only and is "
        "discarded when the page is closed or reloaded. Do not upload confidential material.</div>",
        unsafe_allow_html=True,
    )
    if "user_conn" not in st.session_state:
        st.session_state["user_conn"] = U.new_session_db()
        st.session_state["user_docs"] = []
    user_conn = st.session_state["user_conn"]
    user_docs: list[dict] = st.session_state["user_docs"]

    server_key = os.environ.get("ANTHROPIC_API_KEY")
    api_key: str | None = None
    with st.expander("API key (optional settings)"):
        options = [OWN_KEY] + ([SITE_KEY] if server_key else [])
        mode = st.radio("API key used for extraction", options, key="u_key_mode", horizontal=True)
        if mode == OWN_KEY:
            own = st.text_input(
                "Anthropic API key",
                type="password",
                key="u_own_key",
                help="Used only for this extraction and kept in this session only; never stored. Create a key in the Anthropic Console.",
            )
            api_key = own.strip() or None
        else:
            required = os.environ.get("C2K_UPLOAD_PASSWORD")
            if required:
                entered = st.text_input("Access password for the site's API key", type="password", key="u_password")
                api_key = server_key if hmac.compare_digest(entered.encode(), required.encode()) else None
            else:
                api_key = server_key

    matcher_for_passages = get_matcher()
    uploaded = st.file_uploader("PDF document", type=["pdf"], key="u_file")
    plan = None
    if uploaded is not None:
        if st.session_state.get("u_plan_id") != uploaded.file_id:
            try:
                st.session_state["u_plan"] = U.plan_upload(uploaded.name, uploaded.getvalue())
                st.session_state["u_plan_error"] = None
            except U.UploadError as exc:
                st.session_state["u_plan"] = None
                st.session_state["u_plan_error"] = str(exc)
            st.session_state["u_plan_id"] = uploaded.file_id
        plan = st.session_state.get("u_plan")
        if st.session_state.get("u_plan_error"):
            st.error(st.session_state["u_plan_error"])

    if plan is not None:
        already = any(d["doc_id"] == plan.doc_id for d in user_docs)
        full = len(user_docs) >= U.MAX_DOCS_PER_SESSION
        st.markdown(
            f"**{S.md_escape(plan.name)}**: {plan.n_pages} page{'s' if plan.n_pages != 1 else ''}, "
            f"{plan.n_chunks} section{'s' if plan.n_chunks != 1 else ''} to process. "
            f"Estimated cost about **${plan.est_cost_usd:.2f}** (Anthropic API) and about "
            f"**{max(1, round(plan.est_seconds / 60))} minute{'s' if round(plan.est_seconds / 60) > 1 else ''}**."
        )
        if already:
            st.caption("This document has already been added.")
        elif full:
            st.caption(f"A session can hold up to {U.MAX_DOCS_PER_SESSION} documents. Remove one to add another.")
        elif api_key is None:
            st.caption("Open 'API key (optional settings)' above and provide an API key (or the access password) to continue.")
        if st.button(
            f"Extract facts (about ${plan.est_cost_usd:.2f})",
            disabled=already or full or api_key is None,
            key="u_go",
            type="primary",
        ):
            with st.status("Validating the API key…", expanded=True) as status:
                try:
                    U.check_key(api_key)
                    status.update(label="Processing the document…")
                    bar = st.progress(0.0)
                    result = U.extract_upload(
                        plan,
                        api_key,
                        progress=lambda done, total: bar.progress(done / total, text=f"Section {done} of {total}"),
                    )
                except U.UploadError as exc:
                    status.update(label="Extraction failed", state="error")
                    st.error(str(exc))
                else:
                    U.add_to_session_db(user_conn, result)
                    if matcher_for_passages is not None:
                        st.session_state.setdefault("user_index", {})[result.doc_id] = (
                            result.name,
                            P.build_index(plan.text, matcher_for_passages.embedder),
                        )
                    user_docs.append(
                        {"doc_id": result.doc_id, "name": result.name, "n_facts": len(result.rows), "cost": result.cost_usd}
                    )
                    cost = f"cost about ${result.cost_usd:.3f}" if result.cost_usd is not None else "cost unknown"
                    note = f" {result.n_failed_chunks} section(s) could not be processed." if result.n_failed_chunks else ""
                    status.update(
                        label=f"Extracted {len(result.rows)} facts ({cost}, {result.seconds:.0f} s).{note}", state="complete"
                    )

    if user_docs:
        st.markdown('<div class="c2k-h">Documents in this session</div>', unsafe_allow_html=True)
        for d in list(user_docs):
            c_name, c_btn = st.columns([5, 1])
            cost = f"; cost about ${d['cost']:.3f}" if d["cost"] is not None else ""
            c_name.markdown(f"{S.md_escape(d['name'])}: {d['n_facts']} facts{cost}")
            if c_btn.button("Remove", key=f"u_rm_{d['doc_id']}"):
                U.remove_from_session_db(user_conn, d["doc_id"])
                st.session_state.get("user_index", {}).pop(d["doc_id"], None)
                user_docs.remove(d)
                st.rerun()

        st.markdown('<div class="c2k-h">Query your documents</div>', unsafe_allow_html=True)
        question = st.text_input(
            "Question",
            placeholder="Who acquired …?   Where is … based?   What was … evaluated on?",
            max_chars=S.MAX_INPUT_CHARS,
            key="u_q",
        )
        if question:
            ufacts, uu = S.ask(user_conn, question, matcher=get_matcher(), lex=S.lexicon(user_conn))
            render_interpretation(uu)
            if ufacts:
                render_results(ufacts, "upload")
            found = []
            if matcher_for_passages is not None and st.session_state.get("user_index"):
                found = P.search(
                    {name: idx for name, idx in st.session_state["user_index"].values()},
                    question,
                    matcher_for_passages.embedder,
                )
            if found:
                st.markdown(
                    '<div class="c2k-h">Closest passages (quoted verbatim)</div>'
                    if ufacts
                    else '<div class="c2k-h">No extracted fact answers this question. Closest passages (quoted verbatim)</div>',
                    unsafe_allow_html=True,
                )
                for passage in found:
                    with st.container(border=True):
                        st.markdown(S.md_escape(passage.text))
                        st.caption(f"{S.md_escape(passage.doc_name)} · similarity {passage.score:.2f}")
            elif not ufacts:
                st.info("Nothing in your documents matches. Refer to a name or topic that appears in them.")
        if st.checkbox("Show all extracted facts", key="u_all"):
            render_results(S.search(user_conn), "upload-all")
    elif plan is None:
        st.caption("No documents added yet.")

# ------------------------------------------------------------------ entities
with tab_entities:
    names = S.entity_names(conn)
    name = st.selectbox("Entity (resolved name)", names, key="entity_pick")
    if name:
        page = S.entity_page(conn, name)
        st.markdown(f'<div class="c2k-h">{escape(page["name"])}</div>', unsafe_allow_html=True)
        if len(page["surfaces"]) > 1:
            st.caption("Written in the documents as: " + ", ".join(f"“{S.md_escape(s)}”" for s in page["surfaces"]))
        for relation_name, facts in page["by_relation"].items():
            st.markdown(f"**{relation_name.replace('_', ' ').capitalize()}** ({len(facts)})")
            for i, f in enumerate(facts[:5]):
                render_fact(f, f"ent-{relation_name}-{i}")

# ------------------------------------------------------------------ methodology
with tab_about, st.container(key="method"):
    st.markdown(
        """
### Overview
C2K reads documents and records facts as *subject, relation, object* triples, each with the exact sentence
it came from. The set of relations is fixed (14 types, such as `acquired`, `headquartered_in`,
`evaluated_on` and `outperforms`). A fact is retained only if its quoted sentence appears in the source text.

### Reliability
On a small hand-annotated test set (80 expected facts across 15 of these documents), the system recovered
about 36% of the expected facts. Precision against the annotations is only a few percent, in part because the
annotation set is sparse and many correct extractions are counted as misses. An automated reviewer, not a
person, assessed 100 extracted facts and found approximately 21% not fully supported by their evidence. A
human-labelled error rate has not been measured.

### Limitations
- The SEC filings are mostly routine Form 8-K reports, and only 10 of the arXiv papers are loaded, so many
  questions return few results.
- Some relations are applied loosely: `ceo_of` also captures other officers, and the direction of a fact is
  occasionally reversed (for example, an acquisition stated the wrong way round).
- Entity resolution can merge distinct names. The Entities tab lists the wordings that were merged.
- Always review the highlighted source sentence.

### How questions are interpreted
A question is interpreted first by keywords and then by meaning. A small language model running on this
server compares the question with descriptions of each fact type, so alternative wordings reach the same
facts. On a held-out set of unseen questions the correct fact type was among those selected about 95% of
the time; near-ties return two or three types, and unusual phrasing can still be missed. The line beginning
"Interpreted as" shows how a question was read.

### How summaries are written
Summary sentences are generated from the extracted facts using fixed templates; no language model writes them
at query time. They cannot state anything absent from the facts, but they reproduce the facts' errors, so the
supporting evidence is the authoritative part.

### Uploaded documents
The My documents tab extracts facts from a PDF using the Anthropic API, with the same procedure as the
documents above. Uploads remain in the browser session and are neither stored nor shared.
Questions that extracted facts cannot answer fall back to quoting the closest passages of the document.

### Source material
Only short excerpts are displayed, each linking to the original document.
        """
    )

st.markdown(
    '<div class="c2k-footer">C2K research prototype. Extracted facts are machine-generated and unaudited.</div>',
    unsafe_allow_html=True,
)
