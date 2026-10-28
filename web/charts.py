"""Facts-by-type bar chart as plain HTML and CSS (no script, no library).
Every colour comes from the theme tokens in `style.css`; every label is escaped."""

from __future__ import annotations

from html import escape


def label_for(relation: str) -> str:
    return relation.replace("_", " ")


def chart_html(rows: list[tuple[str, int]], highlight: str | None = None) -> str:
    """Ranked horizontal bars, one scale, one hue. `rows` is (relation, count)
    in display order; the `highlight` relation's row is marked. The bars are
    decorative for assistive technology: the same numbers are in a data table
    that is visually hidden, and each row has a hover title."""
    if not rows:
        return '<div class="c2k-chart c2k-chart-empty">No facts to chart.</div>'
    top = max(n for _, n in rows) or 1
    total = sum(n for _, n in rows)
    body = []
    for relation, n in rows:
        label = escape(label_for(relation))
        pct = 100 * n / top
        cls = "c2k-bar-row c2k-bar-on" if relation == highlight else "c2k-bar-row"
        body.append(
            f'<div class="{cls}" title="{label}: {n:,} facts" aria-hidden="true">'
            f'<span class="c2k-bar-label">{label}</span>'
            f'<span class="c2k-bar-track"><span class="c2k-bar-fill" style="width:{pct:.2f}%"></span></span>'
            f'<span class="c2k-bar-value">{n:,}</span></div>'
        )
    table = "".join(f"<tr><td>{escape(label_for(r))}</td><td>{n}</td></tr>" for r, n in rows)
    return (
        '<div class="c2k-chart">'
        f'<div class="c2k-h">Facts by type</div>'
        f'<div class="c2k-chart-sub">{total:,} extracted facts across {len(rows)} types</div>'
        + "".join(body)
        + f'<table class="c2k-sr-only"><caption>Facts by type</caption>'
        f"<thead><tr><th>Type</th><th>Facts</th></tr></thead><tbody>{table}</tbody></table></div>"
    )
