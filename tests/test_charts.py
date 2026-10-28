"""Facts-by-type chart: data, geometry, escaping, placement and contrast."""

from __future__ import annotations

import sqlite3

from streamlit.testing.v1 import AppTest

from tests.test_theme import contrast
from pathlib import Path

from web import search as S
from web.charts import chart_html
from web.theme import THEMES


APP = Path(__file__).resolve().parent.parent / "web" / "app.py"


def test_counts_sorted_and_sum_to_total() -> None:
    conn = S.connect("web/demo_data/demo.sqlite")
    rows = S.facts_by_type(conn)
    assert sum(n for _, n in rows) == S.stats(conn)["facts"]
    assert len(rows) == S.stats(conn)["fact_types"]
    assert rows == sorted(rows, key=lambda r: (-r[1], r[0]))


def test_bar_widths_are_proportional_to_the_largest() -> None:
    html = chart_html([("a_b", 200), ("c", 50)])
    assert "width:100.00%" in html and "width:25.00%" in html
    assert "a b" in html


def test_highlight_marks_only_the_selected_row() -> None:
    html = chart_html([("a", 3), ("b", 2)], highlight="b")
    assert html.count("c2k-bar-on") == 1
    assert "c2k-bar-on" not in chart_html([("a", 3)], highlight=None)


def test_labels_are_escaped_and_empty_and_zero_are_safe() -> None:
    assert "<script>" not in chart_html([("<script>", 1)])
    assert "No facts" in chart_html([])
    assert "width:0.00%" in chart_html([("a", 0)])


def test_overview_shows_chart_and_highlight_follows_selection() -> None:
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    first = [m.value for m in at.markdown if 'class="c2k-chart"' in m.value]
    assert first and first[0].count('c2k-bar-row c2k-bar-on') <= 1
    buttons = [b for b in at.button if b.key.startswith("uc-") and b.label == "Run query"]
    buttons[0].click().run()
    assert not at.exception
    assert any('class="c2k-chart"' in m.value for m in at.markdown)


def test_bar_colour_is_visible_against_its_track_and_card_in_both_themes() -> None:
    for palette in THEMES.values():
        assert contrast(palette["accent"], palette["surface"]) >= 3
        assert contrast(palette["accent"], palette["surface-2"]) >= 3
        assert contrast(palette["accent"], palette["accent-soft"]) >= 3
