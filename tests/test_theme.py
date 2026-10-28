"""The two colour themes: same tokens, readable contrast, and a working toggle."""

from __future__ import annotations

from pathlib import Path

import pytest

from web.theme import DARK, LIGHT, THEMES, tokens_css

ROOT = Path(__file__).resolve().parent.parent


def _luminance(hex_colour: str) -> float:
    r, g, b = (int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5))
    lin = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4  # noqa: E731
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def contrast(a: str, b: str) -> float:
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def test_both_themes_define_the_same_tokens() -> None:
    assert set(LIGHT) == set(DARK)


@pytest.mark.parametrize("mode", ["light", "dark"])
@pytest.mark.parametrize(
    "fg, bg",
    [
        ("ink", "bg"), ("ink", "surface"), ("ink", "surface-2"), ("ink", "accent-soft"),
        ("muted", "bg"), ("muted", "surface"), ("muted", "surface-2"),
        ("accent", "bg"), ("accent", "surface"),
        ("accent-ink", "accent"), ("mark-ink", "mark"),
    ],
)
def test_text_colours_meet_the_contrast_minimum_on_their_backgrounds(mode, fg, bg) -> None:
    palette = THEMES[mode]
    assert contrast(palette[fg], palette[bg]) >= 4.5, f"{mode}: {fg} on {bg}"


def test_the_stylesheet_uses_only_tokens_for_colour() -> None:
    import re

    css = (ROOT / "web" / "style.css").read_text()
    literal = re.findall(r"#[0-9a-fA-F]{3,8}\b|rgba?\(", css)
    assert literal == [], f"colour literals in style.css: {literal}"
    used = set(re.findall(r"var\(--([a-z0-9-]+)\)", css))
    assert used <= set(LIGHT), f"undefined tokens: {used - set(LIGHT)}"


def test_tokens_css_sets_color_scheme_so_native_controls_follow() -> None:
    assert "color-scheme:dark" in tokens_css("dark") and "color-scheme:light" in tokens_css("light")
    assert f"--bg:{DARK['bg']}" in tokens_css("dark") and f"--bg:{LIGHT['bg']}" in tokens_css("light")


# ---- the toggle on the page (needs the exported demo data)

BUNDLE = ROOT / "web" / "demo_data" / "demo.sqlite"
needs_bundle = pytest.mark.skipif(not BUNDLE.exists(), reason="run scripts/export_demo.py first")


def _style_text(at) -> str:
    return " ".join(m.value for m in at.markdown if "<style>" in m.value)


@needs_bundle
def test_the_page_starts_light_and_the_toggle_switches_to_dark_and_back() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "web" / "app.py"), default_timeout=90).run()
    assert not at.exception and f"--bg:{LIGHT['bg']}" in _style_text(at)
    assert at.toggle(key="dark_mode").value is False

    at.toggle(key="dark_mode").set_value(True).run()
    assert not at.exception and f"--bg:{DARK['bg']}" in _style_text(at)
    assert at.query_params.get("theme") == "dark" or at.query_params.get("theme") == ["dark"]

    at.toggle(key="dark_mode").set_value(False).run()
    assert f"--bg:{LIGHT['bg']}" in _style_text(at)


@needs_bundle
def test_a_link_with_theme_dark_opens_in_dark_mode() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "web" / "app.py"), default_timeout=90)
    at.query_params["theme"] = "dark"
    at.run()
    assert not at.exception and f"--bg:{DARK['bg']}" in _style_text(at)
    assert at.toggle(key="dark_mode").value is True
