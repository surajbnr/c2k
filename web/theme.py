"""Colour tokens for the two themes. Neutrals lean toward the accent (a deep
teal) so the greys look chosen, the evidence highlight is a marker yellow
because the page is about quoted evidence, and the status colours are kept
apart from the accent. Both palettes are checked for contrast; the dark one is
designed, not an inversion of the light one."""

from __future__ import annotations

LIGHT = {
    "bg": "#fbfcfc",
    "surface": "#ffffff",
    "surface-2": "#f0f4f5",
    "line": "#d5dddf",
    "ink": "#14232b",
    "muted": "#566a72",
    "accent": "#0b6b78",
    "accent-ink": "#ffffff",
    "accent-soft": "#e0eff1",
    "mark": "#ffe7a0",
    "mark-ink": "#14232b",
    "good": "#1e7b4f",
    "warn": "#8f5f00",
    "crit": "#b3261e",
}

DARK = {
    "bg": "#0d161a",
    "surface": "#132026",
    "surface-2": "#192a31",
    "line": "#2a3b42",
    "ink": "#e4edef",
    "muted": "#9bb0b7",
    "accent": "#52b6c4",
    "accent-ink": "#04222a",
    "accent-soft": "#123039",
    "mark": "#6b5413",
    "mark-ink": "#fff2c7",
    "good": "#5bc592",
    "warn": "#e0b04a",
    "crit": "#f08a82",
}

THEMES = {"light": LIGHT, "dark": DARK}


def tokens_css(mode: str) -> str:
    """A `:root` block defining every token for `mode`, plus `color-scheme` so
    native controls and scrollbars follow the palette."""
    palette = THEMES["dark" if mode == "dark" else "light"]
    declarations = "".join(f"--{name}:{value};" for name, value in palette.items())
    return f":root{{{declarations}color-scheme:{'dark' if mode == 'dark' else 'light'};}}"
