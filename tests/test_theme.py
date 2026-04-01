import pytest
from theme import get_theme, get_theme_css, get_plotly_template, THEMES

def test_get_theme_light():
    theme = get_theme("light")
    assert theme["background"] == "#ffffff"
    assert theme["text"] == "#31333F"

def test_get_theme_dark():
    theme = get_theme("dark")
    assert theme["background"] == "#0e1117"
    assert theme["text"] == "#f0f0f0"

def test_get_theme_unknown():
    theme = get_theme("unknown")
    assert theme == THEMES["light"]  # 回退到 light

def test_get_theme_css():
    css = get_theme_css("light")
    assert "#ffffff" in css
    assert "#1f77b4" in css

def test_get_plotly_template_light():
    assert get_plotly_template("light") == "plotly"

def test_get_plotly_template_dark():
    assert get_plotly_template("dark") == "plotly_dark"