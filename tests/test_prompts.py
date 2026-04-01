import pytest
from prompts import (
    PROMPT_TEMPLATES,
    get_template_names,
    get_template,
    fill_template,
    render_template_ui,
)

def test_get_template_names_zh():
    names = get_template_names("zh")
    assert "问答" in names
    assert "翻译" in names

def test_get_template_names_en():
    names = get_template_names("en")
    assert "Q&A" in names
    assert "Translation" in names

def test_get_template():
    template = get_template("问答", "zh")
    assert template is not None
    assert "{question}" in template.template

def test_fill_template():
    template = get_template("问答", "zh")
    result = fill_template(template, question="What is AI?")
    assert "What is AI?" in result
    assert "问题：What is AI?" in result

def test_render_template_ui_zh():
    template = get_template("代码补全", "zh")
    ui = render_template_ui(template, "zh")
    assert ui["name"] == "代码补全"
    assert "function_name" in ui["template"]

def test_render_template_ui_en():
    template = get_template("Code Completion", "en")
    ui = render_template_ui(template, "en")
    assert ui["name"] == "Code Completion"