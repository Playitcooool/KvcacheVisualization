# tests/test_i18n.py
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from i18n import get_text, t, TRANSLATIONS

def test_get_text_zh():
    assert get_text("model_settings", "zh") == "模型设置"

def test_get_text_en():
    assert get_text("model_settings", "en") == "Model Settings"

def test_get_text_default():
    assert get_text("model_settings") == "模型设置"  # 默认中文

def test_get_text_unknown():
    assert get_text("unknown_key") == "unknown_key"  # 回退到 key

def test_t_shorthand():
    assert t("model_settings", "en") == "Model Settings"

def test_all_keys_have_translation():
    """确保所有 key 都有中英文翻译"""
    zh_keys = set(TRANSLATIONS["zh"].keys())
    en_keys = set(TRANSLATIONS["en"].keys())
    assert zh_keys == en_keys, "中英文 key 不匹配"