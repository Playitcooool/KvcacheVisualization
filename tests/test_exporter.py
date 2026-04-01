# tests/test_exporter.py
import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exporter import (
    export_kvcache_to_dict,
    export_to_json,
    export_to_csv,
)

def test_export_kvcache_to_dict():
    tokens = ["Hello", "world"]
    k_cache_list = [torch.randn(1, 12, 1, 64), torch.randn(1, 12, 2, 64)]
    v_cache_list = [torch.randn(1, 12, 1, 64), torch.randn(1, 12, 2, 64)]
    stats = {"num_tokens": 2}

    result = export_kvcache_to_dict(tokens, k_cache_list, v_cache_list, stats)

    assert result["tokens"] == tokens
    assert result["token_count"] == 2
    assert len(result["k_cache_summary"]) == 2
    assert result["k_cache_summary"][0]["position"] == 1

def test_export_to_json():
    tokens = ["Hello"]
    k_cache_list = [torch.randn(1, 12, 1, 64)]
    v_cache_list = [torch.randn(1, 12, 1, 64)]
    stats = {"num_tokens": 1}

    json_str = export_to_json(tokens, k_cache_list, v_cache_list, stats)

    assert isinstance(json_str, str)
    assert "Hello" in json_str
    assert '"position": 1' in json_str

def test_export_to_csv():
    tokens = ["Hello", "world"]
    k_cache_list = [torch.randn(1, 12, 1, 64), torch.randn(1, 12, 2, 64)]
    v_cache_list = [torch.randn(1, 12, 1, 64), torch.randn(1, 12, 2, 64)]
    stats = {"num_tokens": 2}

    csv_str = export_to_csv(tokens, k_cache_list, v_cache_list, stats)

    assert "position,token,k_shape" in csv_str
    assert "Hello" in csv_str
    assert "world" in csv_str