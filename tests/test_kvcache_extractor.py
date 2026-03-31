# tests/test_kvcache_extractor.py
import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvcache_extractor import KVCacheExtractor

def test_extractor_initialization():
    """测试 extractor 初始化"""
    extractor = KVCacheExtractor(num_layers=12, num_heads=12, head_dim=64)
    assert extractor.num_layers == 12
    assert extractor.num_heads == 12
    assert extractor.head_dim == 64
    assert len(extractor.kvcache_history) == 0

def test_clear_history():
    """测试清空历史"""
    extractor = KVCacheExtractor(num_layers=12, num_heads=12, head_dim=64)
    extractor.kvcache_history.append(('fake', torch.randn(12, 12, 1, 1, 64)))
    assert len(extractor.kvcache_history) == 1
    extractor.clear_history()
    assert len(extractor.kvcache_history) == 0

def test_register_hooks():
    """测试 hook 注册（需要 mock 模型）"""
    import torch.nn as nn

    class DummyAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn_k_proj = nn.Linear(64, 64)
            self.attn_v_proj = nn.Linear(64, 64)

        def forward(self, x):
            return self.attn_k_proj(x), self.attn_v_proj(x)

    model = DummyAttention()
    extractor = KVCacheExtractor(num_layers=1, num_heads=1, head_dim=64)
    # 注册 hook
    handles = extractor.register_hooks(model)
    assert len(handles) > 0
    # 移除 hook
    for h in handles:
        h.remove()