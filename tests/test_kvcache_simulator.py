# tests/test_kvcache_simulator.py
import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvcache_simulator import KVCacheSimulator
from kvcache_extractor import KVCacheEntry

def test_initialization():
    sim = KVCacheSimulator(num_layers=2, num_heads=2, head_dim=8)
    assert sim.num_layers == 2
    assert sim.num_heads == 2
    assert sim.head_dim == 8
    assert sim.current_position == 0

def test_add_entry():
    sim = KVCacheSimulator(num_layers=2, num_heads=2, head_dim=8)
    k = torch.randn(2, 2, 1, 1, 8)
    v = torch.randn(2, 2, 1, 1, 8)
    sim.add_entry(1, k, v, token_id=123, token_str="Hello")
    assert sim.current_position == 1
    assert len(sim.history) == 1

def test_get_state_at_position():
    sim = KVCacheSimulator(num_layers=2, num_heads=2, head_dim=8)
    k = torch.randn(2, 2, 1, 1, 8)
    v = torch.randn(2, 2, 1, 1, 8)
    sim.add_entry(1, k, v, token_id=123, token_str="Hello")
    sim.add_entry(2, k, v, token_id=456, token_str="world")
    state = sim.get_state_at_position(2)
    assert state is not None
    assert state['position'] == 2

def test_get_state_at_invalid_position():
    sim = KVCacheSimulator(num_layers=2, num_heads=2, head_dim=8)
    state = sim.get_state_at_position(999)
    assert state is None

def test_reset():
    sim = KVCacheSimulator(num_layers=2, num_heads=2, head_dim=8)
    k = torch.randn(2, 2, 1, 1, 8)
    v = torch.randn(2, 2, 1, 1, 8)
    sim.add_entry(1, k, v)
    sim.reset()
    assert sim.current_position == 0
    assert len(sim.history) == 0