# tests/test_visualizer.py
import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualizer import KVCacheVisualizer
import plotly.graph_objects as go

def test_initialization():
    viz = KVCacheVisualizer(num_layers=2, num_heads=2, head_dim=8)
    assert viz.num_layers == 2
    assert viz.num_heads == 2
    assert viz.head_dim == 8

def test_create_heatmap():
    viz = KVCacheVisualizer(num_layers=2, num_heads=2, head_dim=8)
    k_cache = torch.randn(2, 2, 1, 5, 8)
    fig = viz.create_heatmap(k_cache, title="Test Heatmap")
    assert isinstance(fig, go.Figure)

def test_create_sequence_view():
    viz = KVCacheVisualizer(num_layers=2, num_heads=2, head_dim=8)
    tokens = ["Hello", "world", "!"]
    k_cache_list = [torch.randn(2, 2, 1, 1, 8) for _ in tokens]
    fig = viz.create_sequence_view(tokens, k_cache_list)
    assert isinstance(fig, go.Figure)

def test_create_layer_view():
    viz = KVCacheVisualizer(num_layers=2, num_heads=2, head_dim=8)
    k_cache = torch.randn(2, 2, 1, 5, 8)
    fig = viz.create_layer_view(k_cache, title="Test Layer View")
    assert isinstance(fig, go.Figure)

def test_create_layer_energy_evolution():
    viz = KVCacheVisualizer(num_layers=2, num_heads=2, head_dim=8)
    # k_cache_list: 3 positions, each with 2 layers
    k_cache_list = [torch.randn(1, 2, 1, 1, 8) for _ in range(3)]
    fig = viz.create_layer_energy_evolution(k_cache_list)
    assert isinstance(fig, go.Figure)
    # Should have 2 lines (one per layer)
    assert len(fig.data) == 2