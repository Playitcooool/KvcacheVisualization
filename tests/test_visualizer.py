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

def test_calculate_attention_stats_by_layer():
    viz = KVCacheVisualizer(num_layers=2, num_heads=2, head_dim=8)
    # attn_weights: [batch=1, heads=2, seq=3, seq=3]
    attn = torch.softmax(torch.randn(1, 2, 3, 3), dim=-1)
    attn_list = [attn, attn, attn]
    stats = viz.calculate_attention_stats_by_layer(attn_list)
    assert isinstance(stats, dict)
    assert len(stats) == 2  # 2 layers
    assert 'coverage' in stats[0]
    assert 'sparsity' in stats[0]
    assert 'max_val' in stats[0]

def test_create_attention_layer_stats():
    viz = KVCacheVisualizer(num_layers=2, num_heads=2, head_dim=8)
    stats = {
        0: {'coverage': 0.5, 'sparsity': 0.3, 'max_val': 0.8},
        1: {'coverage': 0.6, 'sparsity': 0.2, 'max_val': 0.9},
    }
    fig = viz.create_attention_layer_stats(stats, metric='coverage')
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1