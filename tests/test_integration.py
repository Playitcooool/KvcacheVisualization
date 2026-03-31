# tests/test_integration.py
import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_loader import ModelLoader
from kvcache_extractor import KVCacheExtractor
from kvcache_simulator import KVCacheSimulator
from visualizer import KVCacheVisualizer


def test_full_pipeline_no_model():
    """测试完整流程（不需要真实模型加载）"""
    # 1. 初始化各组件
    extractor = KVCacheExtractor(num_layers=4, num_heads=4, head_dim=16)
    simulator = KVCacheSimulator(num_layers=4, num_heads=4, head_dim=16)
    visualizer = KVCacheVisualizer(num_layers=4, num_heads=4, head_dim=16)

    # 2. 模拟 KV Cache 数据
    for i in range(10):
        k = torch.randn(4, 4, 1, 1, 16)
        v = torch.randn(4, 4, 1, 1, 16)
        simulator.add_entry(
            position=i + 1,
            k_cache=k,
            v_cache=v,
            token_id=i,
            token_str=f"token_{i}"
        )

    # 3. 验证状态管理
    assert simulator.current_position == 10
    assert len(simulator.history) == 10

    # 4. 验证回放
    for pos in [1, 5, 10]:
        state = simulator.get_state_at_position(pos)
        assert state is not None
        assert state['position'] == pos

    # 5. 验证可视化
    fig = visualizer.create_heatmap(simulator.history[-1].k_cache)
    assert fig is not None

    fig = visualizer.create_sequence_view(
        [f"token_{i}" for i in range(10)],
        [h.k_cache for h in simulator.history]
    )
    assert fig is not None

    fig = visualizer.create_layer_view(simulator.history[-1].k_cache)
    assert fig is not None

    fig = visualizer.create_dashboard(
        [f"token_{i}" for i in range(10)],
        [h.k_cache for h in simulator.history],
        [h.v_cache for h in simulator.history],
        current_position=10
    )
    assert fig is not None

    # 6. 验证统计计算
    stats = visualizer.calculate_cache_stats(
        [h.k_cache for h in simulator.history],
        [h.v_cache for h in simulator.history]
    )
    assert stats['num_generated_tokens'] == 10
    assert stats['num_cached_tokens'] == 10
    assert stats['cache_efficiency'] > 0


def test_model_loader_factory():
    """测试 loader 工厂方法"""
    hf_loader = ModelLoader.create("huggingface", model_name="gpt2")
    assert hf_loader.loader_type == "huggingface"

    pt_loader = ModelLoader.create("pytorch", checkpoint_path="dummy.pt")
    assert pt_loader.loader_type == "pytorch"


def test_kvcache_extractor_no_hooks():
    """测试 extractor 不依赖真实模型"""
    extractor = KVCacheExtractor(num_layers=2, num_heads=2, head_dim=8)
    assert len(extractor.kvcache_history) == 0

    extractor.clear_history()
    assert len(extractor.kvcache_history) == 0

    summary = extractor.get_cache_summary()
    assert summary['num_entries'] == 0


def test_device_utils_available():
    """测试设备工具可用"""
    from device_utils import get_available_device, list_available_devices, get_device_from_string

    device = get_available_device()
    assert device is not None

    devices = list_available_devices()
    assert len(devices) > 0
    assert 'cpu' in devices

    cpu_device = get_device_from_string("cpu")
    assert cpu_device.type == 'cpu'