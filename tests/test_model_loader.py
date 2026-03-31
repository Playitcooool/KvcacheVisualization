# tests/test_model_loader.py
import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_loader import ModelLoader, HuggingFaceLoader, PyTorchLoader

def test_huggingface_loader_initialization():
    """测试 HuggingFace loader 可以初始化（不实际下载模型）"""
    loader = HuggingFaceLoader(model_name="gpt2")
    assert loader.model_name == "gpt2"
    assert loader.loader_type == "huggingface"

def test_pytorch_loader_initialization():
    """测试 PyTorch loader 可以初始化"""
    loader = PyTorchLoader(checkpoint_path="dummy.pt")
    assert loader.checkpoint_path == "dummy.pt"
    assert loader.loader_type == "pytorch"

def test_model_loader_factory():
    """测试工厂方法"""
    hf_loader = ModelLoader.create("huggingface", model_name="gpt2")
    assert hf_loader.loader_type == "huggingface"

    pt_loader = ModelLoader.create("pytorch", checkpoint_path="model.pt")
    assert pt_loader.loader_type == "pytorch"

def test_get_model_config():
    """测试获取模型配置"""
    loader = HuggingFaceLoader(model_name="gpt2")
    # 不加载模型，只获取配置
    config = loader.get_config()
    assert 'num_layers' in config or 'n_layer' in config