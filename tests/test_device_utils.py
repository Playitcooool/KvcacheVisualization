# tests/test_device_utils.py
import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from device_utils import get_available_device, get_device_from_string, DeviceManager

def test_get_available_device():
    """测试自动设备检测"""
    device = get_available_device()
    assert isinstance(device, torch.device)
    assert device.type in ['cuda', 'mps', 'cpu']

def test_get_device_from_string_auto():
    """测试 auto 设备选择"""
    device = get_device_from_string("auto")
    assert isinstance(device, torch.device)

def test_get_device_from_string_cuda():
    """测试 CUDA 设备"""
    if torch.cuda.is_available():
        device = get_device_from_string("cuda:0")
        assert device.type == 'cuda'

def test_get_device_from_string_mps():
    """测试 MPS 设备"""
    if torch.backends.mps.is_available():
        device = get_device_from_string("mps")
        assert device.type == 'mps'

def test_get_device_from_string_cpu():
    """测试 CPU 设备"""
    device = get_device_from_string("cpu")
    assert device.type == 'cpu'

def test_device_manager():
    """测试 DeviceManager 类"""
    dm = DeviceManager()
    device = dm.get_device()
    assert isinstance(device, torch.device)

    # 手动设置设备
    dm.set_device("cpu")
    assert dm.current_device.type == 'cpu'