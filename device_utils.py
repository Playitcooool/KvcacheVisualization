# device_utils.py
import torch
from typing import List, Union

def get_available_device() -> torch.device:
    """
    自动检测可用设备

    优先级: CUDA > MPS > CPU

    Returns:
        torch.device: 可用的设备
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps:0")
    else:
        return torch.device("cpu")


def get_device_from_string(device_str: Union[str, torch.device]) -> torch.device:
    """
    从字符串解析设备

    Args:
        device_str: 设备字符串 ("auto", "cuda:0", "cuda:1", "mps", "mps:0", "cpu")

    Returns:
        torch.device: 对应的设备
    """
    if isinstance(device_str, torch.device):
        return device_str

    device_str = device_str.lower().strip()

    if device_str == "auto":
        return get_available_device()
    elif device_str.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device_str)
        else:
            # CUDA 不可用，降级到 CPU
            return torch.device("cpu")
    elif device_str in ("mps", "mps:0"):
        if torch.backends.mps.is_available():
            return torch.device("mps:0")
        else:
            # MPS 不可用，降级到 CPU
            return torch.device("cpu")
    else:
        return torch.device("cpu")


def list_available_devices() -> List[str]:
    """
    列出所有可用设备

    Returns:
        List[str]: 可用设备列表，如 ["cuda:0", "cuda:1"] 或 ["mps:0"] 或 ["cpu"]
    """
    devices = []

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(f"cuda:{i}")
    elif torch.backends.mps.is_available():
        devices.append("mps:0")

    devices.append("cpu")
    return devices


class DeviceManager:
    """
    设备管理器

    统一管理应用的设备选择和模型放置
    """

    def __init__(self, device: Union[str, torch.device] = "auto"):
        """
        初始化设备管理器

        Args:
            device: 设备字符串或设备对象，默认 "auto" 自动检测
        """
        self._current_device = get_device_from_string(device)

    @property
    def current_device(self) -> torch.device:
        """获取当前设备"""
        return self._current_device

    def set_device(self, device: Union[str, torch.device]):
        """设置设备"""
        self._current_device = get_device_from_string(device)

    def get_device(self) -> torch.device:
        """获取当前设备（兼容方法）"""
        return self._current_device

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """将 tensor 移动到当前设备"""
        return tensor.to(self._current_device)

    def to_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        """将 tensor 移动到 CPU（用于可视化）"""
        if tensor.device.type != 'cpu':
            return tensor.cpu()
        return tensor

    def __repr__(self) -> str:
        return f"DeviceManager(device={self._current_device})"