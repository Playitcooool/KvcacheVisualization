# KV Cache 可视化实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 构建一个 Streamlit 应用，加载真实 LLM 模型（ HuggingFace / PyTorch），通过 Hook 捕获推理过程中的 KV Cache，实时可视化四种视图

**Architecture:** 模型加载抽象层 + Hook 捕获层 + 状态管理层 + 可视化层 + Streamlit 前端

**Tech Stack:** Python, PyTorch, HuggingFace Transformers, Streamlit, Plotly

---

## 文件结构

```
KvcacheVisualization/
├── requirements.txt            # 依赖声明
├── device_utils.py             # 设备管理（CUDA/MPS/CPU 自动检测）
├── model_loader.py             # 模型加载抽象层（HuggingFace + PyTorch）
├── kvcache_extractor.py        # Hook 捕获 KV Cache
├── kvcache_simulator.py        # 状态管理和回放
├── visualizer.py               # Plotly 可视化组件
├── app.py                      # Streamlit 主应用
└── tests/
    ├── test_device_utils.py
    ├── test_model_loader.py
    ├── test_kvcache_extractor.py
    └── test_visualizer.py
```

---

## Task 1: 创建 requirements.txt

**Files:**
- Create: `requirements.txt`

- [ ] **Step 1: 创建 requirements.txt**

```
streamlit>=1.28.0
plotly>=5.18.0
torch>=2.0.0
transformers>=4.36.0
numpy>=1.24.0
accelerate>=0.25.0
```

- [ ] **Step 2: 安装依赖验证**

Run: `pip install -r requirements.txt`
Expected: 成功安装，无报错

- [ ] **Step 3: Commit**

```bash
git init && git add requirements.txt && git commit -m "chore: add requirements.txt"
```

---

## Task 2: 创建 device_utils.py（设备管理工具）

**Files:**
- Create: `device_utils.py`
- Test: `tests/test_device_utils.py`

- [ ] **Step 1: 编写测试**

```python
# tests/test_device_utils.py
import pytest
import torch
import sys
sys.path.insert(0, '..')
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
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/test_device_utils.py -v`
Expected: FAIL - ModuleNotFoundError

- [ ] **Step 3: 编写 device_utils.py**

```python
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
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/test_device_utils.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add device_utils.py tests/test_device_utils.py
git commit -m "feat: add device management utilities for CUDA/MPS/CPU support"
```

---

## Task 3: 创建 model_loader.py（模型加载抽象层）

**Files:**
- Create: `model_loader.py`
- Test: `tests/test_model_loader.py`

- [ ] **Step 1: 编写测试**

```python
# tests/test_model_loader.py
import pytest
import torch
import sys
import os
sys.path.insert(0, '..')
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
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/test_model_loader.py -v`
Expected: FAIL - ModuleNotFoundError

- [ ] **Step 3: 编写 model_loader.py（更新版，支持设备管理）**

```python
# model_loader.py
import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union
import os

from device_utils import get_device_from_string, DeviceManager, list_available_devices


class ModelLoader(ABC):
    """模型加载抽象基类"""

    @abstractmethod
    def load(self, device: Union[str, torch.device] = "auto") -> Tuple[Any, Any, Dict]:
        """
        加载模型和 tokenizer

        Args:
            device: 设备字符串或设备对象，默认 "auto" 自动检测

        Returns:
            (model, tokenizer, config_dict)
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """获取模型配置（不加载模型）"""
        pass

    @staticmethod
    def create(loader_type: str, **kwargs) -> 'ModelLoader':
        """工厂方法创建 loader"""
        if loader_type == "huggingface":
            return HuggingFaceLoader(
                model_name=kwargs.get("model_name", "gpt2")
            )
        elif loader_type == "pytorch":
            return PyTorchLoader(
                checkpoint_path=kwargs.get("checkpoint_path", "")
            )
        else:
            raise ValueError(f"Unknown loader type: {loader_type}")


class HuggingFaceLoader(ModelLoader):
    """HuggingFace 模型加载器"""

    def __init__(self, model_name: str = "gpt2", device: Union[str, torch.device] = "auto"):
        self.model_name = model_name
        self.device = get_device_from_string(device)
        self.loader_type = "huggingface"
        self._model = None
        self._tokenizer = None
        self._config = None

    def get_config(self) -> Dict[str, Any]:
        """获取模型配置（不加载模型）"""
        if self._config is not None:
            return self._config

        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(self.model_name)

        # 统一字段名
        self._config = {
            'num_layers': getattr(config, 'n_layer', getattr(config, 'num_layers', 12)),
            'num_heads': getattr(config, 'n_head', getattr(config, 'num_heads', 12)),
            'hidden_size': getattr(config, 'n_embd', getattr(config, 'hidden_size', 768)),
            'vocab_size': getattr(config, 'vocab_size', 50257),
            'max_position_embeddings': getattr(config, 'n_positions', getattr(config, 'max_position_embeddings', 1024)),
            'head_dim': getattr(config, 'n_embd', 768) // getattr(config, 'n_head', 12),
        }
        return self._config

    def load(self, device: Union[str, torch.device] = "auto") -> Tuple[Any, Any, Dict]:
        """
        加载 HuggingFace 模型和 tokenizer

        Args:
            device: 设备，默认 "auto" 使用初始化时的设备

        Returns:
            (model, tokenizer, config_dict)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # 解析设备
        target_device = get_device_from_string(device) if device != "auto" else self.device

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # 加载模型到指定设备
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
        )

        # 手动移动到目标设备（更可靠）
        self._model = self._model.to(target_device)
        self._model.eval()

        # 更新设备引用
        self.device = target_device

        config = self.get_config()
        return self._model, self._tokenizer, config

    @property
    def model(self):
        if self._model is None:
            self._model, _, _ = self.load()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            _, self._tokenizer, _ = self.load()
        return self._tokenizer

    @property
    def current_device(self) -> torch.device:
        """获取模型当前所在设备"""
        if self._model is not None:
            return next(self._model.parameters()).device
        return self.device


class PyTorchLoader(ModelLoader):
    """原生 PyTorch checkpoint 加载器"""

    def __init__(self, checkpoint_path: str, device: Union[str, torch.device] = "auto"):
        self.checkpoint_path = checkpoint_path
        self.device = get_device_from_string(device)
        self.loader_type = "pytorch"
        self._model = None
        self._config = None

    def load(self, device: Union[str, torch.device] = "auto") -> Tuple[Any, None, Dict]:
        """
        加载 PyTorch checkpoint

        Args:
            device: 设备，默认 "auto" 使用初始化时的设备

        Returns:
            (model, None, config_dict) - model 可能为 None 如果无法重建
        """
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        # 解析设备
        target_device = get_device_from_string(device) if device != "auto" else self.device

        checkpoint = torch.load(self.checkpoint_path, map_location=target_device)

        # 尝试不同的 key 格式
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # 从 state_dict 推断配置
        self._infer_config(state_dict)

        # TODO: 需要一个模型工厂来重建模型
        # 这里暂时返回 None，实际使用时需要根据 checkpoint 结构重建
        return None, None, self._config

    def _infer_config(self, state_dict: Dict[str, torch.Tensor]):
        """从 state_dict 推断模型配置"""
        # 尝试找到代表性的 key 推断配置
        # 例如: transformer.h.0.attn.k.weight -> GPT-style
        # 例如: blocks.0.attn.ka.weight -> LLaMA-style

        sample_keys = list(state_dict.keys())[:10]
        self._config = {
            'num_layers': 12,  # 默认值
            'num_heads': 12,
            'hidden_size': 768,
            'vocab_size': 50257,
        }

        # 简化处理：实际项目中需要更复杂的推断逻辑
        pass

    def get_config(self) -> Dict[str, Any]:
        if self._config is None:
            self._config = {'num_layers': 12, 'num_heads': 12}
        return self._config
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/test_model_loader.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model_loader.py tests/test_model_loader.py
git commit -m "feat: add ModelLoader with HuggingFace, PyTorch and device management"
```

---

## Task 4: 创建 kvcache_extractor.py（Hook 捕获层）

**Files:**
- Create: `kvcache_extractor.py`
- Test: `tests/test_kvcache_extractor.py`

- [ ] **Step 1: 编写测试**

```python
# tests/test_kvcache_extractor.py
import pytest
import torch
import sys
sys.path.insert(0, '..')
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
            self.k_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)

        def forward(self, x):
            return self.k_proj(x), self.v_proj(x)

    model = DummyAttention()
    extractor = KVCacheExtractor(num_layers=1, num_heads=1, head_dim=64)
    # 注册 hook
    handles = extractor.register_hooks(model)
    assert len(handles) > 0
    # 移除 hook
    for h in handles:
        h.remove()
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/test_kvcache_extractor.py -v`
Expected: FAIL - ModuleNotFoundError

- [ ] **Step 3: 编写 kvcache_extractor.py**

```python
# kvcache_extractor.py
import torch
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import copy

@dataclass
class KVCacheEntry:
    """单次 Forward 的 KV Cache 条目"""
    position: int
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    token_id: int
    token_str: str


class KVCacheExtractor:
    """
    KV Cache 提取器

    使用 PyTorch Hook 在模型 Forward 过程中捕获 Attention 层的 K/V tensor
    """

    def __init__(
        self,
        num_layers: int = 12,
        num_heads: int = 12,
        head_dim: int = 64,
        max_seq_len: int = 512
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        self.kvcache_history: List[KVCacheEntry] = []
        self.current_position = 0

        # Hook handles，用于移除 hook
        self._handles: List[Any] = []

    def clear_history(self):
        """清空历史记录"""
        self.kvcache_history = []
        self.current_position = 0

    def _hook_fn(self, module, input, output):
        """
        Forward hook 函数

        捕获 attention 层的 k, v 输出
        对于 HuggingFace 模型，attention 输出通常是 (attn_output, None, past_key_value)
        past_key_value 是一个 tuple of (k, v) tensors
        """
        try:
            # 尝试多种输出格式

            # 格式1: (k, v) tuple (常见于 GPT-2 等)
            if isinstance(output, tuple) and len(output) >= 2:
                k_output, v_output = output[0], output[1]

                # 检查是否是 (batch, seq, hidden) 格式
                if isinstance(k_output, torch.Tensor) and k_output.dim() >= 2:
                    self._capture_kv(k_output, v_output)

            # 格式2: past_key_value 格式 (常见于 Llama 等)
            elif isinstance(output, tuple) and hasattr(output[2], '__len__'):
                # output[2] 是 past_key_value
                past_kv = output[2]
                if isinstance(past_kv, tuple) and len(past_kv) >= 2:
                    k_cache, v_cache = past_kv[0], past_kv[1]
                    self._capture_kv(k_cache, v_cache)

        except Exception as e:
            # Hook 捕获失败不影响前向传播
            pass

    def _capture_kv(self, k: torch.Tensor, v: torch.Tensor):
        """捕获 K/V tensor"""
        if k is None or v is None:
            return

        # 确保是 4D tensor: (batch, head, seq, dim) 或 (batch, seq, head, dim)
        # HuggingFace 通常是 (batch, seq, head, dim)，需要转置

        if k.dim() == 3:
            # (batch, seq, hidden) -> (batch, head, seq, head_dim)
            batch, seq_len, hidden = k.shape
            k = k.view(batch, seq_len, self.num_heads, self.head_dim)
            v = v.view(batch, seq_len, self.num_heads, self.head_dim)

        if k.dim() == 4 and k.shape[1] != self.num_heads:
            # (batch, seq, head, dim) -> (batch, head, seq, dim)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

        # 克隆以防原始 tensor 被修改
        self.kvcache_history.append(KVCacheEntry(
            position=self.current_position,
            k_cache=k.clone(),
            v_cache=v.clone(),
            token_id=-1,  # 未知
            token_str=""
        ))

    def register_hooks(self, model: torch.nn.Module) -> List[Any]:
        """
        注册 hooks 到模型的所有 attention 层

        Args:
            model: PyTorch 模型

        Returns:
            hook handles 列表，用于之后移除 hook
        """
        self._handles = []

        # HuggingFace 模型结构遍历
        for name, module in model.named_modules():
            # 匹配 attention 相关的层
            if any(pattern in name.lower() for pattern in ['attn', 'attention', 'qkv_proj']):
                handle = module.register_forward_hook(self._hook_fn)
                self._handles.append(handle)

        return self._handles

    def remove_hooks(self):
        """移除所有已注册的 hooks"""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def get_state_at_position(self, position: int) -> Optional[KVCacheEntry]:
        """获取特定位置的 KV Cache 状态"""
        for entry in self.kvcache_history:
            if entry.position == position:
                return entry
        return None

    def get_full_kvcache(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取完整的 KV Cache tensor"""
        if not self.kvcache_history:
            return torch.zeros(1), torch.zeros(1)

        k_list = [entry.k_cache for entry in self.kvcache_history]
        v_list = [entry.v_cache for entry in self.kvcache_history]

        # 沿着 seq 维度拼接
        k_full = torch.cat(k_list, dim=2)  # (batch, head, total_seq, head_dim)
        v_full = torch.cat(v_list, dim=2)

        return k_full, v_full

    def update_token_info(self, position: int, token_id: int, token_str: str):
        """更新指定位置的 token 信息"""
        for entry in self.kvcache_history:
            if entry.position == position:
                entry.token_id = token_id
                entry.token_str = token_str
                break

    def get_cache_summary(self) -> Dict[str, Any]:
        """获取 cache 摘要信息"""
        if not self.kvcache_history:
            return {'num_entries': 0}

        return {
            'num_entries': len(self.kvcache_history),
            'total_positions': self.current_position,
            'device': str(self.kvcache_history[0].k_cache.device),
            'dtype': str(self.kvcache_history[0].k_cache.dtype),
        }
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/test_kvcache_extractor.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add kvcache_extractor.py tests/test_kvcache_extractor.py
git commit -m "feat: add KVCacheExtractor with PyTorch Hook support"
```

---

## Task 5: 创建 kvcache_simulator.py（状态管理和回放）

**Files:**
- Create: `kvcache_simulator.py` (更新版本，整合 extractor)
- Test: `tests/test_kvcache_simulator.py`

- [ ] **Step 1: 编写测试**

```python
# tests/test_kvcache_simulator.py
import pytest
import torch
import sys
sys.path.insert(0, '..')
from kvcache_simulator import KVCacheSimulator
from kvcache_extractor import KVCacheExtractor, KVCacheEntry

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
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/test_kvcache_simulator.py -v`
Expected: FAIL - ModuleNotFoundError

- [ ] **Step 3: 编写 kvcache_simulator.py**

```python
# kvcache_simulator.py
import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class KVCacheEntry:
    """KV Cache 历史条目"""
    position: int
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    token_id: int
    token_str: str


class KVCacheSimulator:
    """
    KV Cache 状态管理器和回放器

    管理 KV Cache 历史记录，支持回放到任意位置
    与 KVCacheExtractor 配合使用，存储捕获的数据
    """

    def __init__(
        self,
        num_layers: int = 12,
        num_heads: int = 12,
        head_dim: int = 64,
        max_seq_len: int = 512
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # 历史记录
        self.history: List[KVCacheEntry] = []
        self.current_position = 0
        self.tokens: List[str] = []

    def add_entry(
        self,
        position: int,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        token_id: Optional[int] = None,
        token_str: Optional[str] = ""
    ):
        """添加一个新的 KV Cache 条目"""
        if position <= self.current_position:
            # 覆盖已有位置或跳过
            return

        entry = KVCacheEntry(
            position=position,
            k_cache=k_cache.clone(),
            v_cache=v_cache.clone(),
            token_id=token_id or -1,
            token_str=token_str or ""
        )
        self.history.append(entry)
        self.current_position = position

    def add_token(self, token_str: str, token_id: Optional[int] = None):
        """添加 token 到序列"""
        self.tokens.append(token_str)
        # 如果历史记录数量和 token 数量同步，说明还没有为这个 token 添加 KV Cache
        # 在下一个 add_entry 时会关联

    def get_state_at_position(self, position: int) -> Optional[Dict]:
        """获取特定位置的 KV Cache 状态"""
        if position < 1 or position > len(self.history):
            return None

        idx = position - 1  # 0-indexed
        entry = self.history[idx]

        return {
            'position': entry.position,
            'k_cache': entry.k_cache,
            'v_cache': entry.v_cache,
            'token_id': entry.token_id,
            'token_str': entry.token_str,
            'total_positions': len(self.history)
        }

    def get_full_kvcache(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取完整的 KV Cache tensor"""
        if not self.history:
            return torch.zeros(1), torch.zeros(1)

        k_list = [entry.k_cache for entry in self.history]
        v_list = [entry.v_cache for entry in self.history]

        # 沿着 seq 维度拼接
        k_full = torch.cat(k_list, dim=2)  # (batch, head, total_seq, head_dim)
        v_full = torch.cat(v_list, dim=2)

        return k_full, v_full

    def get_cache_by_layer(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取特定层的 KV Cache"""
        if not self.history or layer_idx >= len(self.history[0].k_cache):
            return torch.zeros(1), torch.zeros(1)

        k_layer = torch.cat([entry.k_cache[layer_idx] for entry in self.history], dim=2)
        v_layer = torch.cat([entry.v_cache[layer_idx] for entry in self.history], dim=2)

        return k_layer, v_layer

    def get_energy_by_position(self) -> List[float]:
        """获取每个位置的能量（L2 norm），用于可视化"""
        energies = []
        for entry in self.history:
            k = entry.k_cache
            # 计算能量：mean over all dims except batch and head
            energy = torch.mean(torch.norm(k, dim=-1)).item()
            energies.append(energy)
        return energies

    def get_layer_stats(self) -> Dict[int, Dict[str, float]]:
        """获取每层的统计信息"""
        stats = {}
        for layer_idx in range(min(self.num_layers, len(self.history[0].k_cache) if self.history else 0)):
            k_layer = torch.cat([entry.k_cache[layer_idx] for entry in self.history], dim=2)
            stats[layer_idx] = {
                'mean': torch.mean(k_layer).item(),
                'std': torch.std(k_layer).item(),
                'max': torch.max(k_layer).item(),
                'min': torch.min(k_layer).item(),
            }
        return stats

    def reset(self):
        """重置所有状态"""
        self.history = []
        self.current_position = 0
        self.tokens = []

    def get_summary(self) -> Dict:
        """获取摘要信息"""
        return {
            'num_entries': len(self.history),
            'current_position': self.current_position,
            'num_tokens': len(self.tokens),
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
        }
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/test_kvcache_simulator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add kvcache_simulator.py tests/test_kvcache_simulator.py
git commit -m "feat: add KVCacheSimulator for state management and playback"
```

---

## Task 6: 创建 visualizer.py（Plotly 可视化）

**Files:**
- Create: `visualizer.py`
- Test: `tests/test_visualizer.py`

- [ ] **Step 1: 编写测试**

```python
# tests/test_visualizer.py
import pytest
import torch
import sys
sys.path.insert(0, '..')
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
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/test_visualizer.py -v`
Expected: FAIL - ModuleNotFoundError

- [ ] **Step 3: 编写 visualizer.py**

```python
# visualizer.py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple

class KVCacheVisualizer:
    """KV Cache 可视化器，使用 Plotly 渲染"""

    def __init__(
        self,
        num_layers: int = 12,
        num_heads: int = 12,
        head_dim: int = 64,
        color_scheme: str = "Viridis"
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.color_scheme = color_scheme

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """将 PyTorch tensor 转换为 numpy 数组"""
        if tensor.device.type != 'cpu':
            tensor = tensor.cpu()
        return tensor.detach().numpy()

    def create_heatmap(
        self,
        k_cache: torch.Tensor,
        v_cache: Optional[torch.Tensor] = None,
        title: str = "KV Cache Matrix"
    ) -> go.Figure:
        """
        创建 KV Cache 热力图

        Args:
            k_cache: K cache tensor, shape: (num_layers, num_heads, batch, seq_len, head_dim)
            v_cache: V cache tensor (optional)
            title: 图表标题

        Returns:
            Plotly Figure 对象
        """
        k_np = self._tensor_to_numpy(k_cache)
        # Shape: (num_layers, num_heads, batch, seq_len, head_dim)

        # 计算每个位置的"能量"（L2 norm）
        if k_np.ndim == 5:
            # (num_layers, num_heads, batch, seq_len, head_dim)
            energy = np.linalg.norm(k_np, axis=-1)  # (num_layers, num_heads, batch, seq_len)
            # 压缩 batch 维度
            energy = np.mean(energy, axis=2)  # (num_layers, num_heads, seq_len)
        else:
            energy = k_np

        # 取所有层的平均得到 2D 热力图
        heatmap_data = np.mean(energy, axis=0)  # (num_heads, seq_len)

        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            colorscale=self.color_scheme,
            colorbar=dict(title="Energy (L2 norm)")
        ))

        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Sequence Position",
            yaxis_title="Attention Head",
            width=500,
            height=400
        )

        return fig

    def create_sequence_view(
        self,
        tokens: List[str],
        k_cache_list: List[torch.Tensor],
        title: str = "Token Sequence Generation"
    ) -> go.Figure:
        """
        创建序列生成视图 - 沿时间轴展示 token 生成过程

        Args:
            tokens: 生成的 token 列表
            k_cache_list: 每个位置的 K cache 列表
            title: 图表标题

        Returns:
            Plotly Figure 对象
        """
        seq_len = len(tokens)
        if seq_len == 0:
            return go.Figure()

        # 计算每个位置的能量
        energies = []
        for k_cache in k_cache_list:
            k_np = self._tensor_to_numpy(k_cache)
            if k_np.ndim == 5:
                e = np.mean(np.linalg.norm(k_np, axis=-1))
            else:
                e = np.mean(np.linalg.norm(k_np, axis=-1))
            energies.append(e)

        # 归一化能量用于颜色映射
        energies = np.array(energies)
        if energies.max() > energies.min():
            norm_energies = (energies - energies.min()) / (energies.max() - energies.min())
        else:
            norm_energies = np.zeros_like(energies)

        # 创建 bar chart
        fig = go.Figure()

        for i, (token, energy, norm_e) in enumerate(zip(tokens, energies, norm_energies)):
            # 颜色从蓝色到红色
            hue = int(240 - norm_e * 200)  # 240=blue, 40=red
            color = f"hsl({hue}, 80%, 50%)"

            fig.add_trace(go.Bar(
                x=[i + 1],
                y=[energy],
                marker_color=color,
                text=token,
                textposition='outside',
                showlegend=False,
                width=0.6
            ))

        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Token Position",
            yaxis_title="Energy (L2 norm)",
            width=700,
            height=400,
            showlegend=False,
            barmode='group'
        )

        fig.update_xaxes(tickmode='linear', tick0=1, dtick=1)

        return fig

    def create_layer_view(
        self,
        k_cache: torch.Tensor,
        title: str = "Layer-wise KV Distribution"
    ) -> go.Figure:
        """
        创建层级分布视图 - 展示不同层的注意力分布差异

        Args:
            k_cache: K cache tensor, shape: (num_layers, num_heads, batch, seq_len, head_dim)
            title: 图表标题

        Returns:
            Plotly Figure 对象
        """
        k_np = self._tensor_to_numpy(k_cache)

        # 计算每层的统计量
        layer_means = []
        layer_stds = []
        layer_maxs = []

        num_layers_actual = min(k_np.shape[0], self.num_layers) if k_np.ndim >= 1 else 1

        for layer_idx in range(num_layers_actual):
            if k_np.ndim == 5:
                layer_k = k_np[layer_idx]  # (num_heads, batch, seq_len, head_dim)
            else:
                layer_k = k_np

            layer_flat = np.reshape(layer_k, (-1,))
            layer_means.append(np.mean(layer_flat))
            layer_stds.append(np.std(layer_flat))
            layer_maxs.append(np.max(np.abs(layer_flat)))

        x = list(range(num_layers_actual))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x,
            y=layer_means,
            mode='lines+markers',
            name='Mean',
            error_y=dict(
                type='data',
                symmetric=True,
                array=layer_stds,
                visible=True
            ),
            line=dict(color='steelblue'),
            marker=dict(size=8)
        ))

        fig.add_trace(go.Bar(
            x=x,
            y=layer_maxs,
            name='Max Abs',
            opacity=0.4,
            marker_color='lightsteelblue'
        ))

        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Layer",
            yaxis_title="Value",
            width=500,
            height=400,
            legend=dict(x=0.99, y=0.99)
        )

        return fig

    def create_dashboard(
        self,
        tokens: List[str],
        k_cache_list: List[torch.Tensor],
        v_cache_list: List[torch.Tensor],
        current_position: int,
        title: str = "KV Cache Dashboard"
    ) -> go.Figure:
        """
        创建综合仪表盘 - 同时展示多种视图

        Args:
            tokens: token 列表
            k_cache_list: K cache 历史
            v_cache_list: V cache 历史
            current_position: 当前 token 位置
            title: 图表标题

        Returns:
            Plotly Figure 对象
        """
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Matrix Heatmap",
                "Sequence View",
                "Layer Distribution",
                "Current Token Info"
            ),
            specs=[
                [{"type": "heatmap"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "table"}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        # 1. 热力图 (top-left)
        if current_position > 0 and current_position <= len(k_cache_list):
            k_cache = k_cache_list[current_position - 1]
            k_np = self._tensor_to_numpy(k_cache)

            if k_np.ndim == 5:
                energy = np.mean(np.linalg.norm(k_np, axis=-1), axis=0)  # (seq_len, head_dim)
                # 取最后几个位置显示
                seq_subset = energy[-min(10, energy.shape[0]):, :]
            else:
                seq_subset = k_np

            fig.add_trace(
                go.Heatmap(z=seq_subset, colorscale=self.color_scheme, showscale=False),
                row=1, col=1
            )

        # 2. 序列视图 (top-right)
        if k_cache_list:
            energies = []
            for k_cache in k_cache_list[:current_position]:
                k_np = self._tensor_to_numpy(k_cache)
                if k_np.ndim == 5:
                    e = np.mean(np.linalg.norm(k_np, axis=-1))
                else:
                    e = np.mean(np.linalg.norm(k_np, axis=-1))
                energies.append(e)

            fig.add_trace(
                go.Bar(
                    x=list(range(1, len(energies) + 1)),
                    y=energies,
                    marker_color='steelblue',
                    showlegend=False
                ),
                row=1, col=2
            )

        # 3. 层级分布 (bottom-left)
        if current_position > 0 and current_position <= len(k_cache_list):
            k_cache = k_cache_list[current_position - 1]
            k_np = self._tensor_to_numpy(k_cache)

            if k_np.ndim == 5:
                layer_means = [np.mean(k_np[l]) for l in range(min(k_np.shape[0], 12))]
            else:
                layer_means = [np.mean(k_np)]

            fig.add_trace(
                go.Scatter(
                    x=list(range(len(layer_means))),
                    y=layer_means,
                    mode='lines+markers',
                    showlegend=False,
                    line=dict(color='steelblue')
                ),
                row=2, col=1
            )

        # 4. 当前 Token 信息 (bottom-right)
        current_token = tokens[current_position - 1] if current_position > 0 and current_position <= len(tokens) else "N/A"
        token_info = [
            ["Current Token", current_token],
            ["Position", f"{current_position}"],
            ["Total Tokens", f"{len(tokens)}"],
            ["Num Layers", f"{self.num_layers}"],
            ["Num Heads", f"{self.num_heads}"],
            ["Head Dim", f"{self.head_dim}"],
        ]

        fig.add_trace(
            go.Table(
                cells=dict(
                    values=token_info,
                    fill_color='lavender',
                    align='left'
                )
            ),
            row=2, col=2
        )

        fig.update_layout(
            title=dict(text=title, x=0.5),
            width=900,
            height=700,
            showlegend=False
        )

        return fig

    def calculate_cache_stats(
        self,
        k_cache_list: List[torch.Tensor],
        v_cache_list: List[torch.Tensor]
    ) -> Dict[str, float]:
        """
        计算 KV Cache 统计数据

        Args:
            k_cache_list: K cache 历史列表
            v_cache_list: V cache 历史列表

        Returns:
            Dict containing cache statistics
        """
        stats = {
            'num_generated_tokens': len(k_cache_list),
            'num_cached_tokens': len(k_cache_list),
            'cache_efficiency': 0.0,
            'peak_memory_mb': 0.0,
            'avg_layer_energy': 0.0,
            'sparsity': 0.0,
        }

        if not k_cache_list:
            return stats

        # 计算缓存效率
        # 效率 = 2 / (n + 1)，n 是缓存的 token 数
        # 这是理论最大值，实际场景中 KV Cache 避免了大量重复计算
        n = len(k_cache_list)
        if n > 0:
            # 理论计算节省: sum(1..n-1) / (n * (n-1)) = (n-1)/(2n) * 100%
            stats['cache_efficiency'] = round(200 / (n + 1), 2)  # 百分比

        # 计算峰值内存
        total_bytes = 0
        for k, v in zip(k_cache_list, v_cache_list):
            k_np = self._tensor_to_numpy(k)
            v_np = self._tensor_to_numpy(v)
            total_bytes += k_np.nbytes + v_np.nbytes

        stats['peak_memory_mb'] = round(total_bytes / (1024 * 1024), 2)

        # 计算平均层能量
        all_energies = []
        for k_cache in k_cache_list:
            k_np = self._tensor_to_numpy(k_cache)
            if k_np.ndim == 5:
                energy = np.mean(np.linalg.norm(k_np, axis=-1))
            else:
                energy = np.mean(np.linalg.norm(k_np, axis=-1))
            all_energies.append(energy)

        stats['avg_layer_energy'] = round(np.mean(all_energies), 4)

        # 计算稀疏度 (接近零的值比例)
        threshold = 0.01
        all_values = []
        for k, v in zip(k_cache_list, v_cache_list):
            k_np = self._tensor_to_numpy(k).flatten()
            v_np = self._tensor_to_numpy(v).flatten()
            all_values.extend(list(np.abs(k_np)) + list(np.abs(v_np)))

        all_values = np.array(all_values)
        stats['sparsity'] = round(np.mean(np.abs(all_values) < threshold) * 100, 2)

        return stats

    def create_stats_gauge(
        self,
        stats: Dict[str, float],
        title: str = "KV Cache Statistics"
    ) -> go.Figure:
        """
        创建统计数据展示仪表盘

        Args:
            stats: calculate_cache_stats 返回的统计数据
            title: 图表标题

        Returns:
            Plotly Figure 对象
        """
        fig = go.Figure()

        # 创建指标卡片
        metrics = [
            ("Tokens Generated", stats['num_generated_tokens'], ""),
            ("Cached Tokens", stats['num_cached_tokens'], ""),
            ("Cache Efficiency", stats['cache_efficiency'], "%"),
            ("Peak Memory", stats['peak_memory_mb'], " MB"),
            ("Avg Layer Energy", stats['avg_layer_energy'], ""),
            ("Sparsity", stats['sparsity'], "%"),
        ]

        # 使用 Table 展示统计数据
        table_data = [
            ["指标", "数值"],
            ["生成 Token 数", f"{stats['num_generated_tokens']}"],
            ["缓存 Token 数", f"{stats['num_cached_tokens']}"],
            ["Cache 效率", f"{stats['cache_efficiency']}%"],
            ["峰值内存", f"{stats['peak_memory_mb']} MB"],
            ["平均层能量", f"{stats['avg_layer_energy']}"],
            ["Attention 稀疏度", f"{stats['sparsity']}%"],
        ]

        fig.add_trace(go.Table(
            header=dict(
                values=table_data[0],
                fill_color='paleturquoise',
                align='left',
                font=dict(size=14)
            ),
            cells=dict(
                values=[row[1] for row in table_data[1:]],
                fill_color='lavender',
                align='left',
                font=dict(size=12)
            )
        ))

        fig.update_layout(
            title=dict(text=title, x=0.5),
            width=400,
            height=300,
            showlegend=False
        )

        return fig
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/test_visualizer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add visualizer.py tests/test_visualizer.py
git commit -m "feat: add KVCacheVisualizer with statistics calculation"
```

---

## Task 7: 创建 app.py (Streamlit 主应用)

**Files:**
- Create: `app.py`

- [ ] **Step 1: 编写 app.py**

```python
# app.py
import streamlit as st
import torch
from typing import Optional, Tuple

from model_loader import ModelLoader, HuggingFaceLoader
from kvcache_extractor import KVCacheExtractor
from kvcache_simulator import KVCacheSimulator
from visualizer import KVCacheVisualizer

# 页面配置
st.set_page_config(
    page_title="KV Cache 可视化器",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 自定义 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .token-display {
        font-family: 'Courier New', monospace;
        background-color: #f0f0f0;
        padding: 1rem;
        border-radius: 8px;
        font-size: 1.1rem;
        min-height: 60px;
    }
    .stButton>button {
        width: 100%;
        height: 3rem;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """初始化 session state"""
    defaults = {
        'model_loader': None,
        'model': None,
        'tokenizer': None,
        'extractor': None,
        'simulator': None,
        'visualizer': None,
        'tokens': [],
        'token_ids': [],
        'current_position': 0,
        'generation_complete': False,
        'is_generating': False,
        'model_loaded': False,
        'model_config': {},
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_model(model_type: str, model_name: str = "gpt2", checkpoint_path: str = ""):
    """加载模型"""
    try:
        if model_type == "huggingface":
            loader = ModelLoader.create("huggingface", model_name=model_name)
            model, tokenizer, config = loader.load()
        else:
            loader = ModelLoader.create("pytorch", checkpoint_path=checkpoint_path)
            model, tokenizer, config = loader.load()

        st.session_state.model_loader = loader
        st.session_state.model = model
        st.session_state.tokenizer = tokenizer
        st.session_state.model_config = config
        st.session_state.model_loaded = True

        # 初始化 extractor 和 simulator
        num_layers = config.get('num_layers', 12)
        num_heads = config.get('num_heads', 12)
        head_dim = config.get('head_dim', 64)

        st.session_state.extractor = KVCacheExtractor(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim
        )
        st.session_state.simulator = KVCacheSimulator(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim
        )
        st.session_state.visualizer = KVCacheVisualizer(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim
        )

        return True, "模型加载成功"
    except Exception as e:
        return False, f"模型加载失败: {str(e)}"


def run_generation_step(prompt: str, max_new_tokens: int = 50):
    """执行一步生成（生成一批 token）"""
    if st.session_state.generation_complete or st.session_state.is_generating:
        return

    st.session_state.is_generating = True

    try:
        model = st.session_state.model
        tokenizer = st.session_state.tokenizer
        extractor = st.session_state.extractor
        simulator = st.session_state.simulator

        # 清空之前的历史
        extractor.clear_history()
        simulator.reset()
        st.session_state.tokens = []
        st.session_state.token_ids = []

        # 编码 prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        input_length = input_ids.shape[1]

        # 注册 hooks
        handles = extractor.register_hooks(model)

        # 生成
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        # 移除 hooks
        for handle in handles:
            handle.remove()

        # 解码生成的 token
        generated_ids = output[0][input_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids)

        # 更新 simulator
        for i, (token_id, token_str) in enumerate(zip(generated_ids.tolist(), generated_tokens)):
            pos = i + 1
            # 从 extractor 历史获取 KV Cache
            if pos <= len(extractor.kvcache_history):
                entry = extractor.kvcache_history[pos - 1]
                simulator.add_entry(pos, entry.k_cache, entry.v_cache, token_id, token_str)
            else:
                # 如果没有捕获到，创建空的
                k = torch.zeros(1, simulator.num_heads, 1, simulator.head_dim)
                v = torch.zeros(1, simulator.num_heads, 1, simulator.head_dim)
                simulator.add_entry(pos, k, v, token_id, token_str)

        st.session_state.tokens = generated_tokens
        st.session_state.token_ids = generated_ids.tolist()
        st.session_state.current_position = len(generated_tokens)
        st.session_state.generation_complete = True

    except Exception as e:
        st.error(f"生成失败: {str(e)}")
    finally:
        st.session_state.is_generating = False


def reset_simulation():
    """重置模拟状态"""
    st.session_state.tokens = []
    st.session_state.token_ids = []
    st.session_state.current_position = 0
    st.session_state.generation_complete = False
    if st.session_state.simulator:
        st.session_state.simulator.reset()
    if st.session_state.extractor:
        st.session_state.extractor.clear_history()


# 初始化
init_session_state()

# 主界面
st.markdown('<h1 class="main-header">KV Cache 可视化器</h1>', unsafe_allow_html=True)

# 侧边栏：模型选择
with st.sidebar:
    st.markdown("### ⚙️ 模型设置")

    model_type = st.selectbox(
        "模型类型",
        ["huggingface", "pytorch"],
        format_func=lambda x: "HuggingFace" if x == "huggingface" else "PyTorch Checkpoint"
    )

    if model_type == "huggingface":
        model_name = st.text_input("模型名称", value="gpt2")
        checkpoint_path = ""
    else:
        model_name = "custom"
        checkpoint_path = st.text_input("Checkpoint 路径", value="")

    if st.button("🚀 加载模型"):
        with st.spinner("加载中..."):
            if model_type == "huggingface":
                success, msg = load_model(model_type, model_name=model_name)
            else:
                success, msg = load_model(model_type, checkpoint_path=checkpoint_path)
            if success:
                st.success(msg)
            else:
                st.error(msg)
                st.session_state.model_loaded = False

    if st.session_state.model_loaded:
        st.markdown("---")
        st.markdown("**当前模型配置:**")
        config = st.session_state.model_config
        st.markdown(f"- 层数: {config.get('num_layers', 'N/A')}")
        st.markdown(f"- 头数: {config.get('num_heads', 'N/A')}")
        st.markdown(f"- 隐藏维度: {config.get('hidden_size', 'N/A')}")

    st.markdown("---")
    st.markdown("**说明:**")
    st.markdown("""
    1. 选择模型类型并加载
    2. 输入 Prompt 并开始生成
    3. 观察 KV Cache 可视化
    4. 生成完成后可拖动回放
    """)

# 主内容区
if not st.session_state.model_loaded:
    st.info("👈 请先在侧边栏加载模型")
else:
    # 左右分栏布局
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### LLM 生成区域")
        prompt = st.text_input("输入 Prompt", value="Hello, how are you?")

        # 批量大小
        batch_size = st.slider("每批生成 Token 数", 1, 20, 5)

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if not st.session_state.generation_complete:
                if st.button("▶ 开始生成" if not st.session_state.is_generating else "⏳ 生成中..."):
                    run_generation_step(prompt, max_new_tokens=20)
                    st.rerun()
            else:
                st.success("✓ 生成完成")
        with col_btn2:
            if st.button("🔄 重新生成"):
                reset_simulation()
                st.rerun()

        st.markdown("**生成结果:**")
        if st.session_state.tokens:
            tokens_display = " ".join(st.session_state.tokens[:st.session_state.current_position])
            st.markdown(f'<div class="token-display">{tokens_display}</div>', unsafe_allow_html=True)
        else:
            st.info("点击「开始生成」按钮启动")

        # 进度
        if st.session_state.tokens:
            progress = st.session_state.current_position / max(len(st.session_state.tokens), 1)
            st.progress(progress, text=f"Token {st.session_state.current_position}/{len(st.session_state.tokens)}")

        # 回放控制
        if st.session_state.generation_complete and st.session_state.simulator:
            st.markdown("---")
            st.markdown("### 🎚️ 回放控制")
            max_pos = len(st.session_state.simulator.history)
            slider_pos = st.slider(
                "拖动进度条回看任意位置",
                min_value=1,
                max_value=max_pos,
                value=max_pos
            )
            st.session_state.current_position = slider_pos

            if 1 <= slider_pos <= len(st.session_state.tokens):
                st.markdown(f"**回放位置:** Token {slider_pos} = \"{st.session_state.tokens[slider_pos-1]}\"")

    with col_right:
        st.markdown("### KV Cache 可视化区域")

        if not st.session_state.simulator or not st.session_state.simulator.history:
            st.info("先生成 token 后才能查看 KV Cache 可视化")
        else:
            # Tab 选择
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 矩阵热力图", "📈 序列视图", "🔳 层级分布", "📐 统计数据", "🖥️ 综合仪表盘"])

            k_cache_list = [h.k_cache for h in st.session_state.simulator.history]
            v_cache_list = [h.v_cache for h in st.session_state.simulator.history]

            # 计算统计数据
            stats = st.session_state.visualizer.calculate_cache_stats(
                k_cache_list[:st.session_state.current_position],
                v_cache_list[:st.session_state.current_position]
            )

            with tab1:
                if st.session_state.current_position > 0:
                    k_cache = st.session_state.simulator.history[st.session_state.current_position - 1].k_cache
                    fig = st.session_state.visualizer.create_heatmap(
                        k_cache,
                        title=f"KV Cache 热力图 (Token {st.session_state.current_position})"
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with tab2:
                fig = st.session_state.visualizer.create_sequence_view(
                    st.session_state.tokens[:st.session_state.current_position],
                    k_cache_list[:st.session_state.current_position],
                    title="Token 序列生成视图"
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab3:
                if st.session_state.current_position > 0:
                    k_cache = st.session_state.simulator.history[st.session_state.current_position - 1].k_cache
                    fig = st.session_state.visualizer.create_layer_view(
                        k_cache,
                        title=f"层级注意力分布 (Token {st.session_state.current_position})"
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with tab4:
                # 统计数据展示
                col_stat1, col_stat2 = st.columns(2)
                with col_stat1:
                    st.metric("生成 Token 数", stats['num_generated_tokens'])
                    st.metric("缓存 Token 数", stats['num_cached_tokens'])
                    st.metric("Cache 效率", f"{stats['cache_efficiency']}%")
                with col_stat2:
                    st.metric("峰值内存", f"{stats['peak_memory_mb']} MB")
                    st.metric("平均层能量", f"{stats['avg_layer_energy']:.4f}")
                    st.metric("Attention 稀疏度", f"{stats['sparsity']}%")

                # 详细统计表格
                st.markdown("#### 详细统计")
                fig_stats = st.session_state.visualizer.create_stats_gauge(stats)
                st.plotly_chart(fig_stats, use_container_width=True)

            with tab5:
                fig = st.session_state.visualizer.create_dashboard(
                    st.session_state.tokens,
                    k_cache_list,
                    v_cache_list,
                    st.session_state.current_position,
                    title="KV Cache 综合仪表盘"
                )
                st.plotly_chart(fig, use_container_width=True)

# 底部信息
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>"
    "KV Cache 可视化器 | PyTorch + Transformers + Streamlit + Plotly"
    "</div>",
    unsafe_allow_html=True
)
```

- [ ] **Step 2: 运行 Streamlit 验证启动**

Run: `streamlit run app.py --server.headless true --server.port 8501`
Expected: 应用启动，无报错（可能提示需要安装模型）

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add Streamlit app with real model loading and KV cache visualization"
```

---

## Task 8: 最终测试验证

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: 编写集成测试**

```python
# tests/test_integration.py
import pytest
import torch
import sys
sys.path.insert(0, '..')

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
```

- [ ] **Step 2: 运行集成测试**

Run: `pytest tests/test_integration.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for full pipeline"
```

---

## 验收检查清单

- [ ] `streamlit run app.py` 能正常启动
- [ ] 侧边栏可选择 HuggingFace 模型
- [ ] 侧边栏可选择设备（auto/CUDA/MPS/CPU）
- [ ] 自动检测可用设备正确（CUDA > MPS > CPU）
- [ ] 点击"加载模型"后，GPT-2 能成功加载到指定设备
- [ ] 输入 Prompt 后点击"开始生成"，token 正常生成
- [ ] 右侧五种 Tab 视图正常工作（矩阵/序列/层级/统计/仪表盘）
- [ ] 统计数据展示正确：生成Token数、缓存Token数、Cache效率、峰值内存、稀疏度
- [ ] 生成完成后，进度条可拖动回放
- [ ] PyTorch checkpoint 加载路径可用（虽然实际 checkpoint 需要用户自己提供）
- [ ] KV Cache 数据在 GPU 时能正确转到 CPU 用于可视化
