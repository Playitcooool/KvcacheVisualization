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
        """加载 HuggingFace 模型和 tokenizer"""
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
        self._tokenizer = None
        self._config = None

    def load(self, device: Union[str, torch.device] = "auto") -> Tuple[Any, Any, Dict]:
        """加载 PyTorch checkpoint"""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        # 解析设备
        target_device = get_device_from_string(device) if device != "auto" else self.device

        checkpoint = torch.load(self.checkpoint_path, map_location=target_device)

        # 支持 safetensors 格式
        if self.checkpoint_path.endswith('.safetensors'):
            try:
                from safetensors.torch import load_file
                state_dict = load_file(self.checkpoint_path, device=str(target_device))
            except ImportError:
                raise ImportError("请安装 safetensors: uv pip install safetensors")

        # 尝试不同的 key 格式
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # 从 state_dict 推断配置
        self._infer_config(state_dict)

        # 尝试加载为 HuggingFace 格式模型
        # 如果 checkpoint 目录包含 config.json，则是 HuggingFace 格式
        checkpoint_dir = os.path.dirname(self.checkpoint_path)
        config_path = os.path.join(checkpoint_dir, "config.json")

        if os.path.exists(config_path):
            # 是 HuggingFace 本地模型
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
            self._model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
            self._model = self._model.to(target_device)
            self._model.eval()
            self.device = target_device
            return self._model, self._tokenizer, self.get_config()

        # 纯 PyTorch checkpoint - 尝试作为 GPT-2 加载
        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            # 尝试推断是否是 GPT-2 架构
            sample_key = list(state_dict.keys())[0] if state_dict else ""

            if any(k in sample_key for k in ['attn.c_attn', 'attn.qkv_proj', 'transformer.h']):
                # 看起来是 GPT-2 架构
                self._model = GPT2LMHeadModel.from_pretrained("gpt2")
                self._model.load_state_dict(state_dict)
                self._model = self._model.to(target_device)
                self._model.eval()
                self._tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                self._tokenizer.pad_token = self._tokenizer.eos_token
                self.device = target_device
                return self._model, self._tokenizer, self.get_config()
        except Exception as e:
            print(f"Failed to load as GPT-2: {e}")

        # 无法识别格式，返回配置信息
        return None, None, self._config

    def _infer_config(self, state_dict: Dict[str, torch.Tensor]):
        """从 state_dict 推断模型配置"""
        if not state_dict:
            self._config = {
                'num_layers': 12,
                'num_heads': 12,
                'hidden_size': 768,
                'vocab_size': 50257,
            }
            return

        # 尝试从 key 模式推断层数
        layer_keys = [k for k in state_dict.keys() if '.h.' in k or '.layer.' in k]
        max_layer = 0
        for k in layer_keys:
            parts = k.split('.')
            for p in parts:
                if p.isdigit():
                    max_layer = max(max_layer, int(p))

        num_layers = max_layer + 1 if max_layer > 0 else 12

        # 尝试获取隐藏维度
        hidden_size = 768
        for k in ['attn.c_attn.weight', 'qkv_proj.weight', 'transformer.wte.weight']:
            if k in state_dict:
                hidden_size = state_dict[k].shape[0]
                break

        # 尝试获取 vocab 大小
        vocab_size = 50257
        for k in ['transformer.wte.weight', 'embed_tokens.weight']:
            if k in state_dict:
                vocab_size = state_dict[k].shape[0]
                break

        self._config = {
            'num_layers': num_layers,
            'num_heads': 12,
            'hidden_size': hidden_size,
            'vocab_size': vocab_size,
        }

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

    def get_config(self) -> Dict[str, Any]:
        if self._config is None:
            self._config = {'num_layers': 12, 'num_heads': 12}
        return self._config