# model_loader.py
import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union
import os

from device_utils import get_device_from_string, DeviceManager, list_available_devices

# 本地模型缓存目录
LOCAL_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)

# 获取默认的 HuggingFace 缓存目录
HF_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")


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


def check_quantization_available():
    """检查可用的量化库"""
    libs = {
        'bitsandbytes': False,
        'auto_gptq': False,
        'awq': False,
    }

    try:
        import bitsandbytes
        libs['bitsandbytes'] = True
    except ImportError:
        pass

    try:
        import auto_gptq
        libs['auto_gptq'] = True
    except ImportError:
        pass

    try:
        import awq
        libs['awq'] = True
    except ImportError:
        pass

    return libs


class HuggingFaceLoader(ModelLoader):
    """HuggingFace 模型加载器，支持自动量化检测"""

    def __init__(self, model_name: str = "gpt2", device: Union[str, torch.device] = "auto"):
        self.model_name = model_name
        self.device = get_device_from_string(device)
        self.loader_type = "huggingface"
        self._model = None
        self._tokenizer = None
        self._config = None
        self._quantization_info = None

    def _get_cache_dir(self) -> Optional[str]:
        """获取缓存目录：优先本地models/，模型不存在则使用默认缓存"""
        import os

        # 如果是本地路径，不需要指定cache_dir
        if os.path.isdir(self.model_name):
            return None

        # 检查模型是否已存在于本地models/目录
        # HuggingFace会把模型下载到 cache_dir/models/ 下
        local_model_path = os.path.join(LOCAL_MODELS_DIR, self.model_name.replace("/", "--"))
        if os.path.exists(local_model_path):
            return LOCAL_MODELS_DIR

        # 模型不在本地，返回None使用默认缓存
        return None

    def get_config(self) -> Dict[str, Any]:
        """获取模型配置（不加载模型）"""
        if self._config is not None:
            return self._config

        from transformers import AutoConfig
        cache_dir = self._get_cache_dir()
        config = AutoConfig.from_pretrained(
            self.model_name,
            cache_dir=cache_dir
        )

        # 检测架构类型
        model_type = getattr(config, 'model_type', '')

        if model_type in ['t5', 'mt5', 'flan-t5']:
            # T5 系列
            num_layers = getattr(config, 'num_layers', 12)
            num_heads = getattr(config, 'num_heads', 12)
            hidden_size = getattr(config, 'd_model', 768)
            architecture = "encoder-decoder"
            num_kv_heads = num_heads  # T5 没有 GQA
        else:
            # GPT/LLaMA/Qwen 系列 (支持 GQA)
            num_layers = getattr(config, 'n_layer', getattr(config, 'num_layers', 12))
            num_heads = getattr(config, 'n_head', getattr(config, 'num_heads', 12))
            # GQA: num_key_value_heads 可能不同于 num_heads
            num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
            hidden_size = getattr(config, 'n_embd', getattr(config, 'hidden_size', 768))
            architecture = "causal"

        self._config = {
            'num_layers': num_layers,
            'num_heads': num_heads,
            'num_kv_heads': num_kv_heads,
            'hidden_size': hidden_size,
            'vocab_size': getattr(config, 'vocab_size', 50257),
            'max_position_embeddings': getattr(config, 'n_positions', getattr(config, 'max_position_embeddings', 1024)),
            'head_dim': hidden_size // num_heads if num_heads > 0 else 64,
            'architecture': architecture,
        }
        if self._quantization_info:
            self._config['quantization'] = self._quantization_info
        return self._config

    def _detect_quantization(self) -> Optional[str]:
        """自动检测模型量化格式"""
        from transformers import AutoConfig
        import os

        model_path = self.model_name

        # 如果是本地路径，检测目录内容
        if os.path.isdir(model_path):
            files = os.listdir(model_path)
            files_lower = [f.lower() for f in files]

            # 检测量化格式
            if any('4bit' in f or '-4bit' in f for f in files_lower):
                return "4bit"
            if any('8bit' in f or '-8bit' in f for f in files_lower):
                return "8bit"
            if any('gptq' in f for f in files_lower):
                return "gptq"
            if any('awq' in f for f in files_lower):
                return "awq"

            # 检查 config.json 中的量化配置
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path) as f:
                    config = json.load(f)
                quant_config = config.get("quantization_config", {})
                if quant_config:
                    load_in_4bit = quant_config.get("load_in_4bit", False)
                    load_in_8bit = quant_config.get("load_in_8bit", False)
                    if load_in_4bit:
                        return "4bit"
                    if load_in_8bit:
                        return "8bit"

        elif isinstance(model_path, str) and not os.path.exists(model_path):
            # 在线模型名检测
            name_lower = model_path.lower()
            if "4bit" in name_lower or "-4bit" in name_lower:
                return "4bit"
            if "8bit" in name_lower or "-8bit" in name_lower:
                return "8bit"
            if "gptq" in name_lower:
                return "gptq"
            if "awq" in name_lower:
                return "awq"

        return None

    def load(self, device: Union[str, torch.device] = "auto") -> Tuple[Any, Any, Dict]:
        """加载 HuggingFace 模型和 tokenizer，自动检测量化"""
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

        # 解析设备
        target_device = get_device_from_string(device) if device != "auto" else self.device
        if target_device.type == 'cuda':
            target_device = torch.device("cuda:0")

        # 获取缓存目录
        cache_dir = self._get_cache_dir()

        # 获取配置以确定模型架构
        config = self.get_config()
        architecture = config.get('architecture', 'causal')

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=cache_dir
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # 自动检测量化
        quantization = self._detect_quantization()

        # 根据架构选择模型类
        if architecture == "encoder-decoder":
            model_class = AutoModelForSeq2SeqLM
        else:
            model_class = AutoModelForCausalLM

        # 加载模型
        if quantization:
            self._model = self._load_quantized_model(quantization, target_device, model_class)
        else:
            # 普通加载
            self._model = model_class.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                cache_dir=cache_dir,
            )
            self._model = self._model.to(target_device)

        self._model.eval()
        self.device = target_device

        return self._model, self._tokenizer, config

    def _load_quantized_model(self, quantization: str, device: torch.device, model_class: Any = None) -> Any:
        """加载量化模型"""
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

        if model_class is None:
            model_class = AutoModelForCausalLM

        quant_libs = check_quantization_available()

        # 4-bit/8-bit 量化 (bitsandbytes)
        if quantization in ("4bit", "8bit"):
            if not quant_libs['bitsandbytes']:
                raise ImportError(
                    f"模型需要 bitsandbytes 量化支持，但未安装。\n"
                    f"请运行: uv pip install bitsandbytes"
                )

            if quantization == "4bit":
                load_in_4bit, load_in_8bit = True, False
                quant_desc = "4-bit"
            else:
                load_in_4bit, load_in_8bit = False, True
                quant_desc = "8-bit"

            self._quantization_info = f"{quant_desc} (bitsandbytes)"

            model = model_class.from_pretrained(
                self.model_name,
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                device_map="auto",
            )
            return model

        # GPTQ 量化
        elif quantization == "gptq":
            if not quant_libs['auto_gptq']:
                raise ImportError(
                    f"模型需要 auto-gptq 量化支持，但未安装。\n"
                    f"请运行: uv pip install auto-gptq"
                )

            self._quantization_info = "GPTQ"

            from auto_gptq import AutoGPTQForCausalLM
            model = AutoGPTQForCausalLM.from_quantized(
                self.model_name,
                device_map="auto",
                use_safetensors=True,
            )
            return model

        # AWQ 量化
        elif quantization == "awq":
            if not quant_libs['awq']:
                raise ImportError(
                    f"模型需要 awq 量化支持，但未安装。\n"
                    f"请运行: uv pip install awq"
                )

            self._quantization_info = "AWQ"

            from awq import AutoAWQForCausalLM
            model = AutoAWQForCausalLM.from_quantized(
                self.model_name,
                device_map="auto",
                use_safetensors=True,
            )
            return model

        else:
            raise ValueError(f"不支持的量化方式: {quantization}")

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
            try:
                return next(self._model.parameters()).device
            except StopIteration:
                return self.device
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