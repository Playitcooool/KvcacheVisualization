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
        max_seq_len: int = 512,
        debug: bool = False
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.debug = debug

        self.kvcache_history: List[KVCacheEntry] = []
        self.current_position = 0

        # Hook handles，用于移除 hook
        self._handles: List[Any] = []
        self._debug_info: List[str] = []

    def clear_history(self):
        """清空历史记录"""
        self.kvcache_history = []
        self.current_position = 0
        self._debug_info = []

    def _hook_fn(self, module, input, output):
        """
        Forward hook 函数

        捕获 attention 层的 k, v 输出
        对于 HuggingFace 模型，attention 输出通常是 (attn_output, None, past_key_value)
        past_key_value 是一个 tuple of (k, v) tensors
        """
        try:
            if self.debug:
                self._debug_info.append(f"Output type: {type(output)}, len: {len(output) if isinstance(output, tuple) else 'N/A'}")

            # 尝试多种输出格式

            # 格式1: (k, v) tuple (常见于 GPT-2 等)
            if isinstance(output, tuple) and len(output) >= 2:
                k_output, v_output = output[0], output[1]

                # 检查是否是 (batch, seq, hidden) 格式
                if isinstance(k_output, torch.Tensor) and k_output.dim() >= 2:
                    if self.debug:
                        self._debug_info.append(f"Format 1: k={k_output.shape}, v={v_output.shape if isinstance(v_output, torch.Tensor) else 'N/A'}")
                    self._capture_kv(k_output, v_output)
                    return

            # 格式2: past_key_value 格式 (常见于 Llama 等)
            # output = (attn_output, None, past_key_value) 或
            # output = (attn_output, past_key_value)
            if isinstance(output, tuple):
                for i, item in enumerate(output):
                    if isinstance(item, tuple) and len(item) == 2:
                        # 这看起来像 (k, v)
                        k_cache, v_cache = item[0], item[1]
                        if isinstance(k_cache, torch.Tensor) and isinstance(v_cache, torch.Tensor):
                            if self.debug:
                                self._debug_info.append(f"Format 2: k={k_cache.shape}, v={v_cache.shape}")
                            self._capture_kv(k_cache, v_cache)
                            return

        except Exception as e:
            if self.debug:
                self._debug_info.append(f"Hook error: {e}")
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

        # 尝试多种 attention 层命名模式
        patterns = [
            'attn', 'attention', 'qkv_proj', 'self_attn',
            'h.', 'layer.',  # 常见 transformer 层前缀
        ]

        # HuggingFace 模型结构遍历
        for name, module in model.named_modules():
            # 匹配 attention 相关的层
            if any(pattern in name.lower() for pattern in patterns):
                # 跳过非 nn.Module 的东西
                if not isinstance(module, torch.nn.Module):
                    continue
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

    def detect_model_architecture(self, model) -> str:
        """检测模型架构类型"""
        model_name_lower = model.__class__.__name__.lower()

        if "t5" in model_name_lower or "flan" in model_name_lower:
            return "encoder-decoder"  # T5, FLAN-T5, mT5
        elif "gpt" in model_name_lower or "llama" in model_name_lower or "qwen" in model_name_lower or "bloom" in model_name_lower:
            return "causal"  # GPT, LLaMA, Qwen, Bloom
        elif "opt" in model_name_lower:
            return "causal"  # OPT is causal
        else:
            return "causal"  # 默认

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

    def get_debug_info(self) -> List[str]:
        """获取调试信息"""
        return self._debug_info

    @staticmethod
    def print_model_attn_modules(model: torch.nn.Module) -> List[str]:
        """打印模型中所有可能的 attention 模块名称（用于调试）"""
        attn_modules = []
        for name, module in model.named_modules():
            if any(p in name.lower() for p in ['attn', 'attention', 'qkv', 'self_attn', 'h.', 'layer.', 'blocks']):
                attn_modules.append(f"{name}: {module.__class__.__name__}")
        return attn_modules