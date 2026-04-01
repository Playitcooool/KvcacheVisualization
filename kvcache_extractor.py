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

    通过拦截模型的 forward pass 捕获 Attention 层的 K/V tensor
    支持 GQA (Grouped Query Attention) - Qwen, LLaMA 等模型使用
    """

    def __init__(
        self,
        num_layers: int = 12,
        num_heads: int = 12,
        num_kv_heads: int = None,
        head_dim: int = 64,
        max_seq_len: int = 512,
        debug: bool = False
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads  # Query heads
        self.num_kv_heads = num_kv_heads if num_kv_heads else num_heads  # KV heads (for GQA)
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

    def _capture_kv(self, k: torch.Tensor, v: torch.Tensor, position: int):
        """捕获 K/V tensor"""
        if k is None or v is None:
            return

        # 如果是 3D tensor (batch, seq, hidden_kv)，需要转换
        # GQA: hidden_kv = num_kv_heads * head_dim
        if k.dim() == 3:
            batch, seq_len, hidden_kv = k.shape
            k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim)
            v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim)

        # 转换为 (batch, head, seq, dim)
        if k.dim() == 4 and k.shape[1] != self.num_kv_heads:
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

        # 克隆以防原始 tensor 被修改
        self.kvcache_history.append(KVCacheEntry(
            position=position,
            k_cache=k.clone(),
            v_cache=v.clone(),
            token_id=-1,
            token_str=""
        ))

    def register_hooks(self, model: torch.nn.Module) -> List[Any]:
        """
        注册 hooks 到模型的 attention 层

        Args:
            model: PyTorch 模型

        Returns:
            hook handles 列表，用于之后移除 hook
        """
        self._handles = []

        # 遍历模型找到 k_proj 和 v_proj (GQA 架构)
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue

            # Hook k_proj 和 v_proj
            if 'k_proj' in name.lower():
                handle = module.register_forward_hook(self._create_k_hook(name))
                self._handles.append(handle)

            elif 'v_proj' in name.lower():
                handle = module.register_forward_hook(self._create_v_hook(name))
                self._handles.append(handle)

        return self._handles

    def _create_k_hook(self, name: str):
        """创建 k_proj 的 hook"""
        def hook_fn(module, input, output):
            if self.debug:
                self._debug_info.append(f"k_proj {name}: {output.shape if isinstance(output, torch.Tensor) else type(output)}")
            if isinstance(output, torch.Tensor):
                self._temp_k = output
        return hook_fn

    def _create_v_hook(self, name: str):
        """创建 v_proj 的 hook"""
        def hook_fn(module, input, output):
            if self.debug:
                self._debug_info.append(f"v_proj {name}: {output.shape if isinstance(output, torch.Tensor) else type(output)}")
            if isinstance(output, torch.Tensor):
                self._temp_v = output
                # 当 v 被捕获时，说明一个 token 的处理完成了
                if hasattr(self, '_temp_k') and self._temp_k is not None:
                    self.current_position += 1
                    self._capture_kv(self._temp_k, self._temp_v, self.current_position)
                    self._temp_k = None
        return hook_fn

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
