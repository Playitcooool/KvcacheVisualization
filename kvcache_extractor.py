# kvcache_extractor.py
import torch
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import copy
from utils.logger import setup_logger
logger = setup_logger(__name__)

@dataclass
class KVCacheEntry:
    """单次 Forward 的 KV Cache 条目"""
    position: int
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    token_id: int
    token_str: str
    attn_weights: Optional[torch.Tensor] = None  # [batch, heads, seq, seq] attention weights


class KVCacheExtractor:
    """
    KV Cache 提取器

    通过拦截模型的 forward pass 捕获 Attention 层的 K/V tensor
    支持多种模型架构：
    - GPT-2: c_attn (QKV 合并)
    - LLaMA/Qwen/Mistral: q_proj, k_proj, v_proj (GQA 支持)
    - T5: q, k, v, o (Encoder-Decoder)
    - Bloom: 类似 GPT-2
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
        self._attention_type = None  # 'gpt2', 'gqa', 't5'

    def clear_history(self):
        """清空历史记录"""
        self.kvcache_history = []
        self.current_position = 0
        self._debug_info = []

    def _capture_kv(self, k: torch.Tensor, v: torch.Tensor, position: int, q: torch.Tensor = None):
        """捕获 K/V tensor，可选地计算并存储 attention weights"""
        if k is None or v is None:
            return

        # 如果是 3D tensor (batch, seq, hidden_kv)，需要转换
        if k.dim() == 3:
            batch, seq_len, hidden_kv = k.shape
            k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim)
            v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim)

        # 转换为 (batch, head, seq, dim)
        if k.dim() == 4 and k.shape[1] != self.num_kv_heads:
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

        # 计算 attention weights if Q is provided
        attn_weights = None
        if q is not None:
            scale = self.head_dim ** 0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) / scale
            attn_weights = torch.softmax(scores, dim=-1)

        # 克隆以防原始 tensor 被修改
        self.kvcache_history.append(KVCacheEntry(
            position=position,
            k_cache=k.clone(),
            v_cache=v.clone(),
            token_id=-1,
            token_str="",
            attn_weights=attn_weights.clone() if attn_weights is not None else None
        ))

    def _detect_attention_type(self, model: torch.nn.Module):
        """检测模型的 attention 类型"""
        # 检查模块名称模式
        has_qkv = False
        has_separate_qkv = False
        has_c_attn = False

        for name, module in model.named_modules():
            name_lower = name.lower()

            # GPT-2 style: transformer.h.0.attn.c_attn
            if 'c_attn' in name_lower or 'qkv_proj' in name_lower:
                has_c_attn = True

            # LLaMA/Qwen style: model.layers.0.self_attn.q_proj
            if '.q_proj' in name_lower or '.k_proj' in name_lower:
                has_separate_qkv = True

        if has_c_attn:
            self._attention_type = 'gpt2'
        elif has_separate_qkv:
            self._attention_type = 'gqa'
        else:
            self._attention_type = 'gqa'  # 默认

        if self.debug:
            self._debug_info.append(f"Detected attention type: {self._attention_type}")
            logger.debug(f"Detected attention type: {self._attention_type}")

    def register_hooks(self, model: torch.nn.Module) -> List[Any]:
        """
        注册 hooks 到模型的 attention 层

        Args:
            model: PyTorch 模型

        Returns:
            hook handles 列表，用于之后移除 hook
        """
        self._handles = []
        self._detect_attention_type(model)

        if self._attention_type == 'gpt2':
            self._register_gpt2_hooks(model)
        else:
            self._register_gqa_hooks(model)

        return self._handles

    def _register_gpt2_hooks(self, model: torch.nn.Module):
        """为 GPT-2 风格模型注册 hooks (c_attn)"""
        for name, module in model.named_modules():
            if 'c_attn' in name.lower():
                # GPT-2 的 c_attn 输出是 (batch, seq, hidden*3)，包含 QKV
                handle = module.register_forward_hook(self._create_c_attn_hook(name))
                self._handles.append(handle)
                if self.debug:
                    self._debug_info.append(f"Registered c_attn hook on: {name}")
                    logger.debug(f"Registered c_attn hook on: {name}")

    def _create_c_attn_hook(self, name: str):
        """创建 GPT-2 c_attn 的 hook"""
        def hook_fn(module, input, output):
            if self.debug:
                self._debug_info.append(f"c_attn {name}: {output.shape if isinstance(output, torch.Tensor) else type(output)}")
                logger.debug(f"c_attn {name}: {output.shape if isinstance(output, torch.Tensor) else type(output)}")
            if isinstance(output, torch.Tensor) and output.dim() == 3:
                # output: (batch, seq, hidden*3)
                batch, seq_len, hidden3 = output.shape
                hidden = hidden3 // 3
                q = output[:, :, 0:hidden]
                k = output[:, :, hidden:hidden*2]
                v = output[:, :, hidden*2:hidden*3]

                # 更新 position
                self.current_position += seq_len

                # GPT-2 不使用 GQA，所以 num_kv_heads = num_heads
                q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

                # 计算 attention weights: softmax(Q @ K^T / sqrt(d_k))
                scale = self.head_dim ** 0.5
                scores = torch.matmul(q, k.transpose(-2, -1)) / scale
                attn_weights = torch.softmax(scores, dim=-1)

                # 为每个 token 位置创建 entry
                for pos in range(seq_len):
                    self.kvcache_history.append(KVCacheEntry(
                        position=self.current_position - seq_len + pos + 1,
                        k_cache=k[:, :, pos:pos+1, :].clone(),
                        v_cache=v[:, :, pos:pos+1, :].clone(),
                        token_id=-1,
                        token_str="",
                        attn_weights=attn_weights[:, :, pos:pos+1, :].clone()
                    ))
        return hook_fn

    def _register_gqa_hooks(self, model: torch.nn.Module):
        """为 GQA 风格模型注册 hooks (q_proj, k_proj, v_proj)"""
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue

            if 'k_proj' in name.lower():
                handle = module.register_forward_hook(self._create_k_hook(name))
                self._handles.append(handle)

            elif 'v_proj' in name.lower():
                handle = module.register_forward_hook(self._create_v_hook(name))
                self._handles.append(handle)

            elif 'q_proj' in name.lower():
                handle = module.register_forward_hook(self._create_q_hook(name))
                self._handles.append(handle)

    def _create_q_hook(self, name: str):
        """创建 q_proj 的 hook"""
        def hook_fn(module, input, output):
            if self.debug:
                self._debug_info.append(f"q_proj {name}: {output.shape if isinstance(output, torch.Tensor) else type(output)}")
                logger.debug(f"q_proj {name}: {output.shape if isinstance(output, torch.Tensor) else type(output)}")
            if isinstance(output, torch.Tensor):
                self._temp_q = output
        return hook_fn

    def _create_k_hook(self, name: str):
        """创建 k_proj 的 hook"""
        def hook_fn(module, input, output):
            if self.debug:
                self._debug_info.append(f"k_proj {name}: {output.shape if isinstance(output, torch.Tensor) else type(output)}")
                logger.debug(f"k_proj {name}: {output.shape if isinstance(output, torch.Tensor) else type(output)}")
            if isinstance(output, torch.Tensor):
                self._temp_k = output
        return hook_fn

    def _create_v_hook(self, name: str):
        """创建 v_proj 的 hook"""
        def hook_fn(module, input, output):
            if self.debug:
                self._debug_info.append(f"v_proj {name}: {output.shape if isinstance(output, torch.Tensor) else type(output)}")
                logger.debug(f"v_proj {name}: {output.shape if isinstance(output, torch.Tensor) else type(output)}")
            if isinstance(output, torch.Tensor):
                self._temp_v = output
                # 当 v 被捕获时，说明一个 token 的处理完成了
                if hasattr(self, '_temp_k') and self._temp_k is not None:
                    self.current_position += 1
                    q = getattr(self, '_temp_q', None)
                    self._capture_kv(self._temp_k, self._temp_v, self.current_position, q)
                    self._temp_k = None
                    self._temp_q = None
        return hook_fn

    def get_cache_summary(self) -> Dict[str, Any]:
        """获取 cache 摘要信息"""
        if not self.kvcache_history:
            return {'num_entries': 0}

        return {
            'num_entries': len(self.kvcache_history),
            'total_positions': self.current_position,
            'attention_type': self._attention_type,
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
