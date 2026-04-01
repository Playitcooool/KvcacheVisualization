# kvcache_simulator.py
import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# 最大历史长度限制，防止内存溢出
MAX_HISTORY_LENGTH = 100

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