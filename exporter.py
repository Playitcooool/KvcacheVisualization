# exporter.py
import json
import pandas as pd
from typing import List, Any, Dict
import torch

def export_kvcache_to_dict(
    tokens: List[str],
    k_cache_list: List[torch.Tensor],
    v_cache_list: List[torch.Tensor],
    stats: Dict[str, Any]
) -> Dict[str, Any]:
    """导出 KV Cache 数据为字典"""
    return {
        "tokens": tokens,
        "token_count": len(tokens),
        "k_cache_summary": [
            {
                "position": i + 1,
                "shape": list(k.shape),
                "l2_norm": torch.norm(k).item() if k.numel() > 0 else 0,
            }
            for i, k in enumerate(k_cache_list)
        ],
        "v_cache_summary": [
            {
                "position": i + 1,
                "shape": list(v.shape),
                "l2_norm": torch.norm(v).item() if v.numel() > 0 else 0,
            }
            for i, v in enumerate(v_cache_list)
        ],
        "statistics": stats,
    }

def export_to_json(
    tokens: List[str],
    k_cache_list: List[torch.Tensor],
    v_cache_list: List[torch.Tensor],
    stats: Dict[str, Any]
) -> str:
    """导出 KV Cache 数据为 JSON 字符串"""
    data = export_kvcache_to_dict(tokens, k_cache_list, v_cache_list, stats)
    return json.dumps(data, indent=2, ensure_ascii=False)

def export_to_csv(
    tokens: List[str],
    k_cache_list: List[torch.Tensor],
    v_cache_list: List[torch.Tensor],
    stats: Dict[str, Any]
) -> str:
    """导出 KV Cache 数据为 CSV 字符串"""
    rows = []
    for i, (token, k, v) in enumerate(zip(tokens, k_cache_list, v_cache_list)):
        rows.append({
            "position": i + 1,
            "token": token,
            "k_shape": str(list(k.shape)),
            "v_shape": str(list(v.shape)),
            "k_l2_norm": torch.norm(k).item() if k.numel() > 0 else 0,
            "v_l2_norm": torch.norm(v).item() if v.numel() > 0 else 0,
        })
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)

def download_file(content: str, filename: str, mime_type: str) -> bytes:
    """将内容转换为下载格式"""
    return content.encode("utf-8")