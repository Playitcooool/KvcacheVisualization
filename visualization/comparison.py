"""Multi-model comparison visualization components."""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional


class ComparisonVisualizer:
    """Multi-model comparison visualizer."""

    def __init__(self):
        """Initialize comparison visualizer."""
        pass

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """将 PyTorch tensor 转换为 numpy 数组"""
        if tensor.device.type != 'cpu':
            tensor = tensor.cpu()
        return tensor.detach().numpy()

    def create_split_view(
        self,
        model_data: Dict[str, Dict],
        view_type: str = "sequence",
        title: str = "Model Comparison (Split View)"
    ) -> go.Figure:
        """
        创建分屏对比视图 - 左右两个完整面板。

        Args:
            model_data: Dict of {model_id: {"tokens": [...], "k_cache_list": [...], ...}}
            view_type: 可视化类型 (sequence, layer, energy)
            title: 图表标题

        Returns:
            Plotly Figure with split layout
        """
        model_ids = list(model_data.keys())
        if len(model_ids) < 2:
            # 单模型返回空图
            return go.Figure()

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[f"{model_ids[0]}", f"{model_ids[1]}"],
            horizontal_spacing=0.1
        )

        # 获取第一个模型的数据作为参考
        m1_data = model_data[model_ids[0]]
        m2_data = model_data[model_ids[1]]

        if view_type == "sequence":
            # Token 序列视图 - 能量柱状图对比
            m1_energies = self._calculate_energies(m1_data.get("k_cache_list", []))
            m2_energies = self._calculate_energies(m2_data.get("k_cache_list", []))

            seq_len = min(len(m1_energies), len(m2_energies), 50)  # 限制显示长度

            fig.add_trace(
                go.Bar(
                    x=list(range(seq_len)),
                    y=m1_energies[:seq_len],
                    name=model_ids[0],
                    marker_color='steelblue'
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Bar(
                    x=list(range(seq_len)),
                    y=m2_energies[:seq_len],
                    name=model_ids[1],
                    marker_color='coral'
                ),
                row=1, col=2
            )

        elif view_type == "layer":
            # Layer 能量对比
            m1_k = m1_data.get("k_cache")
            m2_k = m2_data.get("k_cache")

            if m1_k is not None and m2_k is not None:
                m1_layer_energies = self._calculate_layer_means(m1_k)
                m2_layer_energies = self._calculate_layer_means(m2_k)

                layers = list(range(min(len(m1_layer_energies), len(m2_layer_energies))))

                fig.add_trace(
                    go.Scatter(
                        x=layers,
                        y=m1_layer_energies[:len(layers)],
                        name=model_ids[0],
                        mode='lines+markers',
                        marker_color='steelblue'
                    ),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=layers,
                        y=m2_layer_energies[:len(layers)],
                        name=model_ids[1],
                        mode='lines+markers',
                        marker_color='coral'
                    ),
                    row=1, col=2
                )

        fig.update_layout(
            title=dict(text=title, x=0.5),
            showlegend=False,
            height=400
        )

        return fig

    def create_overlay_view(
        self,
        model_data: Dict[str, Dict],
        view_type: str = "sequence",
        title: str = "Model Comparison (Overlay)"
    ) -> go.Figure:
        """
        创建叠加对比视图 - 同一图表叠加不同模型的曲线。

        Args:
            model_data: Dict of {model_id: {"tokens": [...], "k_cache_list": [...], ...}}
            view_type: 可视化类型 (sequence, layer, energy)
            title: 图表标题

        Returns:
            Plotly Figure with overlaid traces
        """
        colors = ['steelblue', 'coral', 'seagreen', 'mediumpurple']
        fig = go.Figure()

        for idx, (model_id, data) in enumerate(model_data.items()):
            color = colors[idx % len(colors)]

            if view_type == "sequence":
                energies = self._calculate_energies(data.get("k_cache_list", []))
                seq_len = min(len(energies), 100)

                fig.add_trace(go.Scatter(
                    x=list(range(seq_len)),
                    y=energies[:seq_len],
                    mode='lines+markers',
                    name=model_id,
                    marker=dict(color=color),
                    line=dict(color=color)
                ))

            elif view_type == "layer":
                k_cache = data.get("k_cache")
                if k_cache is not None:
                    layer_energies = self._calculate_layer_means(k_cache)

                    fig.add_trace(go.Scatter(
                        x=list(range(len(layer_energies))),
                        y=layer_energies,
                        mode='lines+markers',
                        name=model_id,
                        marker=dict(color=color),
                        line=dict(color=color)
                    ))

        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Position" if view_type == "sequence" else "Layer",
            yaxis_title="Energy",
            height=400,
            legend=dict(x=0.99, y=0.99)
        )

        return fig

    def create_stats_comparison(
        self,
        model_stats: Dict[str, Dict[str, float]],
        title: str = "Model Statistics Comparison"
    ) -> go.Figure:
        """
        创建统计对比视图 - 并排柱状图展示多模型统计指标。

        Args:
            model_stats: Dict of {model_id: {"cache_efficiency": ..., "peak_memory": ..., ...}}
            title: 图表标题

        Returns:
            Plotly Figure with comparison bar charts
        """
        model_ids = list(model_stats.keys())
        if not model_ids:
            return go.Figure()

        # 定义要比较的指标
        metrics = [
            ("Cache Efficiency", "cache_efficiency", "%"),
            ("Avg Layer Energy", "avg_layer_energy", ""),
            ("Sparsity", "sparsity", "%"),
        ]

        rows = len(metrics)
        fig = make_subplots(
            rows=rows, cols=1,
            subplot_titles=[m[0] for m in metrics],
            vertical_spacing=0.15
        )

        for row_idx, (metric_name, metric_key, unit) in enumerate(metrics, 1):
            values = []
            for model_id in model_ids:
                val = model_stats[model_id].get(metric_key, 0)
                values.append(val)

            colors = ['steelblue', 'coral', 'seagreen', 'mediumpurple'][:len(model_ids)]

            fig.add_trace(
                go.Bar(
                    x=model_ids,
                    y=values,
                    marker_color=colors,
                    text=[f"{v:.2f}{unit}" for v in values],
                    textposition='outside',
                    showlegend=False,
                    name=metric_name
                ),
                row=row_idx, col=1
            )

        fig.update_layout(
            title=dict(text=title, x=0.5),
            height=200 * rows,
            showlegend=False
        )

        return fig

    def create_layer_energy_comparison_heatmap(
        self,
        model_data: Dict[str, Dict],
        title: str = "Layer Energy Comparison"
    ) -> go.Figure:
        """
        创建层级能量对比热力图。

        Args:
            model_data: Dict of {model_id: {"k_cache_list": [...], ...}}
            title: 图表标题

        Returns:
            Plotly Figure
        """
        model_ids = list(model_data.keys())
        if not model_ids:
            return go.Figure()

        # 计算每个模型的层级能量
        max_layers = 0
        max_seq = 0
        all_energies = {}

        for model_id, data in model_data.items():
            k_list = data.get("k_cache_list", [])
            if not k_list:
                continue

            energies = np.zeros((len(k_list[0]) if k_list else 0, len(k_list)))
            for pos, k in enumerate(k_list):
                if k is not None and k.numel() > 0:
                    for layer_idx in range(min(k.shape[0], energies.shape[0])):
                        layer_k = k[layer_idx] if len(k.shape) > 3 else k
                        energies[layer_idx, pos] = torch.norm(layer_k).item()

            all_energies[model_id] = energies
            max_layers = max(max_layers, energies.shape[0])
            max_seq = max(max_seq, energies.shape[1])

        if not all_energies:
            return go.Figure()

        # 创建热力图
        fig = make_subplots(
            rows=1, cols=len(model_ids),
            subplot_titles=model_ids,
            horizontal_spacing=0.1
        )

        for col_idx, model_id in enumerate(model_ids, 1):
            energies = all_energies.get(model_id)
            if energies is None:
                continue

            # 调整大小
            if energies.shape[0] < max_layers or energies.shape[1] < max_seq:
                padded = np.zeros((max_layers, max_seq))
                padded[:energies.shape[0], :energies.shape[1]] = energies
                energies = padded

            fig.add_trace(
                go.Heatmap(
                    z=energies,
                    colorscale="Viridis",
                    showscale=(col_idx == len(model_ids)),
                    colorbar=dict(len=0.8, y=0.5)
                ),
                row=1, col=col_idx
            )

        fig.update_layout(
            title=dict(text=title, x=0.5),
            height=350,
            showlegend=False
        )

        return fig

    def _calculate_energies(self, k_cache_list: List[torch.Tensor]) -> List[float]:
        """计算每个位置的 KV energy"""
        energies = []
        for k in k_cache_list:
            if k is not None and k.numel() > 0:
                energy = torch.norm(k).item()
            else:
                energy = 0
            energies.append(energy)
        return energies

    def _calculate_layer_means(self, k_cache: torch.Tensor) -> List[float]:
        """计算每层的平均能量"""
        k_np = self._tensor_to_numpy(k_cache)
        layer_means = []

        if k_np.ndim >= 1:
            num_layers = min(k_np.shape[0], 50)  # 限制数量
            for layer_idx in range(num_layers):
                if k_np.ndim == 5:
                    layer_k = k_np[layer_idx]
                else:
                    layer_k = k_np
                layer_flat = np.reshape(layer_k, (-1,))
                layer_means.append(np.mean(np.abs(layer_flat)))
        else:
            layer_means = [0]

        return layer_means

    def create_attention_comparison(
        self,
        model_data: Dict[str, Dict],
        title: str = "Attention Pattern Comparison"
    ) -> go.Figure:
        """
        创建 Attention 对比视图。

        Args:
            model_data: Dict of {model_id: {"attn_weights": [...], "tokens": [...]}}
            title: 图表标题

        Returns:
            Plotly Figure
        """
        model_ids = list(model_data.keys())
        if len(model_ids) < 2:
            return go.Figure()

        colors = ['steelblue', 'coral']

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[f"{m}" for m in model_ids],
            horizontal_spacing=0.1
        )

        for col_idx, model_id in enumerate(model_ids, 1):
            data = model_data.get(model_id, {})
            attn_weights = data.get("attn_weights")

            if attn_weights and len(attn_weights) > 0:
                # 取最后一个位置的 attention
                attn = attn_weights[-1]
                if attn is not None:
                    attn_np = self._tensor_to_numpy(attn)

                    if attn_np.ndim == 4:
                        attn_avg = np.mean(attn_np, axis=1)  # 平均 heads
                        attn_avg = attn_avg[-1, :] if attn_avg.shape[0] > 1 else attn_avg[0]
                    else:
                        attn_avg = attn_np

                    fig.add_trace(
                        go.Bar(
                            x=list(range(len(attn_avg))),
                            y=attn_avg,
                            marker_color=colors[col_idx - 1],
                            showlegend=False
                        ),
                        row=1, col=col_idx
                    )

        fig.update_layout(
            title=dict(text=title, x=0.5),
            height=350,
            showlegend=False
        )

        return fig
