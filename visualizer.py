# visualizer.py
import plotly.graph_objects as go
import plotly.express as px
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
        k_np = self._tensor_to_numpy(k_cache)
        if k_np.ndim == 5:
            energy = np.linalg.norm(k_np, axis=-1)
            energy = np.mean(energy, axis=2)
        else:
            energy = k_np
        heatmap_data = np.mean(energy, axis=0)
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
        seq_len = len(tokens)
        if seq_len == 0:
            return go.Figure()
        energies = []
        for k_cache in k_cache_list:
            k_np = self._tensor_to_numpy(k_cache)
            if k_np.ndim == 5:
                e = np.mean(np.linalg.norm(k_np, axis=-1))
            else:
                e = np.mean(np.linalg.norm(k_np, axis=-1))
            energies.append(e)
        energies = np.array(energies)
        if energies.max() > energies.min():
            norm_energies = (energies - energies.min()) / (energies.max() - energies.min())
        else:
            norm_energies = np.zeros_like(energies)
        fig = go.Figure()
        for i, (token, energy, norm_e) in enumerate(zip(tokens, energies, norm_energies)):
            hue = int(240 - norm_e * 200)
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
        k_np = self._tensor_to_numpy(k_cache)
        layer_means = []
        layer_stds = []
        layer_maxs = []
        num_layers_actual = min(k_np.shape[0], self.num_layers) if k_np.ndim >= 1 else 1
        for layer_idx in range(num_layers_actual):
            if k_np.ndim == 5:
                layer_k = k_np[layer_idx]
            else:
                layer_k = k_np
            layer_flat = np.reshape(layer_k, (-1,))
            layer_means.append(np.mean(layer_flat))
            layer_stds.append(np.std(layer_flat))
            layer_maxs.append(np.max(np.abs(layer_flat)))
        x = list(range(num_layers_actual))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=layer_means, mode='lines+markers', name='Mean',
            error_y=dict(type='data', symmetric=True, array=layer_stds, visible=True),
            line=dict(color='steelblue'), marker=dict(size=8)
        ))
        fig.add_trace(go.Bar(x=x, y=layer_maxs, name='Max Abs', opacity=0.4, marker_color='lightsteelblue'))
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Layer", yaxis_title="Value",
            width=500, height=400, legend=dict(x=0.99, y=0.99)
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
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Matrix Heatmap", "Sequence View", "Layer Distribution", "Current Token Info"),
            specs=[[{"type": "heatmap"}, {"type": "bar"}], [{"type": "scatter"}, {"type": "table"}]],
            vertical_spacing=0.15, horizontal_spacing=0.1
        )
        if current_position > 0 and current_position <= len(k_cache_list):
            k_cache = k_cache_list[current_position - 1]
            k_np = self._tensor_to_numpy(k_cache)
            if k_np.ndim == 5:
                energy = np.mean(np.linalg.norm(k_np, axis=-1), axis=0)
                seq_subset = energy[-min(10, energy.shape[0]):, :]
            else:
                seq_subset = k_np
            fig.add_trace(go.Heatmap(z=seq_subset, colorscale=self.color_scheme, showscale=False), row=1, col=1)
        if k_cache_list:
            energies = []
            for k_cache in k_cache_list[:current_position]:
                k_np = self._tensor_to_numpy(k_cache)
                if k_np.ndim == 5:
                    e = np.mean(np.linalg.norm(k_np, axis=-1))
                else:
                    e = np.mean(np.linalg.norm(k_np, axis=-1))
                energies.append(e)
            fig.add_trace(go.Bar(x=list(range(1, len(energies) + 1)), y=energies, marker_color='steelblue', showlegend=False), row=1, col=2)
        if current_position > 0 and current_position <= len(k_cache_list):
            k_cache = k_cache_list[current_position - 1]
            k_np = self._tensor_to_numpy(k_cache)
            if k_np.ndim == 5:
                layer_means = [np.mean(k_np[l]) for l in range(min(k_np.shape[0], 12))]
            else:
                layer_means = [np.mean(k_np)]
            fig.add_trace(go.Scatter(x=list(range(len(layer_means))), y=layer_means, mode='lines+markers', showlegend=False, line=dict(color='steelblue')), row=2, col=1)
        current_token = tokens[current_position - 1] if current_position > 0 and current_position <= len(tokens) else "N/A"
        token_info = [
            ["Current Token", current_token],
            ["Position", f"{current_position}"],
            ["Total Tokens", f"{len(tokens)}"],
            ["Num Layers", f"{self.num_layers}"],
            ["Num Heads", f"{self.num_heads}"],
            ["Head Dim", f"{self.head_dim}"],
        ]
        fig.add_trace(go.Table(cells=dict(values=token_info, fill_color='lavender', align='left')), row=2, col=2)
        fig.update_layout(title=dict(text=title, x=0.5), width=900, height=700, showlegend=False)
        return fig

    def calculate_cache_stats(
        self,
        k_cache_list: List[torch.Tensor],
        v_cache_list: List[torch.Tensor]
    ) -> Dict[str, float]:
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
        n = len(k_cache_list)
        if n > 0:
            stats['cache_efficiency'] = round(200 / (n + 1), 2)
        total_bytes = 0
        for k, v in zip(k_cache_list, v_cache_list):
            k_np = self._tensor_to_numpy(k)
            v_np = self._tensor_to_numpy(v)
            total_bytes += k_np.nbytes + v_np.nbytes
        stats['peak_memory_mb'] = round(total_bytes / (1024 * 1024), 2)
        all_energies = []
        for k_cache in k_cache_list:
            k_np = self._tensor_to_numpy(k_cache)
            if k_np.ndim == 5:
                energy = np.mean(np.linalg.norm(k_np, axis=-1))
            else:
                energy = np.mean(np.linalg.norm(k_np, axis=-1))
            all_energies.append(energy)
        stats['avg_layer_energy'] = round(np.mean(all_energies), 4)
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
        fig = go.Figure()
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
            header=dict(values=table_data[0], fill_color='paleturquoise', align='left', font=dict(size=14)),
            cells=dict(values=[row[1] for row in table_data[1:]], fill_color='lavender', align='left', font=dict(size=12))
        ))
        fig.update_layout(title=dict(text=title, x=0.5), width=400, height=300, showlegend=False)
        return fig

    def create_layer_energy_heatmap(
        self,
        k_cache_list: List[torch.Tensor],
        title: str = "Layer Energy Heatmap"
    ) -> go.Figure:
        """创建层级能量热力图 - 展示各层 KV Cache 的 L2 范数"""
        import numpy as np

        if not k_cache_list:
            return go.Figure()

        # 计算每层每位置的 L2 范数
        num_layers = self.num_layers
        seq_len = len(k_cache_list)

        k_energies = np.zeros((num_layers, seq_len))
        for pos, k in enumerate(k_cache_list):
            if k is not None and k.numel() > 0:
                # 按层计算
                for layer_idx in range(min(num_layers, k.shape[0] if len(k.shape) > 3 else 1)):
                    layer_k = k[layer_idx] if len(k.shape) > 3 else k
                    k_energies[layer_idx, pos] = torch.norm(layer_k).item()

        fig = px.imshow(
            k_energies,
            x=[f"Tok {i+1}" for i in range(seq_len)],
            y=[f"Layer {i+1}" for i in range(num_layers)],
            title=title,
            color_continuous_scale="Viridis",
            aspect="auto"
        )
        fig.update_layout(
            xaxis_title="Token Position",
            yaxis_title="Layer",
        )
        return fig