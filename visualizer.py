# visualizer.py
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from utils.logger import setup_logger
logger = setup_logger(__name__)

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
            subplot_titles=("Token 序列", "Layer 能量", "Token 信息", "生成进度"),
            specs=[[{"type": "bar"}, {"type": "scatter"}], [{"type": "table"}, {"type": "indicator"}]],
            vertical_spacing=0.15, horizontal_spacing=0.1
        )

        # Token 序列视图
        if tokens:
            token_lens = list(range(1, len(tokens) + 1))
            fig.add_trace(go.Bar(
                x=token_lens,
                y=[1.0] * len(tokens),
                marker_color='steelblue',
                showlegend=False,
                text=tokens,
                textposition='outside'
            ), row=1, col=1)

        # Layer 能量散点图
        if current_position > 0 and current_position <= len(k_cache_list):
            k_cache = k_cache_list[current_position - 1]
            k_np = self._tensor_to_numpy(k_cache)
            if k_np.ndim == 5:
                layer_means = [np.mean(np.abs(k_np[l])) for l in range(min(k_np.shape[0], self.num_layers))]
            else:
                layer_means = [np.mean(np.abs(k_np))]
            fig.add_trace(go.Scatter(
                x=list(range(len(layer_means))),
                y=layer_means,
                mode='lines+markers',
                showlegend=False,
                line=dict(color='steelblue')
            ), row=1, col=2)

        # Token 信息表格
        current_token = tokens[current_position - 1] if current_position > 0 and current_position <= len(tokens) else "N/A"
        token_info = [
            ["当前 Token", current_token if current_token else "N/A"],
            ["位置", f"{current_position}"],
            ["总 Token 数", f"{len(tokens)}"],
            ["Layers", f"{self.num_layers}"],
            ["Heads", f"{self.num_heads}"],
        ]
        fig.add_trace(go.Table(
            cells=dict(values=token_info, fill_color='lavender', align='left')
        ), row=2, col=1)

        # 生成进度
        fig.add_trace(go.Indicator(
            mode="number",
            value=current_position,
            title={"text": "已生成"}
        ), row=2, col=2)

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
        # Cache efficiency = 100% - sparsity (低稀疏度 = 高效率)
        stats['cache_efficiency'] = round(max(0, 100 - stats['sparsity']), 2)
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

    def create_attention_heatmap(
        self,
        attn_weights: torch.Tensor,
        tokens: List[str] = None,
        title: str = "Attention Pattern"
    ) -> go.Figure:
        """
        创建 Attention Pattern 热力图。

        Args:
            attn_weights: Attention weights tensor [batch, heads, seq, seq]
            tokens: 可选的 token 列表用于 x/y 轴标签
            title: 图表标题

        Returns:
            Plotly Figure
        """
        if attn_weights is None:
            return go.Figure()

        attn_np = self._tensor_to_numpy(attn_weights)

        # attn_np shape: [batch, heads, seq, seq] -> 取 batch=0, 平均所有 heads
        if attn_np.ndim == 4:
            # 平均所有 heads，然后取 batch=0
            attn_avg = np.mean(attn_np, axis=1)  # [batch, seq, seq]
            attn_avg = attn_avg[0]  # [seq, seq]
        else:
            attn_avg = attn_np

        # 如果有 tokens，用 tokens 作为标签
        if tokens and len(tokens) == attn_avg.shape[0]:
            x_labels = tokens
            y_labels = tokens
        else:
            x_labels = [f"T{i+1}" for i in range(attn_avg.shape[0])]
            y_labels = x_labels

        fig = px.imshow(
            attn_avg,
            x=x_labels,
            y=y_labels,
            title=title,
            color_continuous_scale="Blues",
            aspect="auto"
        )
        fig.update_layout(
            xaxis_title="Key Tokens",
            yaxis_title="Query Tokens",
        )
        return fig

    def create_attention_per_head(
        self,
        attn_weights: torch.Tensor,
        tokens: List[str] = None,
        title: str = "Attention Pattern per Head"
    ) -> go.Figure:
        """
        创建每个 Head 的 Attention Pattern 子图。

        Args:
            attn_weights: Attention weights tensor [batch, heads, seq, seq]
            tokens: 可选的 token 列表
            title: 图表标题

        Returns:
            Plotly Figure with subplots for each head
        """
        if attn_weights is None:
            return go.Figure()

        attn_np = self._tensor_to_numpy(attn_weights)

        if attn_np.ndim != 4:
            return self.create_attention_heatmap(attn_weights, tokens, title)

        batch, num_heads, seq_len, _ = attn_np.shape

        # 限制显示的 heads 数量（最多 12 个）
        max_heads_to_show = min(num_heads, 12)
        num_rows = (max_heads_to_show + 3) // 4
        num_cols = min(4, max_heads_to_show)

        fig = make_subplots(
            rows=num_rows,
            cols=num_cols,
            subplot_titles=[f"Head {i}" for i in range(max_heads_to_show)],
            horizontal_spacing=0.1,
            vertical_spacing=0.15
        )

        x_labels = tokens if tokens else [f"T{i+1}" for i in range(seq_len)]
        y_labels = x_labels

        for head_idx in range(max_heads_to_show):
            row = head_idx // num_cols + 1
            col = head_idx % num_cols + 1

            head_attn = attn_np[0, head_idx, :, :]  # [seq, seq]

            fig.add_trace(
                go.Heatmap(
                    z=head_attn,
                    x=x_labels,
                    y=y_labels,
                    colorscale="Blues",
                    showscale=(head_idx == 0),
                    colorbar=dict(len=0.4, y=0.8 - (head_idx // 4) * 0.3)
                ),
                row=row,
                col=col
            )

        fig.update_layout(
            title=dict(text=title, x=0.5),
            showlegend=False,
            height=300 * num_rows
        )

        return fig

    def create_attention_summary(
        self,
        attn_weights_list: List[torch.Tensor],
        tokens: List[str],
        title: str = "Attention Summary"
    ) -> go.Figure:
        """
        创建 Attention 汇总视图 - 展示多个位置的 attention patterns。

        Args:
            attn_weights_list: Attention weights 历史列表
            tokens: token 列表
            title: 图表标题

        Returns:
            Plotly Figure
        """
        if not attn_weights_list or attn_weights_list[0] is None:
            return go.Figure()

        # 选择最后几个位置的 attention
        positions_to_show = min(4, len(attn_weights_list))
        start_idx = len(attn_weights_list) - positions_to_show

        num_rows = positions_to_show
        num_cols = 1

        fig = make_subplots(
            rows=num_rows,
            cols=num_cols,
            subplot_titles=[f"Token {start_idx + i + 1}: '{tokens[start_idx + i] if start_idx + i < len(tokens) else '?'}'"
                          for i in range(positions_to_show)],
            vertical_spacing=0.2
        )

        for i, attn in enumerate(attn_weights_list[start_idx:]):
            if attn is None:
                continue

            attn_np = self._tensor_to_numpy(attn)

            if attn_np.ndim == 4:
                attn_avg = np.mean(attn_np, axis=1)  # [batch, seq, seq]
                # 取最后一个位置的注意力向量（行）
                attn_avg = attn_avg[-1, :]  # [seq,]
            else:
                attn_avg = np.asarray(attn_np).squeeze()

            # 限制显示的 token 数量
            max_tokens = min(attn_avg.shape[-1], 20)
            attn_display = attn_avg[..., -max_tokens:] if attn_avg.shape[-1] > max_tokens else attn_avg

            x_labels = tokens[-max_tokens:] if len(tokens) >= max_tokens else tokens

            fig.add_trace(
                go.Bar(
                    x=list(range(len(attn_display))),
                    y=attn_display,
                    marker_color='steelblue',
                    showlegend=False,
                    text=[f"{v.item():.2f}" for v in attn_display],
                    textposition='outside'
                ),
                row=i + 1,
                col=1
            )

        fig.update_layout(
            title=dict(text=title, x=0.5),
            showlegend=False,
            height=200 * num_rows
        )
        fig.update_xaxes(title_text="Token Position")
        fig.update_yaxes(title_text="Attention")

        return fig

    def create_token_importance(
        self,
        k_cache_list: List[torch.Tensor],
        tokens: List[str],
        title: str = "Token Importance (by KV Energy)"
    ) -> go.Figure:
        """
        创建 Token 重要性视图 - 按 KV Energy 排序显示 token。

        Args:
            k_cache_list: KV Cache tensor 列表
            tokens: token 列表
            title: 图表标题

        Returns:
            Plotly Figure
        """
        if not k_cache_list or not tokens:
            return go.Figure()

        # 计算每个 token 的 KV energy
        energies = []
        for k in k_cache_list:
            if k is not None and k.numel() > 0:
                # 计算 L2 norm
                energy = torch.norm(k).item()
            else:
                energy = 0
            energies.append(energy)

        # 创建 (token, energy) 对并排序
        token_energies = list(zip(tokens, energies))
        token_energies.sort(key=lambda x: x[1], reverse=True)

        sorted_tokens = [t for t, _ in token_energies]
        sorted_energies = [e for _, e in token_energies]

        # 颜色映射 - 高能量用深色
        max_energy = max(sorted_energies) if sorted_energies else 1
        colors = [f"hsl(220, {min(100, 30 + (e / max_energy) * 70)}%, {80 - (e / max_energy) * 30}%)"
                  for e in sorted_energies]

        fig = go.Figure(data=go.Bar(
            x=list(range(len(sorted_tokens))),
            y=sorted_energies,
            marker_color=colors,
            text=sorted_tokens,
            textposition='outside',
            hovertemplate="<b>%{text}</b><br>Energy: %{y:.4f}<extra></extra>"
        ))

        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Token Rank (by Energy)",
            yaxis_title="KV Energy (L2 Norm)",
            width=800,
            height=400,
            showlegend=False
        )
        fig.update_xaxes(
            tickmode='array',
            tickvals=list(range(len(sorted_tokens))),
            ticktext=[f"{i+1}. {t[:10]}..." if len(t) > 10 else f"{i+1}. {t}"
                     for i, t in enumerate(sorted_tokens)],
            tickangle=45
        )

        return fig

    def create_token_importance_heatmap(
        self,
        k_cache_list: List[torch.Tensor],
        tokens: List[str],
        title: str = "Token Importance Heatmap"
    ) -> go.Figure:
        """
        创建 Token 重要性热力图 - 展示能量随位置的变化。

        Args:
            k_cache_list: KV Cache tensor 列表
            tokens: token 列表
            title: 图表标题

        Returns:
            Plotly Figure
        """
        if not k_cache_list:
            return go.Figure()

        seq_len = len(k_cache_list)

        # 计算能量
        energies = []
        for k in k_cache_list:
            if k is not None and k.numel() > 0:
                energy = torch.norm(k).item()
            else:
                energy = 0
            energies.append(energy)

        # 创建热力图数据 - 单行多列
        import numpy as np
        energy_matrix = np.array(energies).reshape(1, -1)

        fig = px.imshow(
            energy_matrix,
            x=[f"{i}: {tokens[i][:8]}..." if len(tokens[i]) > 8 else f"{i}: {tokens[i]}"
               for i in range(seq_len)],
            y=["Energy"],
            title=title,
            color_continuous_scale="YlOrRd",
            aspect="auto"
        )

        fig.update_layout(
            xaxis_title="Token Position",
            yaxis_title="",
            showlegend=False
        )

        return fig