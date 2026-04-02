# Phase 5 Implementation Plan: Layer Analysis + Progressive Rendering

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现三个功能：Layer Energy 变化折线图、Attention 层间统计对比、渐进式渲染。

**Architecture:**
- Feature 1 & 2: 新增 `KVCacheVisualizer` 方法 + 新 Tab `render_layer_analysis_tab()`
- Feature 3: Session state `display_token_limit` 控制渐进渲染，"加载更多" 按钮

**Tech Stack:** PyTorch, Plotly, Streamlit

---

## File Map

| File | Change |
|------|--------|
| `visualizer.py` | +3 methods |
| `ui/components.py` | +1 helper, 修改 `render_visualization_tabs` |
| `app.py` | +session state `display_token_limit` |

**Test file:** `tests/test_visualizer.py`

---

## Task 1: Add `create_layer_energy_evolution()` to visualizer.py

**Files:**
- Modify: `visualizer.py` (add method at end of class)
- Test: `tests/test_visualizer.py`

- [ ] **Step 1: Write the failing test**

```python
def test_create_layer_energy_evolution():
    viz = KVCacheVisualizer(num_layers=2, num_heads=2, head_dim=8)
    # k_cache_list: 3 positions, each with 2 layers
    k_cache_list = [torch.randn(1, 2, 1, 1, 8) for _ in range(3)]
    fig = viz.create_layer_energy_evolution(k_cache_list)
    assert isinstance(fig, go.Figure)
    # Should have 2 lines (one per layer)
    assert len(fig.data) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_visualizer.py::test_create_layer_energy_evolution -v`
Expected: FAIL — method not defined

- [ ] **Step 3: Write the implementation**

Add to `visualizer.py` before the closing `EOF`:

```python
def create_layer_energy_evolution(
    self,
    k_cache_list: List[torch.Tensor],
    title: str = "Layer Energy Evolution"
) -> go.Figure:
    """
    展示生成过程中各层 KV energy 的变化曲线。
    X轴=token位置，Y轴=层能量，每条线=一层。

    Args:
        k_cache_list: 每个 token 位置的 K cache 列表
        title: 图表标题

    Returns:
        Plotly Figure
    """
    if not k_cache_list:
        return go.Figure()

    num_positions = len(k_cache_list)
    x = list(range(1, num_positions + 1))

    fig = go.Figure()
    colors = px.colors.qualitative.Set2

    for layer_idx in range(self.num_layers):
        layer_energies = []
        for k_cache in k_cache_list:
            k_np = self._tensor_to_numpy(k_cache)
            # k_np: [batch, heads, seq, head_dim] or [batch, layers, heads, seq, head_dim]
            if k_np.ndim == 5:
                # [layers, heads, seq, head_dim] at batch=0
                layer_k = k_np[layer_idx]
            else:
                layer_k = k_np
            energy = float(np.mean(np.linalg.norm(layer_k, axis=-1)))
            layer_energies.append(energy)

        fig.add_trace(go.Scatter(
            x=x,
            y=layer_energies,
            mode='lines+markers',
            name=f'Layer {layer_idx + 1}',
            line=dict(color=colors[layer_idx % len(colors)]),
            marker=dict(size=6)
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Token Position",
        yaxis_title="Layer Energy (L2 norm)",
        width=600,
        height=400,
        showlegend=True,
        legend=dict(y=0.99, x=0.01)
    )
    return fig
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_visualizer.py::test_create_layer_energy_evolution -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_visualizer.py visualizer.py
git commit -m "feat: add create_layer_energy_evolution for layer energy tracking"
```

---

## Task 2: Add attention layer stats methods to visualizer.py

**Files:**
- Modify: `visualizer.py`
- Test: `tests/test_visualizer.py`

- [ ] **Step 1: Write the failing test**

```python
def test_calculate_attention_stats_by_layer():
    viz = KVCacheVisualizer(num_layers=2, num_heads=2, head_dim=8)
    # attn_weights: [batch=1, heads=2, seq=3, seq=3]
    attn = torch.softmax(torch.randn(1, 2, 3, 3), dim=-1)
    attn_list = [attn, attn, attn]
    stats = viz.calculate_attention_stats_by_layer(attn_list)
    assert isinstance(stats, dict)
    assert len(stats) == 2  # 2 layers
    assert 'coverage' in stats[0]
    assert 'sparsity' in stats[0]
    assert 'max_val' in stats[0]

def test_create_attention_layer_stats():
    viz = KVCacheVisualizer(num_layers=2, num_heads=2, head_dim=8)
    stats = {
        0: {'coverage': 0.5, 'sparsity': 0.3, 'max_val': 0.8},
        1: {'coverage': 0.6, 'sparsity': 0.2, 'max_val': 0.9},
    }
    fig = viz.create_attention_layer_stats(stats, metric='coverage')
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_visualizer.py::test_calculate_attention_stats_by_layer tests/test_visualizer.py::test_create_attention_layer_stats -v`
Expected: FAIL — methods not defined

- [ ] **Step 3: Write the implementation**

Add to `visualizer.py`:

```python
def calculate_attention_stats_by_layer(
    self,
    attn_weights_list: List[torch.Tensor]
) -> Dict[int, Dict[str, float]]:
    """
    按层计算 attention 统计指标。

    Args:
        attn_weights_list: 每个位置的 attention 权重列表 [batch, heads, seq, seq]

    Returns:
        Dict {layer_idx: {coverage, sparsity, max_val}}
    """
    if not attn_weights_list:
        return {}

    stats = {}
    COV_THRESHOLD = 0.01
    SPARSITY_THRESHOLD = 0.001

    for layer_idx in range(self.num_layers):
        all_attn = []
        for attn in attn_weights_list:
            if attn is None:
                continue
            attn_np = self._tensor_to_numpy(attn)
            if attn_np.ndim == 4:
                # [batch, heads, seq, seq] -> take batch=0, specific layer
                layer_attn = attn_np[0, layer_idx, :, :]  # [seq, seq]
                all_attn.append(layer_attn)

        if not all_attn:
            stats[layer_idx] = {'coverage': 0.0, 'sparsity': 0.0, 'max_val': 0.0}
            continue

        # Concatenate all positions along seq axis
        concat = np.concatenate([a[np.newaxis, :] for a in all_attn], axis=0)  # [positions, seq]
        concat = concat.flatten()

        coverage = float(np.mean(concat > COV_THRESHOLD))
        sparsity = float(np.mean(concat < SPARSITY_THRESHOLD))
        max_val = float(np.max(concat))

        stats[layer_idx] = {
            'coverage': coverage,
            'sparsity': sparsity,
            'max_val': max_val
        }

    return stats


def create_attention_layer_stats(
    self,
    stats_by_layer: Dict[int, Dict[str, float]],
    metric: str = "coverage",
    title: str = "Attention Stats by Layer"
) -> go.Figure:
    """
    创建按层统计的折线图。

    Args:
        stats_by_layer: calculate_attention_stats_by_layer() 的输出
        metric: "coverage" | "sparsity" | "max_val"
        title: 图表标题

    Returns:
        Plotly Figure
    """
    if not stats_by_layer:
        return go.Figure()

    layers = sorted(stats_by_layer.keys())
    x = [l + 1 for l in layers]
    y = [stats_by_layer[l].get(metric, 0.0) for l in layers]

    metric_labels = {
        'coverage': 'Attention Coverage (%)',
        'sparsity': 'Attention Sparsity (%)',
        'max_val': 'Max Attention'
    }

    fig = go.Figure(go.Scatter(
        x=x,
        y=y,
        mode='lines+markers',
        name=metric_labels.get(metric, metric),
        line=dict(color='steelblue'),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Layer",
        yaxis_title=metric_labels.get(metric, metric),
        width=600,
        height=400,
        showlegend=False
    )
    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_visualizer.py::test_calculate_attention_stats_by_layer tests/test_visualizer.py::test_create_attention_layer_stats -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_visualizer.py visualizer.py
git commit -m "feat: add attention layer stats methods"
```

---

## Task 3: Add layer analysis tab to ui/components.py

**Files:**
- Modify: `ui/components.py` (add `render_layer_analysis_tab()` function)
- Modify: `ui/__init__.py` (export new function)
- Modify: `app.py` (call new function in tab)

- [ ] **Step 1: Write the implementation**

First, add the new function to `ui/components.py`. Add before the `render_comparison_panel` function:

```python
def render_layer_analysis_tab(k_cache_list, attn_weights_list, cleaned_tokens, visualizer):
    """Render the Layer Analysis tab with energy evolution and attention stats."""
    st.markdown("### 📊 层能量变化")
    fig_evo = visualizer.create_layer_energy_evolution(
        k_cache_list,
        title="各层能量随生成位置的变化"
    )
    st.plotly_chart(fig_evo, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔄 Attention 层间统计")

    metric = st.radio(
        "选择统计指标",
        ["coverage", "sparsity", "max_val"],
        format_func=lambda x: {
            "coverage": "覆盖度",
            "sparsity": "稀疏度",
            "max_val": "最大值"
        }[x],
        horizontal=True
    )

    stats = visualizer.calculate_attention_stats_by_layer(attn_weights_list)
    if stats:
        fig_stats = visualizer.create_attention_layer_stats(
            stats,
            metric=metric,
            title=f"Attention {metric} by Layer"
        )
        st.plotly_chart(fig_stats, use_container_width=True)
    else:
        st.info("先生成 token 后才能查看 Attention 层间统计")
```

- [ ] **Step 2: Update ui/__init__.py to export new function**

Add `render_layer_analysis_tab` to the imports from `ui.components`.

- [ ] **Step 3: Add the new tab in render_visualization_tabs**

Modify `render_visualization_tabs()` in `ui/components.py`. Change the tabs line from:
```python
tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    ["📈 序列视图", "🔳 层级分布", "📐 统计数据", "🖥️ 综合仪表盘", "🔥 层级能量", "🧠 Attention", "⭐ Token重要性"]
)
```
To:
```python
tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
    ["📈 序列视图", "🔳 层级分布", "📐 统计数据", "🖥️ 综合仪表盘", "🔥 层级能量", "🧠 Attention", "⭐ Token重要性", "📊 层分析"]
)
```

Add at the end of `render_visualization_tabs()`, after the tab8 block:
```python
    with tab9:
        render_layer_analysis_tab(
            k_cache_list,
            attn_weights_list,
            cleaned_tokens,
            visualizer
        )
```

- [ ] **Step 4: Verify syntax**

Run: `python3 -m py_compile ui/components.py ui/__init__.py app.py`
Expected: no output

- [ ] **Step 5: Commit**

```bash
git add ui/components.py ui/__init__.py app.py
git commit -m "feat: add layer analysis tab with energy evolution and attention stats"
```

---

## Task 4: Progressive rendering (加载更多)

**Files:**
- Modify: `app.py` (add `display_token_limit` to session state defaults)
- Modify: `ui/components.py` (modify `render_visualization_tabs`)

- [ ] **Step 1: Add session state default in app.py**

In `init_session_state()`, add to the defaults dict:
```python
'display_token_limit': 50,  # 渐进式渲染：初始显示 token 数
```

- [ ] **Step 2: Add _render_load_more helper to ui/components.py**

Add before `render_visualization_tabs()`:

```python
def _render_load_more_button():
    """Render '加载更多' button and update display_token_limit session state."""
    total = len(st.session_state.tokens)
    current_limit = st.session_state.get('display_token_limit', 50)

    if current_limit >= total:
        return  # No more to load

    if st.button("加载更多", key="load_more_btn"):
        st.session_state.display_token_limit = min(current_limit + 25, total)
        st.rerun()
```

- [ ] **Step 3: Add load more to affected tabs in render_visualization_tabs**

In the tab2 (sequence view) section, after `st.plotly_chart`, add:
```python
    _render_load_more_button()
```

Do the same for tab6 (Attention) and tab8 (Token重要性).

**Important**: In tab6 (Attention) and tab8 (Token重要性), also update the data slicing to use `st.session_state.get('display_token_limit', 50)`:
- In Attention tab: `attn_weights_list[:st.session_state.get('display_token_limit', 50)]`
- In Token Importance tab: `k_cache_list[:st.session_state.get('display_token_limit', 50)]`

- [ ] **Step 4: Verify syntax**

Run: `python3 -m py_compile app.py ui/components.py`
Expected: no output

- [ ] **Step 5: Commit**

```bash
git add app.py ui/components.py
git commit -m "feat: add progressive rendering with load-more button"
```

---

## Task 5: Wire progressive rendering to all affected tabs + verify

**Files:**
- Modify: `ui/components.py`

- [ ] **Step 1: Verify all affected tabs slice with display_token_limit**

Review `render_visualization_tabs()` and ensure:
- `tab2` (sequence view): Already uses `[:st.session_state.current_position]`, add load-more button
- `tab5` (层级能量 heatmap): Already uses `[:st.session_state.current_position]`, add load-more button
- `tab6` (Attention): Change `[:st.session_state.current_position]` to `[:min(st.session_state.get('display_token_limit', 50), st.session_state.current_position)]`
- `tab8` (Token Importance): Same slicing change as Attention tab

For `tab5` and `tab2`, just add the `_render_load_more_button()` call — they already respect `current_position`.

- [ ] **Step 2: Verify no regressions**

Check that the replay slider (`render_replay_control`) still controls `current_position` independently of `display_token_limit` — the two are separate controls.

- [ ] **Step 3: Commit**

```bash
git add ui/components.py
git commit -m "chore: wire progressive rendering to all affected tabs"
```

---

## Verification Checklist

After all tasks:
- [ ] `python3 -m pytest tests/test_visualizer.py -v` — all pass
- [ ] Layer analysis tab visible with 2 sub-sections (energy evolution + attention stats)
- [ ] Layer energy evolution shows 12 lines for GPT2
- [ ] Attention stats radio switches between coverage/sparsity/max_val
- [ ] "加载更多" button appears when tokens > 50, expands on click
- [ ] Replay slider still works independently
- [ ] `app.py` line count < 450
