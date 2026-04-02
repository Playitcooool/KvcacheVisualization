# KV Cache 可视化器 - Phase 5 功能扩展设计

> **日期**: 2026-04-02
> **范围**: Layer Energy 变化追踪 + 浅层/深层 Attention 对比 + 渐进式渲染

---

## 1. 项目概述

**当前状态**: Phase 1-4 已完成，包含 Attention Pattern 可视化、Token Importance、多模型对比。
**本次扩展**: 新增层分析 Tab 和渐进式渲染支持。

---

## 2. 功能规格

### 2.1 Feature 1: Layer Energy 变化追踪

**位置**: 新 Tab "📊 层分析" 内
**功能**: 展示生成过程中各层 KV energy 的变化曲线

**实现**:
- `visualizer.create_layer_energy_evolution()` 新方法
- 输入: `k_cache_list: List[Tensor]`（每个位置各层的 K cache）
- 对每个 token 位置，计算所有层的 L2 norm
- 输出: 折线图，X轴=token位置(1..N)，Y轴=层能量，每条线=一层
- GPT2: 12 层 → 12 条线，用不同颜色区分

**数据计算**:
```python
for pos, k_cache in enumerate(k_cache_list):
    # k_cache: [batch, heads, seq=1, head_dim] 或 [batch, layers, heads, seq, head_dim]
    energy_per_layer = []
    for layer_idx in range(num_layers):
        layer_k = k_cache[layer_idx] if k_cache.ndim == 5 else k_cache
        energy = np.mean(np.linalg.norm(layer_k, axis=-1))
        energy_per_layer.append(energy)
    layer_energies[pos] = energy_per_layer
```

### 2.2 Feature 2: 浅层 vs 深层 Attention 统计对比

**位置**: 新 Tab "📊 层分析" 内
**功能**: 按层统计 attention 覆盖度、稀疏度等指标的折线图

**实现**:
- `visualizer.calculate_attention_stats_by_layer()` 新方法
- 输入: `attn_weights_list: List[Tensor]`（每个位置完整的 [batch, heads, seq, seq] 矩阵）
- 对每个位置，计算各层的 attention 统计：
  - **覆盖度**: mean(attn > threshold)，即有多少比例的 attention 权重不是噪声
  - **稀疏度**: mean(attn < 0.01)，即多少比例接近零
  - **最大值**: max(attn)，peak attention 强度
- 输出: 折线图，X轴=层(1..N)，Y轴=统计指标值

**数据计算**:
```python
for layer_idx in range(num_layers):
    # 收集该层所有位置的 attention
    layer_attn = concatenate([attn[0, layer_idx, :, :] for attn in attn_weights_list])
    coverage = np.mean(layer_attn > 0.01)
    sparsity = np.mean(layer_attn < 0.001)
    max_val = np.max(layer_attn)
```

### 2.3 Feature 3: 渐进式渲染

**位置**: 修改现有可视化 Tab（序列视图、层级能量、Attention 等）
**功能**: 初始只渲染前 50 个 token，点击"加载更多"逐步展开

**实现**:
- `st.session_state.display_token_limit` 控制当前渲染的 token 数量，默认 50
- 渲染函数接收 `display_limit` 参数，超过时底部显示按钮
- "加载更多" 按钮: `display_token_limit += 25`
- 最多显示到实际生成的 token 总数
- Session state 管理，不打断现有回放控制

**UI 交互**:
```
[序列视图 Tab]
Token 1 ~ Token 50 ████████████████

[加载更多 ▼]  # 点击展开 +25
Token 51 ~ Token 75 ████████
```

---

## 3. UI 结构

### 3.1 新 Tab 布局

```
Tab: [📈 序列] [🔳 层级] [📐 统计] [🖥️ 仪表盘] [🔥 能量] [🧠 Attention] [⭐ Token重要性] [📊 层分析]
                                                                                                    ↑
                                                                                            新增 Tab
```

**"📊 层分析" Tab 内容**:
```
┌─ 层能量变化 ─────────────────────────────────┐
│ 折线图: X=token位置, Y=层能量, 12条线        │
└─────────────────────────────────────────────┘

┌─ Attention 层间统计 ─────────────────────────┐
│ 折线图选择: [覆盖度] [稀疏度] [最大值]        │
│ 折线图: X=层, Y=所选指标                     │
└─────────────────────────────────────────────┘
```

### 3.2 渐进式渲染修改

- 影响 Tab: 序列视图、层级能量、Attention Pattern、Token 重要性
- 不影响: 统计 Tab（数据量小）、仪表盘（已经是聚合数据）

---

## 4. 技术实现

### 4.1 新增方法

**visualizer.py**:
```python
def create_layer_energy_evolution(
    self,
    k_cache_list: List[torch.Tensor],
    title: str = "Layer Energy Evolution"
) -> go.Figure

def calculate_attention_stats_by_layer(
    self,
    attn_weights_list: List[torch.Tensor]
) -> Dict[int, Dict[str, float]]:  # {layer_idx: {coverage, sparsity, max_val}}

def create_attention_layer_stats(
    self,
    stats_by_layer: Dict[int, Dict[str, float]],
    metric: str = "coverage",
    title: str = "Attention Stats by Layer"
) -> go.Figure
```

**ui/components.py**:
```python
def _render_token_subset(items, display_limit, clean_bpe_token_func):
    """渲染 items 的子集，超过 display_limit 显示加载更多"""
    ...

def render_layer_analysis_tab(k_cache_list, attn_weights_list, cleaned_tokens, visualizer, display_limit):
    """渲染新的"层分析"Tab"""
    ...
```

### 4.2 Session State 新增字段

```python
'display_token_limit': 50,  # 渐进式渲染的当前显示数量
```

### 4.3 数据流

```
app.py render_visualization_tabs()
    │
    ├── 现有 Tab（序列/层级/Attention/Token重要性）
    │       └── 受 display_token_limit 影响
    │
    └── render_layer_analysis_tab()  # 新 Tab
            ├── create_layer_energy_evolution()  ← Feature 1
            └── create_attention_layer_stats()    ← Feature 2
```

---

## 5. 验收标准

- [x] 层能量变化折线图正常显示（12 层 GPT2）
- [x] Attention 层间统计折线图正常显示，支持覆盖度/稀疏度/最大值切换
- [x] 渐进式渲染：初始显示 ≤50 token，"加载更多" 按钮正常扩展
- [x] 渐进式渲染不干扰回放控制（slider 位置不变）
- [x] app.py 总行数不大幅增加（<600 行）

---

## 6. 非目标

- 不修改 attention 捕获逻辑（复用现有 hook）
- 不修改多模型对比面板
- 不做移动端适配
