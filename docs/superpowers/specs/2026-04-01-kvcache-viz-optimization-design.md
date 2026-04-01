# KV Cache 可视化器优化设计方案

> **日期**: 2026-04-01
> **优化优先级**: 功能 > 性能 > UX > 调试

## 1. 项目概述

对 KvcacheVisualization 进行全面优化，扩展功能、提升性能、改善用户体验、增强调试能力。

## 2. 优化领域

### 2.1 功能增强（最高优先级）

#### 2.1.1 多架构支持

| 架构类型 | 支持模型示例 | Hook 模式 |
|---------|-------------|-----------|
| **CausalLM (Decoder-only)** | GPT-2, LLaMA, Qwen, TinyLlama | `attn.c_attn`, `qkv_proj` |
| **T5/FLAN (Encoder-Decoder)** | T5, FLAN-T5, mT5 | `EncDecAttention`, `qkv_proj` |

**实现方案**: 扩展 `kvcache_extractor.py`，支持检测模型架构类型并应用对应的 Hook 模式。

#### 2.1.2 数据导出

```python
# 导出 KV Cache 数据
def export_kvcache(simulator, format="json"):
    """导出为 JSON 或 CSV"""
    data = {
        "tokens": simulator.tokens,
        "k_cache": [k.tolist() for k in k_cache_list],
        "v_cache": [v.tolist() for v in v_cache_list],
        "stats": calculate_cache_stats(...)
    }

    if format == "json":
        return json.dumps(data, indent=2)
    elif format == "csv":
        # 转换为表格格式
        return pd.DataFrame(data).to_csv()
```

#### 2.1.3 生成对比

记录两次生成的 KV Cache 状态，支持在界面上对比差异。

### 2.2 性能优化

#### 2.2.1 批量生成

```python
# 当前：一次性生成 N 个 token
# 优化：分批生成，每批更新可视化

for batch in range(num_batches):
    output = model.generate(...)
    update_visualization(batch_i)
    time.sleep(0.1)  # 可配置的批次间隔
```

#### 2.2.2 增量可视化

```python
# 使用 Plotly 的 updatemode="restyle" 而非完全重绘
fig.update_traces(...)  # 只更新数据，不重建图表
```

#### 2.2.3 GPU 内存优化

```python
# 使用 torch.cuda.empty_cache() 清理缓存
# 限制 KV Cache 历史长度
MAX_HISTORY_LENGTH = 100
```

### 2.3 用户体验

#### 2.3.1 中英双语界面

```python
# i18n.py
TRANSLATIONS = {
    "zh": {
        "model_settings": "模型设置",
        "start_generation": "开始生成",
        "kvcache_visualization": "KV Cache 可视化",
    },
    "en": {
        "model_settings": "Model Settings",
        "start_generation": "Start Generation",
        "kvcache_visualization": "KV Cache Visualization",
    }
}
```

#### 2.3.2 暗色主题

```python
# 主题切换
THEMES = {
    "light": {
        "background": "#ffffff",
        "text": "#31333F",
        "chart_colors": ["#1f77b4", "#ff7f0e", "#2ca02c"],
    },
    "dark": {
        "background": "#0e1117",
        "text": "#f0f0f0",
        "chart_colors": ["#66b3ff", "#ffb366", "#99ff99"],
    }
}
```

#### 2.3.3 预设 Prompt 模板

```python
PROMPT_TEMPLATES = [
    {"name": "问答", "template": "问题：{question}\n回答："},
    {"name": "翻译", "template": "翻译为英文：{text}"},
    {"name": "代码补全", "template": "def {function_name}:\n    "},
    {"name": "故事续写", "template": "从前有座山：{start}"},
]
```

### 2.4 调试友好

#### 2.4.1 Attention 分布图

展示每个 token 位置的 attention weight 分布。

#### 2.4.2 层级能量热力图

```python
# 各层 KV Cache 的 L2 范数热力图
layer_energies = [torch.norm(k, p=2).item() for k in k_cache_list]
fig = px.imshow([layer_energies], title="Layer Energy Distribution")
```

#### 2.4.3 Token 详情悬浮

鼠标悬停在 token 上时显示详细信息（token ID、位置、概率等）。

## 3. 技术架构

### 3.1 模块划分

```
KvcacheVisualization/
├── app.py                    # Streamlit 主应用（增加主题、i18n）
├── model_loader.py           # 模型加载（扩展 T5 支持）
├── kvcache_extractor.py     # KV Cache 捕获（扩展架构检测）
├── kvcache_simulator.py      # 状态管理
├── visualizer.py             # 可视化（增量更新、主题）
├── i18n.py                   # 国际化（中英双语）
├── theme.py                  # 主题管理（暗色/亮色）
├── exporter.py               # 数据导出（JSON/CSV）
└── prompts.py                # 预设 Prompt 模板
```

### 3.2 主题切换实现

```python
def get_theme():
    return st.session_state.get("theme", "light")

def apply_theme(theme_name):
    colors = THEMES[theme_name]
    st.markdown(f"""
    <style>
        body {{ background-color: {colors['background']}; color: {colors['text']}; }}
        .stApp {{ background-color: {colors['background']}; }}
    </style>
    """, unsafe_allow_html=True)
```

## 4. 验收标准

- [ ] 支持 T5/FLAN 模型加载
- [ ] KV Cache 数据可导出为 JSON 和 CSV
- [ ] 支持中英双语切换
- [ ] 支持暗色/亮色主题切换
- [ ] 预设 Prompt 模板可用
- [ ] 批量生成功能正常
- [ ] 可视化支持增量更新
- [ ] Attention 分布图正常显示
- [ ] 层级能量热力图正常显示
- [ ] Token 详情悬浮显示

## 5. 实施顺序

1. **功能增强** - 多架构支持、导出功能
2. **性能优化** - 批量生成、增量更新
3. **用户体验** - 主题切换、双语界面、模板
4. **调试功能** - Attention 分布、层级热力图
