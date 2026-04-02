# KV Cache 可视化器

实时可视化 LLM 生成过程中的 KV Cache 结构与动态变化，支持 Attention Pattern 分析、多模型对比、层间统计对比。

## 功能特性

### 核心功能
- **Hook 注入提取** — PyTorch Forward Hook 捕获 Attention 层的 K/V Tensor 及 Attention Weights
- **状态管理与回放** — 存储历史记录，支持拖动进度条回放到任意生成位置
- **多种可视化图表** — 热力图、序列视图、层级分布、统计仪表盘、层级能量

### Attention 分析（Phase 2+）
- **Attention Pattern 热力图** — 实时捕获并展示 Attention Softmax 权重矩阵
- **Head 分布视图** — 分 Head 查看各自的 Attention Pattern
- **Attention Summary** — 生成过程中 Attention 模式变化追踪

### Token 分析（Phase 2+）
- **Token 重要性** — 按 KV Energy 排序，识别最关键的生成 Token

### 层分析（Phase 5）
- **层能量变化折线图** — 追踪生成过程中各层 KV Energy 的变化曲线
- **Attention 层间统计** — 按层统计覆盖度、稀疏度、最大值指标

### 多模型对比（Phase 3+）
- **分屏对比** — 左右并排显示两个模型的完整面板
- **叠加对比** — 同一图表叠加不同模型的曲线
- **统计对比** — 柱状图对比各项指标

### 其他
- **渐进式渲染** — 长序列时初始只渲染前 50 Token，"加载更多"逐步展开
- **可配置缓存大小** — 侧边栏可调节 KV Cache 历史记录条数（10-500）
- **流式/非流式生成** — 支持批量生成和流式逐 Token 生成
- **数据导出** — 支持 JSON/CSV 格式导出

## 架构

```
KvcacheVisualization/
├── app.py                      # Streamlit 主应用
├── kvcache_extractor.py        # Hook 注入，捕获 K/V 及 Attention Weights
├── kvcache_simulator.py        # 状态管理，缓存回放
├── visualizer.py               # Plotly 可视化渲染
├── model_loader.py             # 模型加载（HuggingFace / PyTorch）
├── visualization/
│   └── comparison.py           # 多模型对比可视化
├── ui/
│   ├── components.py           # UI 组件
│   ├── sidebar.py             # 侧边栏
│   └── layout.py              # 布局组件
└── utils/
    ├── logger.py               # 统一日志
    ├── device.py               # 设备检测（CUDA / MPS / CPU）
    ├── i18n.py               # 国际化
    └── exporter.py            # 数据导出
```

## 安装

```bash
pip install -r requirements.txt
```

## 使用

### 启动 Web 应用

```bash
streamlit run app.py
```

### 操作流程

1. 在侧边栏选择设备（HuggingFace / PyTorch）和模型
2. 点击「加载模型」
3. 输入 Prompt，点击「开始生成」
4. 观察右侧 KV Cache 可视化区域
5. 生成完成后，拖动进度条回放任意位置
6. 切换不同 Tab 查看：序列视图 / 层级分布 / 统计数据 / 综合仪表盘 / 层级能量 / Attention / Token重要性 / 层分析

### 可视化 Tab 说明

| Tab | 说明 |
|-----|------|
| 📈 序列视图 | Token 序列及每步的能量 |
| 🔳 层级分布 | 当前层的 K/V 矩阵热力图 |
| 📐 统计数据 | Cache 效率、峰值内存、稀疏度等指标 |
| 🖥️ 综合仪表盘 | 多指标综合展示 |
| 🔥 层级能量 | 各层各位置的 KV 能量分布热力图 |
| 🧠 Attention | Attention Pattern 热力图（整体/Head/变化追踪） |
| ⭐ Token重要性 | 按 KV Energy 排序的 Token 重要性 |
| 📊 层分析 | 层能量变化折线图 + Attention 层间统计对比 |

### 多模型对比

1. 加载第一个模型并生成
2. 侧边栏点击「添加对比模型」加载第二个模型
3. 生成完成后，展开「📊 模型对比」面板
4. 选择对比模式：分屏 / 叠加 / 统计

### 作为库使用

```python
from kvcache_extractor import KVCacheExtractor
from kvcache_simulator import KVCacheSimulator
from visualizer import KVCacheVisualizer

# 初始化
extractor = KVCacheExtractor(num_layers=12, num_heads=12, head_dim=64)
simulator = KVCacheSimulator(num_layers=12, num_heads=12, head_dim=64)
visualizer = KVCacheVisualizer()

# 注册 Hook
handles = extractor.register_hooks(model)

# 生成文本...

# 移除 Hook
for handle in handles:
    handle.remove()

# 可视化
k_cache = extractor.kvcache_history[0].k_cache
fig = visualizer.create_heatmap(k_cache)
fig.show()
```

## 依赖

- `streamlit >= 1.28.0`
- `plotly >= 5.18.0`
- `torch >= 2.0.0`
- `transformers >= 4.36.0`
- `numpy >= 1.24.0`
- `accelerate >= 0.25.0`
- `safetensors >= 0.4.0`

## 适用场景

- 学习 LLM 推理过程中 KV Cache 的工作机制
- 分析 Attention Pattern 和 Token 重要性
- 对比不同模型的 KV Cache 行为差异
- 调试和优化 LLM 推理性能
