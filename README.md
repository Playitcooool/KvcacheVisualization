# KV Cache 可视化器

实时可视化 LLM 生成过程中的 KV Cache 状态，支持 Hook 注入、缓存播放与统计。

## 功能特性

- **Hook 注入提取** - 使用 PyTorch Forward Hook 捕获 Attention 层的 K/V Tensor
- **状态管理与回放** - 存储历史记录，支持拖动回放到任意生成位置
- **多种可视化图表** - 热力图、序列视图、层级分布、统计仪表盘
- **模型兼容** - 支持 HuggingFace 模型（Qwen、Llama 等）和自定义 PyTorch 模型
- **交互式 Web 界面** - 基于 Streamlit 的实时可视化面板

## 架构

```
kvcache_extractor.py   # Hook 注入，捕获 Attention K/V
kvcache_simulator.py  # 状态管理，缓存回放
visualizer.py         # Plotly 可视化渲染
model_loader.py       # 模型加载（HuggingFace / PyTorch）
device_utils.py       # 设备检测（CUDA / MPS / CPU）
app.py                # Streamlit Web 应用
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

1. 在侧边栏选择设备（HuggingFace / PyTorch）和模型
2. 点击「加载模型」
3. 输入 Prompt，点击「开始生成」
4. 观察右侧 KV Cache 可视化区域
5. 生成完成后，拖动进度条回放任意位置

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
