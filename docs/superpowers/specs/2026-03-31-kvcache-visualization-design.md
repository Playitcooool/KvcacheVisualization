# KV Cache 可视化项目设计

## 1. 项目概述

**项目名称**: KvcacheVisualization
**类型**: 交互式可视化工具
**核心功能**: 加载真实 LLM 模型，实时（伪实时）可视化推理过程中 KV Cache 的结构和动态变化
**目标用户**: 学习和理解 LLM 内部机制的研究者/开发者

---

## 2. 功能规格

### 2.1 核心功能

| 功能 | 描述 |
|------|------|
| **模型加载** | 支持 HuggingFace (GPT-2, TinyLlama 等) 和原生 PyTorch checkpoint |
| **真实推理** | 使用真实模型前向传播，捕获真实 KV Cache |
| **KV Cache 可视化** | 四种视图：矩阵热力图、序列生成动画、层级分布、综合仪表盘 |
| **同步联动** | 左侧 token 生成 ↔ 右侧 KV Cache 可视化实时对应 |
| **伪实时更新** | 每生成 N 个 token 批量更新可视化（可配置） |
| **回放拖动** | 生成完成后，可拖动进度条回看任意 token 位置的 KV Cache 状态 |

### 2.2 模型加载方式

#### HuggingFace 模式
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

model_name = "gpt2"  # 或 " TinyLlama/TinyLlama-1.1B-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

#### PyTorch 原生模式
```python
import torch

# 加载 .pt/.pth checkpoint
state_dict = torch.load("gpt2-pytorch_model.bin")
model.load_state_dict(state_dict)
```

### 2.3 KV Cache 捕获机制

使用 PyTorch Hook 在 Forward 过程中捕获 attention 层的 K/V tensor：

```python
# 注册 forward hook 到 attention 层
def hook_fn(module, input, output):
    k, v = output  # attention 层的输出
    kvcache_history.append((k.clone(), v.clone()))

# 注册到每一层
for layer in model.transformer.h:
    layer.register_forward_hook(hook_fn)
```

### 2.4 可视化维度

1. **矩阵结构视图 (Heatmap)**
   - 展示 K/V 矩阵数值分布
   - 行=token位置, 列=隐层维度
   - 颜色映射：蓝色(低) → 红色(高)

2. **序列生成视图 (Sequence)**
   - 沿时间轴展示 token 生成过程
   - 每个 token 位置对应一个 mini 热力条
   - 新 token 出现时有动画

3. **层级分布视图 (Layer)**
   - 展示 12/24/32 个 transformer 层的注意力分布
   - 层间对比：不同层的 KV 数值差异

4. **综合仪表盘 (Dashboard)**
   - 同时展示以上三种视图
   - 实时更新当前 token 位置指示

### 2.5 统计数据指标

KV Cache 可视化仪表盘包含以下实时统计数据：

| 指标 | 描述 | 计算方式 |
|------|------|----------|
| **生成 Token 数** | 已生成的 token 总数 | 计数器 |
| **缓存 Token 数** | KV Cache 中存储的 token 位置数 | 历史长度 |
| **Cache 效率** | 缓存命中率，衡量重复计算避免程度 | `缓存命中 / 总注意力计算` |
| **峰值内存** | KV Cache 占用的 GPU/CPU 内存 | tensor size 统计 |
| **层间分布** | 各层 KV Cache 能量均值 | mean(L2 norm) per layer |
| **Attention 稀疏度** | K/V 矩阵中接近零的比例 | `count(~0) / total` |

**Cache 效率计算公式：**
```
效率 = (缓存命中的 token 数) / (如果没有缓存需要的计算数)
     ≈ len(cache) / (1 + 2 + ... + len(cache))
     = 2 / (len(cache) + 1)
```

**稀疏度计算公式：**
```
稀疏度 = count(|value| < threshold) / total_count
       ≈ count(|value| < 0.01) / total_count
```

### 2.6 模拟数据规格

```python
# 模拟 LLM 推理参数
num_layers = 12
num_heads = 12
head_dim = 64
max_seq_len = 50

# KV Cache shape: [num_layers, 2, batch, num_heads, seq_len, head_dim]
# 生成一段短 query: "Hello, how are you?"
```

---

## 3. 技术架构

### 3.1 技术栈

- **前端**: Streamlit (Python Web 框架)
- **可视化**: Plotly (交互式图表)
- **模型加载**: HuggingFace `transformers` + PyTorch 原生
- **KV Cache 捕获**: PyTorch Hook 机制
- **内存优化**: 梯度关闭 (torch.no_grad())
- **设备支持**: CUDA / MPS / CPU 自动检测 + 手动选择

### 3.2 设备管理

支持三种计算设备，自动检测可用设备：

```python
def get_available_device():
    """自动检测可用设备，优先级: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps:0")
    else:
        return torch.device("cpu")

def get_device_from_string(device_str: str):
    """从字符串解析设备"""
    if device_str == "auto":
        return get_available_device()
    elif device_str.startswith("cuda"):
        return torch.device(device_str)
    elif device_str == "mps":
        return torch.device("mps:0")
    else:
        return torch.device("cpu")
```

**设备支持矩阵：**

| 设备 | 模型加载 | Hook 捕获 | 可视化 |
|------|----------|-----------|--------|
| CUDA | ✅ | ✅ (GPU tensor) | 转 CPU |
| MPS | ✅ | ✅ (MPS tensor) | 转 CPU |
| CPU | ✅ | ✅ (CPU tensor) | 直接用 |

**Streamlit UI 设备选择：**
```
设备: [自动检测 ▼]
      - auto: 自动检测 (默认)
      - cuda:0: NVIDIA GPU
      - mps:0: Apple Silicon
      - cpu: CPU
```

### 3.3 模块划分

```
KvcacheVisualization/
├── requirements.txt        # 依赖声明
├── model_loader.py         # 模型加载抽象层
├── kvcache_extractor.py    # KV Cache Hook 捕获逻辑
├── kvcache_simulator.py    # KV Cache 状态管理和回放
├── visualizer.py           # Plotly 可视化组件
├── app.py                  # Streamlit 主应用
└── tests/
    ├── test_model_loader.py
    ├── test_kvcache_extractor.py
    └── test_visualizer.py
```

### 3.3 数据流

```
┌─────────────────────────────────────────────────────────────┐
│                     model_loader.py                         │
│   HuggingFace AutoModel / PyTorch state_dict              │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                  kvcache_extractor.py                       │
│   注册 Hook → Forward Pass → 捕获 K/V tensor               │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                  kvcache_simulator.py                       │
│   管理历史状态 → 支持回放 → 输出当前可视化的 slice          │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                     visualizer.py                          │
│   Plotly: 热力图 / 序列图 / 层级分布 / 仪表盘              │
└─────────────────────────────────────────────────────────────┘
```

### 3.4 布局结构

```
┌─────────────────────────────────────────────────────────────┐
│  KV Cache 可视化器          [模型: GPT-2]    [重新生成]    │
├────────────────────────────┬────────────────────────────────┤
│                            │                                │
│   LLM 生成区域             │   KV Cache 可视化区域          │
│                            │                                │
│  当前 Token: "how"         │   [矩阵热力图]                 │
│                            │   [序列动画]                   │
│  已生成: Hello, how         │   [层级分布]                   │
│                            │   [综合仪表盘 Tab]             │
│  ● ○ ○ ○ ○ ● (进度)        │                                │
│  [生成中... 5/50]          │   [Token 1] [2] [3] ...        │
│                            │                                │
│  ━━━━━━━━━━━━━━━━━━━━      │   ← 进度条可拖动回放 →          │
│                            │                                │
└────────────────────────────┴────────────────────────────────┘
```

---

## 4. 交互设计

### 4.1 操作流程

1. **启动**: `streamlit run app.py`
2. **选择模型**: 默认 GPT-2，可切换 HuggingFace 模型名或 PyTorch checkpoint 路径
3. **输入 Prompt**: "Hello, how are you?"
4. **点击"开始生成"**: 真实模型推理，KV Cache Hook 捕获
5. **伪实时更新**: 每 N 个 token 批量更新可视化（默认 5）
6. **生成完成**: 进度条解锁，可拖动任意回放
7. **Tab 切换**: 矩阵/序列/层级/仪表盘 四种视图

### 4.2 控制项

| 控件 | 行为 |
|------|------|
| 模型选择器 | HuggingFace 模型名 或 PyTorch checkpoint 路径 |
| "开始生成" 按钮 | 启动真实推理 + KV Cache 捕获 |
| "重新生成" 按钮 | 清空状态，重新开始 |
| Tab 选择器 | 切换可视化视图 |
| 进度条 | 生成完成后可拖动回放 |
| 批量大小滑块 | 控制每批生成 token 数量 |

---

## 5. 验收标准

- [ ] Streamlit 应用能正常启动 (`streamlit run app.py`)
- [ ] 能加载 HuggingFace GPT-2 模型
- [ ] 能加载 PyTorch .pt checkpoint
- [ ] Forward 过程中 Hook 正确捕获 KV Cache
- [ ] 左侧显示真实模型生成的 token
- [ ] 右侧 KV Cache 热力图正确渲染
- [ ] 每 N token 批量更新一次（可配置）
- [ ] 生成完成后，进度条可拖动回放
- [ ] Tab 切换：矩阵/序列/层级/仪表盘 四种视图正常

---

## 6. 后续计划

1. 实现 `model_loader.py` — HuggingFace / PyTorch 抽象加载层
2. 实现 `kvcache_extractor.py` — Hook 捕获 KV Cache
3. 实现 `kvcache_simulator.py` — 状态管理和回放（保留）
4. 实现 `visualizer.py` — Plotly 四种图表
5. 实现 `app.py` — Streamlit 界面组装
6. 测试真实模型推理和 KV Cache 捕获
