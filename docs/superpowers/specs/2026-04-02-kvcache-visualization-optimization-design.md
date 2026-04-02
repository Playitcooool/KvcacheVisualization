# KV Cache 可视化器 - 全面优化设计

## 1. 项目概述

**项目名称**: KvcacheVisualization
**类型**: 交互式可视化工具
**核心功能**: 加载真实 LLM 模型，实时可视化推理过程中 KV Cache 的结构和动态变化，支持多模型对比
**目标用户**: 研究人员、学习者、开发者

---

## 2. 优化目标

### 2.1 代码质量

| 问题 | 解决方案 |
|------|----------|
| 大量 `print("[DEBUG] ...")` 调试语句 | 用 `logging` 模块替代，统一日志级别 |
| 代码重复（app.py 过长） | 模块化拆分，UI/逻辑分离 |
| 流式生成逻辑复杂 | 简化状态管理，提取独立函数 |
| 错误处理分散 | 统一异常处理类，分类捕获 |
| 类型提示不足 | 添加完整的类型注解 |

### 2.2 可视化增强

| 新增可视化 | 描述 |
|-----------|------|
| **Attention Pattern** | 捕获并展示 attention softmax 权重矩阵热力图 |
| **Layer Energy Heatmap** | 增强：各层各位置的 KV 能量分布 |
| **Token Importance** | 每个 token 的 KV 能量排序和重要性 |
| **Multi-Model Comparison** | 分屏/叠加/统计三种对比模式 |

### 2.3 多模型对比

**默认行为**: 加载 1 个模型，用户可手动添加第 2 个模型进行对比

**三种对比模式**:
| 模式 | 描述 |
|------|------|
| **分屏对比 (Split)** | 左右两个完整面板，1:1 或可调比例 |
| **叠加对比 (Overlay)** | 同一图表叠加不同模型的曲线，颜色区分 |
| **统计对比 (Stats)** | 表格/柱状图展示多模型统计指标 |

---

## 3. 技术架构

### 3.1 目录结构

```
KvcacheVisualization/
├── app.py                      # Streamlit 主应用（精简至 <500 行）
├── core/                       # 核心逻辑模块
│   ├── __init__.py
│   ├── model_manager.py        # 多模型生命周期管理
│   ├── model_base.py           # 抽象基类
│   ├── kvcache_extractor.py    # KV Cache + Attention 捕获
│   ├── kvcache_simulator.py    # 状态管理与回放
│   └── generator.py            # 生成逻辑（流式/非流式）
├── visualization/              # 可视化模块
│   ├── __init__.py
│   ├── base.py                 # 可视化基类
│   ├── attention.py            # Attention Pattern 可视化
│   ├── layer_energy.py         # Layer Energy 热力图
│   ├── sequence.py             # Token 序列可视化
│   ├── stats.py                # 统计指标可视化
│   └── comparison.py            # 多模型对比可视化
├── ui/                         # UI 组件
│   ├── __init__.py
│   ├── components.py           # 通用 UI 组件
│   ├── sidebar.py              # 侧边栏
│   ├── layout.py               # 布局组件
│   └── themes.py               # 主题管理
└── utils/                      # 工具模块
    ├── __init__.py
    ├── logger.py               # 日志配置
    ├── device.py               # 设备管理
    ├── i18n.py                 # 国际化
    └── exporter.py             # 导出功能
```

### 3.2 核心类设计

#### ModelManager
```python
class ModelManager:
    """管理多个模型的生命周期"""

    def __init__(self):
        self.models: Dict[str, ModelHandle] = {}  # {id: handle}
        self.active_id: Optional[str] = None

    def add_model(self, name: str, model_type: str, **kwargs) -> ModelHandle
    def remove_model(self, id: str) -> None
    def get_active(self) -> Optional[ModelHandle]
    def set_active(self, id: str) -> None
```

#### ModelHandle
```python
@dataclass
class ModelHandle:
    """单个模型的句柄"""
    id: str
    name: str
    model: Any
    tokenizer: Any
    config: Dict
    extractor: KVCacheExtractor
    simulator: KVCacheSimulator
    visualizer: KVCacheVisualizer
```

#### KVCacheExtractor (增强)
```python
class KVCacheExtractor:
    """新增 Attention Weight 捕获"""

    def __init__(self, ...):
        # 现有字段...
        self._attn_weights_history: List[torch.Tensor] = []

    def register_hooks(self, model: torch.nn.Module) -> List[Any]:
        """增强：同时捕获 KV Cache 和 Attention Weights"""

    def get_attention_weights(self) -> List[torch.Tensor]:
        """获取捕获的 attention weights 历史"""
```

### 3.3 日志系统

```python
# utils/logger.py
import logging

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """统一日志配置"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # 格式: "[2024-01-01 12:00:00] [ModuleName] INFO: message"
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(name)s] %(levelname)s: %(message)s"
    ))
    logger.addHandler(handler)
    return logger

# 使用示例
logger = setup_logger(__name__)
logger.info("Model loaded successfully")
logger.debug("KV cache entry captured at position %d", pos)
```

---

## 4. 功能规格

### 4.1 Attention Pattern 可视化

**捕获机制**:
- 通过 register_forward_hook 捕获 attention 层的输入（Q, K, V）
- 计算 attention scores: `softmax(Q @ K^T / sqrt(d_k))`
- 历史记录: `attn_weights_history: List[Tensor[batch, heads, seq, seq]]`

**可视化类型**:
1. **Single Attention Heatmap**: 单个位置的 attention pattern
2. **Attention Animation**: 沿生成过程的 attention 变化
3. **Average Attention**: 多层平均的 attention pattern

### 4.2 多模型对比

**UI 流程**:
```
[模型 A] ←── 主模型，始终加载
    ↓
[添加对比模型] 按钮
    ↓
[模型 B] ←── 对比模型，可选
    ↓
选择对比模式: [分屏] [叠加] [统计]
```

**分屏对比 (Split)**:
- 左右 1:1 布局
- 左侧: 模型 A 的完整面板
- 右侧: 模型 B 的完整面板
- 可调节比例 (1:2, 1:1, 2:1)

**叠加对比 (Overlay)**:
- 同一图表，不同颜色曲线
- 模型 A: 蓝色系
- 模型 B: 红色系
- 图例标识

**统计对比 (Stats)**:
- 并排柱状图
- 指标: Cache 效率、峰值内存、平均层能量、稀疏度

### 4.3 简化的流式生成

```python
class GenerationController:
    """简化的生成控制器"""

    def __init__(self, model_handle: ModelHandle):
        self.handle = model_handle
        self.state: Literal["idle", "running", "paused", "complete"] = "idle"

    def start(self, prompt: str, max_tokens: int, batch_size: int):
        """开始生成"""

    def pause(self):
        """暂停生成"""

    def resume(self):
        """继续生成"""

    def reset(self):
        """重置状态"""
```

**状态转换**:
```
idle → running → complete
         ↓ ↑
      paused
         ↓
       idle (reset)
```

---

## 5. UI/UX 设计

### 5.1 布局结构

```
┌──────────────────────────────────────────────────────────────┐
│  KV Cache 可视化器                    [主题] [语言]          │
├────────────┬─────────────────────────────────────────────────┤
│            │                                                  │
│  模型选择   │   主内容区                                        │
│  ───────   │   ─────────────────────────────────────────      │
│  模型 A    │                                                  │
│  [已加载]  │   左侧: LLM 生成区域 / 模型 A 生成区域             │
│            │   右侧: KV Cache 可视化 / Attention 可视化         │
│  [+ 添加]  │                                                  │
│            │   Tab: [序列] [层级] [Attention] [统计] [对比]     │
│  ───────   │                                                  │
│  生成设置   │                                                  │
│  ───────   │                                                  │
│  Prompt    │                                                  │
│  MaxTokens │                                                  │
│  [开始]    │                                                  │
│            │                                                  │
└────────────┴─────────────────────────────────────────────────┘
```

### 5.2 主题系统

**亮色主题**:
- 背景: #FFFFFF
- 主色: #1f77b4 (蓝色)
- 强调色: #ff7f0e (橙色)
- 文字: #1a1a1a

**暗色主题**:
- 背景: #1a1a2e
- 主色: #4a9eff (亮蓝)
- 强调色: #ff9f43 (亮橙)
- 文字: #e0e0e0

### 5.3 响应式设计

- 支持窄屏模式（侧边栏折叠）
- 可视化区域自适应宽度
- 移动端可用性（基础支持）

---

## 6. 性能优化

### 6.1 内存优化

| 优化项 | 方案 |
|--------|------|
| Tensor 克隆 | 仅在必要时克隆，使用 view 而不是 copy |
| 历史限制 | MAX_HISTORY_LENGTH = 100，可配置 |
| GPU 内存 | 及时 `torch.cuda.empty_cache()` |
| 批量更新 | 每 N 个 token 更新一次可视化 |

### 6.2 渲染优化

- 使用 `st.empty()` 原地更新图表
- 限制回放 slider 精度（不需要 0.1 精度）
- 延迟加载非可见 Tab 的内容

---

## 7. 迁移计划

### Phase 1: 代码重构
1. 创建新目录结构
2. 提取 `utils/logger.py`
3. 重构 `model_manager.py`
4. 清理 `kvcache_extractor.py` 和 `kvcache_simulator.py`
5. 简化 `app.py`

### Phase 2: 可视化增强
1. 实现 Attention Pattern 捕获
2. 实现 Attention 热力图可视化
3. 增强 Layer Energy 热力图
4. 添加 Token Importance 可视化

### Phase 3: 多模型对比
1. 实现 ModelManager 多模型支持
2. 实现三种对比模式 UI
3. 实现对比可视化组件

### Phase 4: 测试与调优
1. 功能测试
2. 性能测试
3. UI/UX 优化

---

## 8. 验收标准

- [ ] 所有 debug print 语句已移除，使用 logging 替代
- [ ] app.py 精简至 500 行以内
- [ ] Attention Pattern 可视化正常工作
- [ ] 支持默认加载 1 个模型，可添加第 2 个模型
- [ ] 三种对比模式（分屏/叠加/统计）均可正常工作
- [ ] 流式生成逻辑简化，无状态混乱
- [ ] 所有核心模块有类型注解
- [ ] 应用可正常启动和运行

---

## 9. 后续扩展 (Future)

- [ ] 导出可视化图表为图片
- [ ] 自定义 Prompt 模板管理
- [ ] KV Cache 动画导出为 GIF
- [ ] 更多模型架构支持 (Mistral, Gemma, etc.)
- [ ] 移动端优化
