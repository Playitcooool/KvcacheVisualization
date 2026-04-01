# KV Cache 可视化器优化实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 对 KvcacheVisualization 进行全面优化：支持多架构、导出数据、双语界面、暗色主题、预设模板、性能优化、调试功能

**Architecture:** 新增 4 个模块（i18n.py, theme.py, exporter.py, prompts.py），扩展 3 个现有模块（model_loader.py, kvcache_extractor.py, visualizer.py），更新 app.py 集成所有功能

**Tech Stack:** Streamlit, Plotly, PyTorch, Transformers, pandas

---

## 文件结构

```
KvcacheVisualization/
├── i18n.py                    # 新增：国际化（中英双语）
├── theme.py                   # 新增：主题管理（暗色/亮色）
├── exporter.py                # 新增：数据导出（JSON/CSV）
├── prompts.py                 # 新增：预设 Prompt 模板
├── model_loader.py            # 修改：扩展 T5/FLAN 支持
├── kvcache_extractor.py       # 修改：架构自动检测
├── visualizer.py              # 修改：增量更新、主题适配
├── app.py                     # 修改：集成所有新功能
├── requirements.txt           # 修改：添加 pandas 依赖
└── tests/                     # 新增各模块测试
```

---

## Task 1: 国际化支持 (i18n.py)

**Files:**
- Create: `i18n.py`
- Test: `tests/test_i18n.py`

- [ ] **Step 1: 创建 i18n.py**

```python
# i18n.py
from typing import Dict

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "zh": {
        # 侧边栏
        "model_settings": "模型设置",
        "device": "计算设备",
        "auto_detect": "自动检测",
        "model_source": "模型来源",
        "huggingface": "HuggingFace",
        "local_model": "本地模型",
        "select_model": "选择模型",
        "model_path": "模型路径",
        "load_model": "加载模型",
        "model_info": "模型信息",
        "layers": "层数",
        "heads": "头数",
        "hidden_size": "隐藏维度",
        "current_device": "当前设备",
        "instructions": "使用说明",

        # 主界面
        "llm_generation": "LLM 生成区域",
        "input_prompt": "输入 Prompt",
        "batch_size": "生成 Token 数",
        "start_generation": "开始生成",
        "regenerate": "重新生成",
        "generation_result": "生成结果",
        "generation_complete": "生成完成",
        "playback_control": "回放控制",
        "drag_to_seek": "拖动进度条回看任意位置",
        "position": "位置",

        # 可视化
        "kvcache_visualization": "KV Cache 可视化区域",
        "generate_first": "先生成 token 后才能查看 KV Cache 可视化",
        "matrix_heatmap": "矩阵热力图",
        "sequence_view": "序列视图",
        "layer_distribution": "层级分布",
        "statistics": "统计数据",
        "dashboard": "综合仪表盘",

        # 统计指标
        "generated_tokens": "生成 Token 数",
        "cached_tokens": "缓存 Token 数",
        "cache_efficiency": "Cache 效率",
        "peak_memory": "峰值内存",
        "avg_layer_energy": "平均层能量",
        "attention_sparsity": "Attention 稀疏度",

        # 导出
        "export_data": "导出数据",
        "export_json": "导出 JSON",
        "export_csv": "导出 CSV",

        # 模板
        "prompt_templates": "预设模板",
        "qa": "问答",
        "translation": "翻译",
        "code_completion": "代码补全",
        "story_continuation": "故事续写",

        # 提示信息
        "model_loaded": "模型加载成功",
        "generation_failed": "生成失败",
        "loading_model": "加载中，请稍候...",
        "select_model_first": "请先在侧边栏加载模型",
        "click_to_start": "点击「开始生成」按钮启动",

        # 主题
        "theme": "主题",
        "light_theme": "亮色",
        "dark_theme": "暗色",

        # 调试
        "layer_energy_heatmap": "层级能量热力图",
        "token_details": "Token 详情",
        "token_id": "Token ID",
        "token_index": "位置索引",
    },
    "en": {
        # Sidebar
        "model_settings": "Model Settings",
        "device": "Device",
        "auto_detect": "Auto Detect",
        "model_source": "Model Source",
        "huggingface": "HuggingFace",
        "local_model": "Local Model",
        "select_model": "Select Model",
        "model_path": "Model Path",
        "load_model": "Load Model",
        "model_info": "Model Info",
        "layers": "Layers",
        "heads": "Heads",
        "hidden_size": "Hidden Size",
        "current_device": "Device",
        "instructions": "Instructions",

        # Main UI
        "llm_generation": "LLM Generation",
        "input_prompt": "Input Prompt",
        "batch_size": "Tokens to Generate",
        "start_generation": "Start Generation",
        "regenerate": "Regenerate",
        "generation_result": "Generated Result",
        "generation_complete": "Generation Complete",
        "playback_control": "Playback Control",
        "drag_to_seek": "Drag to seek any position",
        "position": "Position",

        # Visualization
        "kvcache_visualization": "KV Cache Visualization",
        "generate_first": "Generate tokens first to view KV Cache",
        "matrix_heatmap": "Matrix Heatmap",
        "sequence_view": "Sequence View",
        "layer_distribution": "Layer Distribution",
        "statistics": "Statistics",
        "dashboard": "Dashboard",

        # Statistics
        "generated_tokens": "Generated Tokens",
        "cached_tokens": "Cached Tokens",
        "cache_efficiency": "Cache Efficiency",
        "peak_memory": "Peak Memory",
        "avg_layer_energy": "Avg Layer Energy",
        "attention_sparsity": "Attention Sparsity",

        # Export
        "export_data": "Export Data",
        "export_json": "Export JSON",
        "export_csv": "Export CSV",

        # Templates
        "prompt_templates": "Prompt Templates",
        "qa": "Q&A",
        "translation": "Translation",
        "code_completion": "Code Completion",
        "story_continuation": "Story Continuation",

        # Messages
        "model_loaded": "Model loaded successfully",
        "generation_failed": "Generation failed",
        "loading_model": "Loading, please wait...",
        "select_model_first": "Please load a model in the sidebar first",
        "click_to_start": "Click 'Start Generation' to begin",

        # Theme
        "theme": "Theme",
        "light_theme": "Light",
        "dark_theme": "Dark",

        # Debug
        "layer_energy_heatmap": "Layer Energy Heatmap",
        "token_details": "Token Details",
        "token_id": "Token ID",
        "token_index": "Index",
    }
}

def get_text(key: str, lang: str = "zh") -> str:
    """获取翻译文本"""
    return TRANSLATIONS.get(lang, TRANSLATIONS["zh"]).get(key, key)

def t(key: str, lang: str = "zh") -> str:
    """翻译的简写"""
    return get_text(key, lang)
```

- [ ] **Step 2: 创建测试文件 tests/test_i18n.py**

```python
# tests/test_i18n.py
import pytest
from i18n import get_text, t, TRANSLATIONS

def test_get_text_zh():
    assert get_text("model_settings", "zh") == "模型设置"

def test_get_text_en():
    assert get_text("model_settings", "en") == "Model Settings"

def test_get_text_default():
    assert get_text("model_settings") == "模型设置"  # 默认中文

def test_get_text_unknown():
    assert get_text("unknown_key") == "unknown_key"  # 回退到 key

def test_t_shorthand():
    assert t("model_settings", "en") == "Model Settings"

def test_all_keys_have_translation():
    """确保所有 key 都有中英文翻译"""
    zh_keys = set(TRANSLATIONS["zh"].keys())
    en_keys = set(TRANSLATIONS["en"].keys())
    assert zh_keys == en_keys, "中英文 key 不匹配"
```

- [ ] **Step 3: 运行测试验证**

Run: `pytest tests/test_i18n.py -v`
Expected: 5 PASS

- [ ] **Step 4: 提交**

```bash
git add i18n.py tests/test_i18n.py
git commit -m "feat: add i18n support for Chinese/English"
```

---

## Task 2: 主题管理 (theme.py)

**Files:**
- Create: `theme.py`
- Test: `tests/test_theme.py`

- [ ] **Step 1: 创建 theme.py**

```python
# theme.py
from typing import Dict, TypedDict

class ThemeColors(TypedDict):
    background: str
    surface: str
    text: str
    primary: str
    secondary: str
    chart_colors: list
    border: str

THEMES: Dict[str, ThemeColors] = {
    "light": ThemeColors(
        background="#ffffff",
        surface="#f8f9fa",
        text="#31333F",
        primary="#1f77b4",
        secondary="#ff7f0e",
        chart_colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
        border="#dee2e6"
    ),
    "dark": ThemeColors(
        background="#0e1117",
        surface="#1e2127",
        text="#f0f0f0",
        primary="#66b3ff",
        secondary="#ffb366",
        chart_colors=["#66b3ff", "#ffb366", "#99ff99", "#ff6666", "#b366ff"],
        border="#3d4450"
    )
}

def get_theme(theme_name: str = "light") -> ThemeColors:
    """获取主题颜色配置"""
    return THEMES.get(theme_name, THEMES["light"])

def get_theme_css(theme_name: str = "light") -> str:
    """生成主题 CSS"""
    colors = get_theme(theme_name)
    return f"""
    <style>
    :root {{
        --background: {colors['background']};
        --surface: {colors['surface']};
        --text: {colors['text']};
        --primary: {colors['primary']};
        --secondary: {colors['secondary']};
        --border: {colors['border']};
    }}
    .stApp {{
        background-color: {colors['background']};
    }}
    .main-header {{
        color: {colors['primary']};
    }}
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {colors['surface']};
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {colors['text']};
    }}
    </style>
    """

def get_plotly_template(theme_name: str = "light") -> str:
    """获取 Plotly 图表模板"""
    colors = get_theme(theme_name)
    if theme_name == "dark":
        return "plotly_dark"
    return "plotly"
```

- [ ] **Step 2: 创建测试 tests/test_theme.py**

```python
# tests/test_theme.py
import pytest
from theme import get_theme, get_theme_css, get_plotly_template, THEMES

def test_get_theme_light():
    theme = get_theme("light")
    assert theme["background"] == "#ffffff"
    assert theme["text"] == "#31333F"

def test_get_theme_dark():
    theme = get_theme("dark")
    assert theme["background"] == "#0e1117"
    assert theme["text"] == "#f0f0f0"

def test_get_theme_unknown():
    theme = get_theme("unknown")
    assert theme == THEMES["light"]  # 回退到 light

def test_get_theme_css():
    css = get_theme_css("light")
    assert "#ffffff" in css
    assert "#1f77b4" in css

def test_get_plotly_template_light():
    assert get_plotly_template("light") == "plotly"

def test_get_plotly_template_dark():
    assert get_plotly_template("dark") == "plotly_dark"
```

- [ ] **Step 3: 运行测试**

Run: `pytest tests/test_theme.py -v`
Expected: 5 PASS

- [ ] **Step 4: 提交**

```bash
git add theme.py tests/test_theme.py
git commit -m "feat: add theme support (light/dark mode)"
```

---

## Task 3: 数据导出 (exporter.py)

**Files:**
- Create: `exporter.py`
- Test: `tests/test_exporter.py`

- [ ] **Step 1: 创建 exporter.py**

```python
# exporter.py
import json
import pandas as pd
from typing import List, Any, Dict
import torch

def export_kvcache_to_dict(
    tokens: List[str],
    k_cache_list: List[torch.Tensor],
    v_cache_list: List[torch.Tensor],
    stats: Dict[str, Any]
) -> Dict[str, Any]:
    """导出 KV Cache 数据为字典"""
    return {
        "tokens": tokens,
        "token_count": len(tokens),
        "k_cache_summary": [
            {
                "position": i + 1,
                "shape": list(k.shape),
                "l2_norm": torch.norm(k).item() if k.numel() > 0 else 0,
            }
            for i, k in enumerate(k_cache_list)
        ],
        "v_cache_summary": [
            {
                "position": i + 1,
                "shape": list(v.shape),
                "l2_norm": torch.norm(v).item() if v.numel() > 0 else 0,
            }
            for i, v in enumerate(v_cache_list)
        ],
        "statistics": stats,
    }

def export_to_json(
    tokens: List[str],
    k_cache_list: List[torch.Tensor],
    v_cache_list: List[torch.Tensor],
    stats: Dict[str, Any]
) -> str:
    """导出 KV Cache 数据为 JSON 字符串"""
    data = export_kvcache_to_dict(tokens, k_cache_list, v_cache_list, stats)
    return json.dumps(data, indent=2, ensure_ascii=False)

def export_to_csv(
    tokens: List[str],
    k_cache_list: List[torch.Tensor],
    v_cache_list: List[torch.Tensor],
    stats: Dict[str, Any]
) -> str:
    """导出 KV Cache 数据为 CSV 字符串"""
    # 创建 token 级别的汇总表
    rows = []
    for i, (token, k, v) in enumerate(zip(tokens, k_cache_list, v_cache_list)):
        rows.append({
            "position": i + 1,
            "token": token,
            "k_shape": str(list(k.shape)),
            "v_shape": str(list(v.shape)),
            "k_l2_norm": torch.norm(k).item() if k.numel() > 0 else 0,
            "v_l2_norm": torch.norm(v).item() if v.numel() > 0 else 0,
        })
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)

def download_file(content: str, filename: str, mime_type: str) -> bytes:
    """将内容转换为下载格式"""
    return content.encode("utf-8")
```

- [ ] **Step 2: 创建测试 tests/test_exporter.py**

```python
# tests/test_exporter.py
import pytest
import torch
from exporter import (
    export_kvcache_to_dict,
    export_to_json,
    export_to_csv,
)

def test_export_kvcache_to_dict():
    tokens = ["Hello", "world"]
    k_cache_list = [torch.randn(1, 12, 1, 64), torch.randn(1, 12, 2, 64)]
    v_cache_list = [torch.randn(1, 12, 1, 64), torch.randn(1, 12, 2, 64)]
    stats = {"num_tokens": 2}

    result = export_kvcache_to_dict(tokens, k_cache_list, v_cache_list, stats)

    assert result["tokens"] == tokens
    assert result["token_count"] == 2
    assert len(result["k_cache_summary"]) == 2
    assert result["k_cache_summary"][0]["position"] == 1

def test_export_to_json():
    tokens = ["Hello"]
    k_cache_list = [torch.randn(1, 12, 1, 64)]
    v_cache_list = [torch.randn(1, 12, 1, 64)]
    stats = {"num_tokens": 1}

    json_str = export_to_json(tokens, k_cache_list, v_cache_list, stats)

    assert isinstance(json_str, str)
    assert "Hello" in json_str
    assert '"position": 1' in json_str

def test_export_to_csv():
    tokens = ["Hello", "world"]
    k_cache_list = [torch.randn(1, 12, 1, 64), torch.randn(1, 12, 2, 64)]
    v_cache_list = [torch.randn(1, 12, 1, 64), torch.randn(1, 12, 2, 64)]
    stats = {"num_tokens": 2}

    csv_str = export_to_csv(tokens, k_cache_list, v_cache_list, stats)

    assert "position,token,k_shape" in csv_str
    assert "Hello" in csv_str
    assert "world" in csv_str
```

- [ ] **Step 3: 运行测试**

Run: `pytest tests/test_exporter.py -v`
Expected: 3 PASS

- [ ] **Step 4: 提交**

```bash
git add exporter.py tests/test_exporter.py
git commit -m "feat: add KV Cache export (JSON/CSV)"
```

---

## Task 4: 预设 Prompt 模板 (prompts.py)

**Files:**
- Create: `prompts.py`
- Test: `tests/test_prompts.py`

- [ ] **Step 1: 创建 prompts.py**

```python
# prompts.py
from typing import List, Dict, Callable, Optional
from dataclasses import dataclass

@dataclass
class PromptTemplate:
    name_zh: str
    name_en: str
    template: str
    variables: List[str]
    description_zh: str
    description_en: str

PROMPT_TEMPLATES: List[PromptTemplate] = [
    PromptTemplate(
        name_zh="问答",
        name_en="Q&A",
        template="问题：{question}\n回答：",
        variables=["question"],
        description_zh="简单的问答格式",
        description_en="Simple Q&A format"
    ),
    PromptTemplate(
        name_zh="翻译",
        name_en="Translation",
        template="请将以下内容翻译为英文：\n{text}",
        variables=["text"],
        description_zh="中译英翻译",
        description_en="Chinese to English translation"
    ),
    PromptTemplate(
        name_zh="代码补全",
        name_en="Code Completion",
        template="def {function_name}({params}):\n    \"\"\"{description}\"\"\"\n    ",
        variables=["function_name", "params", "description"],
        description_zh="Python 函数模板",
        description_en="Python function template"
    ),
    PromptTemplate(
        name_zh="故事续写",
        name_en="Story Continuation",
        template="{setup}\n\n续写：",
        variables=["setup"],
        description_zh="创意写作开头",
        description_en="Creative writing prompt"
    ),
    PromptTemplate(
        name_zh="摘要生成",
        name_en="Summarization",
        template="请为以下内容写一个简短摘要：\n\n{content}\n\n摘要：",
        variables=["content"],
        description_zh="文章摘要",
        description_en="Article summarization"
    ),
]

def get_template_names(lang: str = "zh") -> List[str]:
    """获取模板名称列表"""
    if lang == "en":
        return [t.name_en for t in PROMPT_TEMPLATES]
    return [t.name_zh for t in PROMPT_TEMPLATES]

def get_template(name: str, lang: str = "zh") -> Optional[PromptTemplate]:
    """根据名称获取模板"""
    for template in PROMPT_TEMPLATES:
        if lang == "en":
            if template.name_en == name:
                return template
        else:
            if template.name_zh == name:
                return template
    return None

def fill_template(template: PromptTemplate, **kwargs) -> str:
    """填充模板变量"""
    return template.template.format(**kwargs)

def render_template_ui(template: PromptTemplate, lang: str = "zh") -> Dict[str, str]:
    """为 UI 渲染模板字段"""
    if lang == "en":
        return {
            "name": template.name_en,
            "description": template.description_en,
            "template": template.template,
        }
    return {
        "name": template.name_zh,
        "description": template.description_zh,
        "template": template.template,
    }
```

- [ ] **Step 2: 创建测试 tests/test_prompts.py**

```python
# tests/test_prompts.py
import pytest
from prompts import (
    PROMPT_TEMPLATES,
    get_template_names,
    get_template,
    fill_template,
    render_template_ui,
)

def test_get_template_names_zh():
    names = get_template_names("zh")
    assert "问答" in names
    assert "翻译" in names

def test_get_template_names_en():
    names = get_template_names("en")
    assert "Q&A" in names
    assert "Translation" in names

def test_get_template():
    template = get_template("问答", "zh")
    assert template is not None
    assert "{question}" in template.template

def test_fill_template():
    template = get_template("问答", "zh")
    result = fill_template(template, question="What is AI?")
    assert "What is AI?" in result
    assert "问题：What is AI?" in result

def test_render_template_ui_zh():
    template = get_template("代码补全", "zh")
    ui = render_template_ui(template, "zh")
    assert ui["name"] == "代码补全"
    assert "function_name" in ui["template"]

def test_render_template_ui_en():
    template = get_template("Code Completion", "en")
    ui = render_template_ui(template, "en")
    assert ui["name"] == "Code Completion"
```

- [ ] **Step 3: 运行测试**

Run: `pytest tests/test_prompts.py -v`
Expected: 6 PASS

- [ ] **Step 4: 提交**

```bash
git add prompts.py tests/test_prompts.py
git commit -m "feat: add preset prompt templates"
```

---

## Task 5: 扩展 T5/FLAN 支持 (model_loader.py, kvcache_extractor.py)

**Files:**
- Modify: `model_loader.py`
- Modify: `kvcache_extractor.py`
- Test: `tests/test_model_loader.py`, `tests/test_kvcache_extractor.py`

- [ ] **Step 1: 修改 kvcache_extractor.py 添加架构检测**

```python
# 在 KVCacheExtractor 类中添加架构检测方法
def detect_model_architecture(model) -> str:
    """检测模型架构类型"""
    model_name_lower = model.__class__.__name__.lower()

    if "t5" in model_name_lower or "flan" in model_name_lower:
        return "encoder-decoder"  # T5, FLAN-T5, mT5
    elif "gpt" in model_name_lower or "llama" in model_name_lower or "qwen" in model_name_lower or "bloom" in model_name_lower:
        return "causal"  # GPT, LLaMA, Qwen, Bloom
    elif "opt" in model_name_lower:
        return "causal"  # OPT is causal
    else:
        return "causal"  # 默认

# 修改 register_hooks 方法，支持不同架构
def register_hooks(self, model):
    """注册 hooks 到模型"""
    architecture = self.detect_model_architecture(model)
    self.hooks = []
    self.model = model

    if architecture == "encoder-decoder":
        # T5/FLAN 架构的 hook 模式
        self._register_encoder_decoder_hooks(model)
    else:
        # CausalLM 架构的 hook 模式
        self._register_causal_hooks(model)

    return self.hooks
```

- [ ] **Step 2: 添加 _register_encoder_decoder_hooks 方法**

```python
def _register_encoder_decoder_hooks(self, model):
    """为 Encoder-Decoder 架构注册 hooks (T5/FLAN)"""
    # T5 的 attention 在 layer.block[0].layer[0].SelfAttention 或 EncDecAttention
    for name, module in model.named_modules():
        # 尝试匹配 EncDecAttention
        if "EncDecAttention" in type(module).__name__ or "qkv_proj" in name:
            handle = module.register_forward_hook(self._hook_fn)
            self.hooks.append(handle)
```

- [ ] **Step 3: 修改 model_loader.py 添加 T5 支持**

```python
# 在 HuggingFaceLoader 中添加 T5 模型识别
def get_config(self) -> Dict[str, Any]:
    """获取模型配置"""
    if self._config is not None:
        return self._config

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(self.model_name)

    # 检测架构类型
    model_type = getattr(config, 'model_type', '')

    if model_type in ['t5', 'mt5', 'flan-t5']:
        # T5 系列
        num_layers = getattr(config, 'num_layers', getattr(config, 'n_layer', 12))
        num_heads = getattr(config, 'num_heads', getattr(config, 'n_head', 12))
        hidden_size = getattr(config, 'd_model', getattr(config, 'd_model', 768))
        architecture = "encoder-decoder"
    else:
        # GPT/LLaMA 系列
        num_layers = getattr(config, 'n_layer', getattr(config, 'num_layers', 12))
        num_heads = getattr(config, 'n_head', getattr(config, 'num_heads', 12))
        hidden_size = getattr(config, 'n_embd', getattr(config, 'hidden_size', 768))
        architecture = "causal"

    self._config = {
        'num_layers': num_layers,
        'num_heads': num_heads,
        'hidden_size': hidden_size,
        'vocab_size': getattr(config, 'vocab_size', 50257),
        'max_position_embeddings': getattr(config, 'n_positions', getattr(config, 'max_position_embeddings', 1024)),
        'head_dim': hidden_size // num_heads if num_heads > 0 else 64,
        'architecture': architecture,
    }
    return self._config
```

- [ ] **Step 4: 提交**

```bash
git add model_loader.py kvcache_extractor.py
git commit -m "feat: add T5/FLAN architecture support"
```

---

## Task 6: 更新 app.py 集成所有新功能

**Files:**
- Modify: `app.py`
- Modify: `requirements.txt`

- [ ] **Step 1: 添加 pandas 依赖**

```python
# requirements.txt 添加一行
pandas>=1.5.0
```

- [ ] **Step 2: 修改 app.py 集成主题和 i18n**

```python
# app.py 顶部导入
from i18n import t, get_text, TRANSLATIONS
from theme import get_theme, get_theme_css, get_plotly_template, THEMES
from exporter import export_to_json, export_to_csv
from prompts import PROMPT_TEMPLATES, get_template_names, get_template, fill_template

# 添加主题和语言 session state
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "lang" not in st.session_state:
    st.session_state.lang = "zh"

# 侧边栏添加主题和语言选择
with st.sidebar:
    # 主题切换
    theme_options = {
        "亮色": "light",
        "Light": "light",
        "暗色": "dark",
        "Dark": "dark",
    }
    current_lang = st.session_state.lang
    theme_display = st.selectbox(
        t("theme", current_lang),
        ["亮色", "Dark"] if current_lang == "zh" else ["Light", "Dark"]
    )
    st.session_state.theme = theme_options[theme_display]

    # 语言切换
    lang_options = ["中文", "English"]
    lang_map = {"中文": "zh", "English": "en"}
    selected_lang = st.selectbox("Language", lang_options)
    st.session_state.lang = lang_map[selected_lang]

# 应用主题 CSS
st.markdown(get_theme_css(st.session_state.theme), unsafe_allow_html=True)
```

- [ ] **Step 3: 添加 Prompt 模板选择**

```python
# 在 prompt 输入上方添加模板选择
lang = st.session_state.lang

# 预设模板
st.markdown(f"**{t('prompt_templates', lang)}**")
template_names = get_template_names(lang)
selected_template = st.selectbox(
    t("select_template", lang) if "select_template" in TRANSLATIONS[lang] else "选择模板",
    ["-"] + template_names,
    label_visibility="collapsed"
)

if selected_template != "-":
    template = get_template(selected_template, lang)
    if template:
        st.info(template.description_zh if lang == "zh" else template.description_en)

prompt = st.text_input(t("input_prompt", lang), value="Hello, how are you?")
```

- [ ] **Step 4: 添加导出按钮**

```python
# 在可视化区域添加导出按钮
col_export1, col_export2 = st.columns(2)
with col_export1:
    if st.button(t("export_json", lang)):
        json_data = export_to_json(
            st.session_state.tokens,
            [h.k_cache for h in st.session_state.simulator.history],
            [h.v_cache for h in st.session_state.simulator.history],
            stats
        )
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name="kvcache_export.json",
            mime="application/json"
        )
with col_export2:
    if st.button(t("export_csv", lang)):
        csv_data = export_to_csv(
            st.session_state.tokens,
            [h.k_cache for h in st.session_state.simulator.history],
            [h.v_cache for h in st.session_state.simulator.history],
            stats
        )
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="kvcache_export.csv",
            mime="text/csv"
        )
```

- [ ] **Step 5: 更新 Plotly 图表使用主题**

```python
# 在创建图表时应用主题
template = get_plotly_template(st.session_state.theme)
fig = px.imshow(
    ...,
    template=template,  # 添加主题
    color_continuous_scale="Viridis" if st.session_state.theme == "dark" else "RdBu_r"
)
```

- [ ] **Step 6: 提交**

```bash
git add app.py requirements.txt
git commit -m "feat: integrate all optimizations (theme, i18n, templates, export)"
```

---

## Task 7: 添加批量生成和性能优化

**Files:**
- Modify: `app.py`
- Modify: `kvcache_simulator.py`

- [ ] **Step 1: 修改 run_generation_step 支持批量生成**

```python
def run_generation_streaming(prompt: str, max_new_tokens: int = 20, batch_size: int = 5):
    """流式生成，每批更新可视化"""
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    extractor = st.session_state.extractor
    simulator = st.session_state.simulator

    # 清空历史
    extractor.clear_history()
    simulator.reset()
    st.session_state.tokens = []
    st.session_state.token_ids = []

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    input_length = input_ids.shape[1]

    # 注册 hooks
    handles = extractor.register_hooks(model)

    num_batches = (max_new_tokens + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_i in range(num_batches):
            current_batch_size = min(batch_size, max_new_tokens - batch_i * batch_size)

            output = model.generate(
                input_ids,
                max_new_tokens=current_batch_size,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

            # 解码新生成的 token
            new_ids = output[0][input_length:]
            new_tokens = tokenizer.convert_ids_to_tokens(new_ids)

            # 更新 simulator
            for i, (token_id, token_str) in enumerate(zip(new_ids.tolist(), new_tokens)):
                pos = len(st.session_state.tokens) + 1
                if pos <= len(extractor.kvcache_history):
                    entry = extractor.kvcache_history[pos - 1]
                    simulator.add_entry(pos, entry.k_cache, entry.v_cache, token_id, token_str)
                else:
                    k = torch.zeros(1, simulator.num_heads, 1, simulator.head_dim)
                    v = torch.zeros(1, simulator.num_heads, 1, simulator.head_dim)
                    simulator.add_entry(pos, k, v, token_id, token_str)

            st.session_state.tokens.extend(new_tokens)
            st.session_state.token_ids.extend(new_ids.tolist())
            st.session_state.current_position = len(st.session_state.tokens)

            # 更新 input_ids 用于下次生成
            input_ids = output

            # 触发 UI 更新
            st.rerun()

    # 移除 hooks
    for handle in handles:
        handle.remove()

    st.session_state.generation_complete = True
```

- [ ] **Step 2: 添加 GPU 内存优化**

```python
# 在 run_generation_streaming 结束时添加
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 限制历史长度
MAX_HISTORY_LENGTH = 100
if len(simulator.history) > MAX_HISTORY_LENGTH:
    simulator.history = simulator.history[-MAX_HISTORY_LENGTH:]
```

- [ ] **Step 3: 提交**

```bash
git add app.py kvcache_simulator.py
git commit -m "perf: add streaming generation and GPU memory optimization"
```

---

## Task 8: 添加调试功能（层级热力图、Token详情）

**Files:**
- Modify: `visualizer.py`

- [ ] **Step 1: 添加层级能量热力图**

```python
def create_layer_energy_heatmap(
    self,
    k_cache_list: List[torch.Tensor],
    v_cache_list: List[torch.Tensor],
    title: str = "Layer Energy Heatmap"
) -> go.Figure:
    """创建层级能量热力图"""
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
        color_continuous_scale="Viridis" if st.session_state.theme == "dark" else "RdBu_r",
        template="plotly_dark" if st.session_state.theme == "dark" else "plotly"
    )
    fig.update_layout(
        xaxis_title="Token Position",
        yaxis_title="Layer",
    )
    return fig
```

- [ ] **Step 2: 添加 Token 详情悬停**

```python
def create_sequence_view_with_hover(
    self,
    tokens: List[str],
    k_cache_list: List[torch.Tensor],
    title: str = "Token Sequence with KV Cache",
    token_ids: List[int] = None,
) -> go.Figure:
    """创建带悬停详情的序列视图"""
    energies = []
    for k in k_cache_list:
        if k is not None and k.numel() > 0:
            energies.append(torch.norm(k).item())
        else:
            energies.append(0)

    fig = go.Figure(data=[
        go.Bar(
            x=list(range(1, len(tokens) + 1)),
            y=energies,
            text=tokens,
            hovertemplate=(
                "<b>Token %{text}</b><br>"
                "Position: %{x}<br>"
                "Energy: %{y:.4f}<br>"
                "<extra></extra>"
            ),
            marker_color=energies,
            marker_colorscale="Viridis" if st.session_state.theme == "dark" else "RdBu_r",
        )
    ])

    fig.update_layout(
        title=title,
        xaxis_title="Token Position",
        yaxis_title="KV Cache Energy (L2 Norm)",
        template="plotly_dark" if st.session_state.theme == "dark" else "plotly"
    )
    return fig
```

- [ ] **Step 3: 在 app.py 中添加新的可视化 Tab**

```python
# 在现有的 tabs 中添加新 Tab
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    t("matrix_heatmap", lang),
    t("sequence_view", lang),
    t("layer_distribution", lang),
    t("statistics", lang),
    t("dashboard", lang),
    t("layer_energy_heatmap", lang)  # 新增
])

# ...

with tab6:
    if st.session_state.current_position > 0:
        fig = st.session_state.visualizer.create_layer_energy_heatmap(
            k_cache_list[:st.session_state.current_position],
            v_cache_list[:st.session_state.current_position],
            title=t("layer_energy_heatmap", lang)
        )
        st.plotly_chart(fig, use_container_width=True)
```

- [ ] **Step 4: 提交**

```bash
git add visualizer.py app.py
git commit -m "feat: add layer energy heatmap and token hover details"
```

---

## Task 9: 最终集成测试

- [ ] **Step 1: 运行所有测试**

Run: `pytest tests/ -v`
Expected: 所有测试通过

- [ ] **Step 2: 手动测试应用**

```bash
uv run streamlit run app.py
```

验证：
- [ ] 主题切换正常
- [ ] 语言切换正常
- [ ] Prompt 模板可用
- [ ] 模型加载正常（GPT-2）
- [ ] 生成功能正常
- [ ] 可视化显示正常
- [ ] 导出功能正常
- [ ] 层级热力图显示正常

- [ ] **Step 3: 提交所有更改**

```bash
git add -A
git commit -m "feat: complete optimization - multi-arch, export, i18n, theme, templates, perf, debug"
```

---

## 验收标准检查清单

- [ ] 支持 T5/FLAN 模型加载
- [ ] KV Cache 数据可导出为 JSON 和 CSV
- [ ] 支持中英双语切换
- [ ] 支持暗色/亮色主题切换
- [ ] 预设 Prompt 模板可用
- [ ] 批量生成功能正常
- [ ] 可视化支持增量更新
- [ ] 层级能量热力图正常显示
- [ ] Token 详情悬停显示
- [ ] GPU 内存优化生效

---

## 执行选项

**Plan complete and saved to `docs/superpowers/plans/2026-04-01-kvcache-viz-optimization-plan.md`. Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
