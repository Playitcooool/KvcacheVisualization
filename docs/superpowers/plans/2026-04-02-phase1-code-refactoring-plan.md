# Phase 1: Code Refactoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create modular directory structure, unify logging system, and simplify app.py to under 500 lines

**Architecture:** Phase 1 focuses on code quality without changing functionality. We create the directory structure, extract utility modules, and prepare for Phase 2 (visualization) and Phase 3 (multi-model).

**Tech Stack:** Python, Streamlit, logging module, Pathlib

---

## File Structure

```
Phase 1 creates:
├── utils/
│   ├── __init__.py           # Exports logger setup
│   └── logger.py              # Unified logging configuration (NEW)
├── core/
│   ├── __init__.py            # Exports core classes
│   └── model_base.py          # ModelHandle dataclass (NEW)
└── (existing files refactored in-place)

Phase 1 modifies (no new files yet):
├── device_utils.py            # Add logger, cleanup
├── model_loader.py            # Add logger, keep structure
├── kvcache_extractor.py       # Add logger, cleanup
├── kvcache_simulator.py       # Add logger, keep structure
├── visualizer.py              # Add logger
├── app.py                     # Import from utils.logger, remove debug prints
└── i18n.py                    # Keep as-is for now
└── exporter.py                # Keep as-is for now
└── theme.py                   # Keep as-is for now
```

---

## Task 1: Create utils/logger.py

**Files:**
- Create: `utils/__init__.py`
- Create: `utils/logger.py`
- Test: `tests/test_logger.py` (basic import test)

- [ ] **Step 1: Create utils directory and __init__.py**

Create file: `utils/__init__.py`
```python
"""Utility modules for KV Cache Visualizer."""

from utils.logger import setup_logger, get_logger

__all__ = ["setup_logger", "get_logger"]
```

- [ ] **Step 2: Create utils/logger.py**

Create file: `utils/logger.py`
```python
"""Unified logging configuration for KV Cache Visualizer."""

import logging
import sys
from typing import Optional


# Global logger registry
_loggers = {}


def setup_logger(
    name: str,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with consistent formatting.

    Args:
        name: Logger name (typically __name__)
        level: Logging level (default INFO)
        format_string: Optional custom format string

    Returns:
        Configured logger instance
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Default format
    if format_string is None:
        format_string = "[%(asctime)s] [%(name)s] %(levelname)s: %(message)s"

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)

    _loggers[name] = logger
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a new one with default settings.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    if name not in _loggers:
        return setup_logger(name)
    return _loggers[name]


# Module-level logger for this file
logger = setup_logger(__name__)
```

- [ ] **Step 3: Test logger module**

Run: `python -c "from utils.logger import setup_logger, get_logger; l = setup_logger('test'); l.info('test message'); print('OK')"`
Expected: Prints log line with timestamp

- [ ] **Step 4: Commit**

```bash
git add utils/__init__.py utils/logger.py
git commit -m "feat: add unified logging system in utils/logger.py"
```

---

## Task 2: Create core/model_base.py with ModelHandle dataclass

**Files:**
- Create: `core/__init__.py`
- Create: `core/model_base.py`
- Modify: `model_loader.py` (import ModelHandle from core)

- [ ] **Step 1: Create core/__init__.py**

Create file: `core/__init__.py`
```python
"""Core modules for KV Cache Visualizer."""

from core.model_base import ModelHandle

__all__ = ["ModelHandle"]
```

- [ ] **Step 2: Create core/model_base.py**

Create file: `core/model_base.py`
```python
"""Model handle data class for multi-model management."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from kvcache_extractor import KVCacheExtractor
from kvcache_simulator import KVCacheSimulator
from visualizer import KVCacheVisualizer


@dataclass
class ModelHandle:
    """
    A handle containing all objects associated with a single loaded model.

    This dataclass groups together the model, tokenizer, extractor,
    simulator, and visualizer for easy management and access.
    """

    id: str
    name: str
    model: Any
    tokenizer: Any
    config: Dict[str, Any]
    extractor: KVCacheExtractor
    simulator: KVCacheSimulator
    visualizer: KVCacheVisualizer
    device: str = "cpu"

    @property
    def num_layers(self) -> int:
        """Get number of layers from config."""
        return self.config.get("num_layers", 12)

    @property
    def num_heads(self) -> int:
        """Get number of attention heads from config."""
        return self.config.get("num_heads", 12)

    @property
    def head_dim(self) -> int:
        """Get head dimension from config."""
        return self.config.get("head_dim", 64)

    @property
    def num_kv_heads(self) -> int:
        """Get number of KV heads (for GQA) from config."""
        return self.config.get("num_kv_heads", self.num_heads)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary dict of the model configuration."""
        return {
            "id": self.id,
            "name": self.name,
            "device": self.device,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "hidden_size": self.config.get("hidden_size", "N/A"),
        }
```

- [ ] **Step 3: Update model_loader.py to use ModelHandle**

Read first: `model_loader.py:1-20`

Add import at top of model_loader.py:
```python
from core.model_base import ModelHandle
```

- [ ] **Step 4: Commit**

```bash
git add core/__init__.py core/model_base.py model_loader.py
git commit -m "refactor: add ModelHandle dataclass in core/model_base.py"
```

---

## Task 2.5: Create core/model_manager.py (basic version)

**Files:**
- Create: `core/model_manager.py`
- Modify: `app.py` (use ModelManager)

**Note:** This creates a basic ModelManager that manages a single model. Full multi-model support will be added in Phase 3.

- [ ] **Step 1: Create core/model_manager.py**

Create file: `core/model_manager.py`
```python
"""Basic model manager for single model support (Phase 1)."""

import torch
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from core.model_base import ModelHandle
from model_loader import ModelLoader
from kvcache_extractor import KVCacheExtractor
from kvcache_simulator import KVCacheSimulator
from visualizer import KVCacheVisualizer
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a model to be loaded."""
    model_type: str  # "huggingface" or "pytorch"
    model_name: str = "gpt2"
    checkpoint_path: str = ""
    device: str = "auto"
    quantization: str = None


class ModelManager:
    """
    Manages model lifecycle (Phase 1: single model, Phase 3: multi-model).

    This class handles loading, unloading, and accessing models.
    """

    def __init__(self):
        self._active: Optional[ModelHandle] = None
        self._loader: Optional[ModelLoader] = None

    @property
    def active_model(self) -> Optional[ModelHandle]:
        """Get the currently active model handle."""
        return self._active

    @property
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._active is not None

    def load(self, config: ModelConfig) -> Tuple[bool, str]:
        """
        Load a model according to the given configuration.

        Args:
            config: ModelConfig with model type and parameters

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            logger.info(f"Loading model: {config.model_name}")

            # Create appropriate loader
            if config.model_type == "huggingface":
                self._loader = ModelLoader.create(
                    "huggingface",
                    model_name=config.model_name,
                    quantization=config.quantization
                )
            else:
                self._loader = ModelLoader.create(
                    "pytorch",
                    checkpoint_path=config.checkpoint_path
                )

            # Load model
            model, tokenizer, model_config = self._loader.load(device=config.device)

            # Create model handle
            num_layers = model_config.get('num_layers', 12)
            num_heads = model_config.get('num_heads', 12)
            num_kv_heads = model_config.get('num_kv_heads', num_heads)
            head_dim = model_config.get('head_dim', 64)

            extractor = KVCacheExtractor(
                num_layers=num_layers,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim
            )

            simulator = KVCacheSimulator(
                num_layers=num_layers,
                num_heads=num_heads,
                head_dim=head_dim
            )

            visualizer = KVCacheVisualizer(
                num_layers=num_layers,
                num_heads=num_heads,
                head_dim=head_dim
            )

            device_str = str(self._loader.current_device) if hasattr(self._loader, 'current_device') else config.device

            self._active = ModelHandle(
                id="model_a",
                name=config.model_name,
                model=model,
                tokenizer=tokenizer,
                config=model_config,
                extractor=extractor,
                simulator=simulator,
                visualizer=visualizer,
                device=device_str
            )

            logger.info(f"Model loaded successfully: {num_layers} layers, {num_heads} heads")
            return True, f"模型加载成功 ({num_layers} 层, {num_heads} 头)"

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False, f"加载失败: {str(e)}"

    def unload(self):
        """Unload the current model and free resources."""
        if self._active is not None:
            logger.info(f"Unloading model: {self._active.name}")
            self._active = None
            self._loader = None

            # GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_active_handle(self) -> Optional[ModelHandle]:
        """Get the active model handle (alias for active_model property)."""
        return self._active
```

- [ ] **Step 2: Update app.py to use ModelManager**

Replace the direct model loading code in app.py with:
```python
from core.model_manager import ModelManager, ModelConfig

# In init_session_state, add:
if 'model_manager' not in st.session_state:
    st.session_state.model_manager = ModelManager()

# Replace load_model function body with:
def load_model(model_type: str, model_name: str = "gpt2", checkpoint_path: str = "", device: str = "auto"):
    config = ModelConfig(
        model_type=model_type,
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        device=device
    )
    success, msg = st.session_state.model_manager.load(config)
    if success:
        st.session_state.model_loaded = True
        st.session_state.model_config = st.session_state.model_manager.active_model.config
    return success, msg
```

- [ ] **Step 3: Commit**

```bash
git add core/model_manager.py app.py
git commit -m "feat: add basic ModelManager class in core/model_manager.py"
```

---

## Task 3: Add logging to existing modules

**Files:**
- Modify: `device_utils.py`
- Modify: `model_loader.py`
- Modify: `kvcache_extractor.py`
- Modify: `kvcache_simulator.py`
- Modify: `visualizer.py`

- [ ] **Step 1: Add logging to device_utils.py**

Read first: `device_utils.py:1-30`

Add import at top:
```python
from utils.logger import setup_logger
logger = setup_logger(__name__)
```

Replace any print statements with logger.info() or logger.debug().

- [ ] **Step 2: Add logging to model_loader.py**

Read first: `model_loader.py:1-15`

Add import (if not already):
```python
from utils.logger import setup_logger
logger = setup_logger(__name__)
```

Replace print statements in the file with appropriate logger calls.

- [ ] **Step 3: Add logging to kvcache_extractor.py**

Read first: `kvcache_extractor.py:1-20`

Add import:
```python
from utils.logger import setup_logger
logger = setup_logger(__name__)
```

Replace any print statements with logger calls. The debug property can stay but should use logger.debug() instead.

- [ ] **Step 4: Add logging to kvcache_simulator.py**

Read first: `kvcache_simulator.py:1-20`

Add import:
```python
from utils.logger import setup_logger
logger = setup_logger(__name__)
```

Replace any print statements.

- [ ] **Step 5: Add logging to visualizer.py**

Read first: `visualizer.py:1-15`

Add import:
```python
from utils.logger import setup_logger
logger = setup_logger(__name__)
```

- [ ] **Step 6: Commit**

```bash
git add device_utils.py model_loader.py kvcache_extractor.py kvcache_simulator.py visualizer.py
git commit -m "refactor: add unified logging to core modules"
```

---

## Task 4: Clean up app.py - Remove debug prints

**Files:**
- Modify: `app.py` (remove debug prints, add logger)

- [ ] **Step 1: Read app.py and identify all debug prints**

Run: `grep -n "print\[" app.py`
Run: `grep -n "print(" app.py | head -30`

- [ ] **Step 2: Remove all print("[DEBUG]" statements**

For each debug print found:
1. Replace with logger.debug() call
2. Or remove entirely if redundant

Specific patterns to remove:
- `print(f"[DEBUG] ...")` statements throughout the file
- Keep important print statements (e.g., user-facing messages)

- [ ] **Step 3: Add logger import at top of app.py**

Add after existing imports:
```python
from utils.logger import setup_logger
logger = setup_logger(__name__)
```

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "refactor: remove debug prints from app.py, use logging"
```

---

## Task 5: Simplify app.py - Extract sidebar and components

**Files:**
- Create: `ui/__init__.py`
- Create: `ui/sidebar.py`
- Create: `ui/components.py`
- Create: `ui/layout.py`
- Modify: `app.py`

**Goal:** Reduce app.py from ~790 lines to under 500 lines

- [ ] **Step 1: Create ui/__init__.py**

Create file: `ui/__init__.py`
```python
"""UI components for KV Cache Visualizer."""

from ui.sidebar import render_sidebar
from ui.components import render_generation_controls, render_debug_panel
from ui.layout import create_two_column_layout

__all__ = [
    "render_sidebar",
    "render_generation_controls",
    "render_debug_panel",
    "create_two_column_layout",
]
```

- [ ] **Step 2: Extract sidebar to ui/sidebar.py**

Create file: `ui/sidebar.py`
```python
"""Sidebar component for model selection and settings."""

import streamlit as st
from typing import Tuple, Optional, Dict, Any

from utils.logger import get_logger
logger = get_logger(__name__)


def render_sidebar(
    model_source: str,
    model_name: str,
    local_path: str,
    selected_device: str,
    theme: str,
    lang: str,
    model_loaded: bool,
    model_config: Dict[str, Any],
) -> Tuple[str, str, str, str, str]:
    """
    Render the sidebar with model selection and settings.

    Returns:
        Tuple of (model_source, model_name, local_path, selected_device, theme)
    """
    st.markdown("### ⚙️ 模型设置")

    # Device selection
    from device_utils import list_available_devices
    available_devices = list_available_devices()
    device_options = ["auto"] + available_devices
    selected_device = st.selectbox(
        "计算设备",
        device_options,
        format_func=lambda x: f"{x} {'(自动)' if x == 'auto' else ''}"
    )

    # Theme selection
    from i18n import t
    theme_display = st.selectbox(
        t("theme", st.session_state.lang),
        ["🌞 亮色", "🌙 暗色"] if st.session_state.lang == "zh" else ["Light", "Dark"]
    )
    theme = "dark" if "暗" in theme_display or "Dark" in theme_display else "light"

    # Language selection
    lang_display = st.selectbox(
        "Language",
        ["中文", "English"]
    )
    lang = "en" if lang_display == "English" else "zh"

    st.markdown("---")

    # Model source selection
    model_source = st.radio(
        "模型来源",
        ["🤖 HuggingFace", "📁 本地模型"],
        help="选择 HuggingFace 在线模型或本地模型"
    )

    model_name = ""
    local_path = ""

    if model_source == "🤖 HuggingFace":
        model_name = _render_huggingface_selector()
    else:
        local_path = _render_local_model_selector()

    st.markdown("---")

    # Cache status
    _render_cache_status(model_source, model_name)

    return model_source, model_name, local_path, selected_device, theme, lang


def _render_huggingface_selector() -> str:
    """Render HuggingFace model selector."""
    st.markdown("**选择模型:**")
    preset_models = {
        "gpt2 (默认, 小)": "gpt2",
        "gpt2-medium (中)": "gpt2-medium",
        "TinyLlama (小)": "TinyLlama/TinyLlama-1.1B-v0.1",
        "Qwen2-0.5B (小)": "Qwen/Qwen2-0.5B",
        "Qwen2-1.5B (中)": "Qwen/Qwen2-1.5B",
    }
    selected_preset = st.selectbox("预设模型", list(preset_models.keys()), label_visibility="collapsed")
    model_name = preset_models[selected_preset]
    st.caption(f"模型名: `{model_name}`")
    return model_name


def _render_local_model_selector() -> str:
    """Render local model selector."""
    import os
    st.markdown("**本地模型路径:**")
    local_path = st.text_input(
        "模型路径",
        placeholder="/path/to/model",
        label_visibility="collapsed",
        help="本地模型目录路径 (包含 config.json) 或 checkpoint 文件 (.pt/.pth/.safetensors)"
    )

    if local_path:
        if os.path.isdir(local_path):
            st.success("📁 检测到模型目录")
        elif os.path.isfile(local_path):
            st.success("📄 检测到 checkpoint 文件")
        else:
            st.error("❌ 路径不存在")
            local_path = ""

    return local_path


def _render_cache_status(model_source: str, model_name: str):
    """Render model cache status."""
    import os
    st.markdown("**📦 模型缓存状态:**")
    cache_status = st.empty()

    if model_source == "🤖 HuggingFace" and model_name:
        model_cache_path = os.path.join("models", model_name.replace("/", "--"))
        hf_cache_path = os.path.expanduser(f"~/.cache/huggingface/hub/models--{model_name.replace('/', '--')}")

        if os.path.exists(model_cache_path):
            cache_status.info(f"✅ 已缓存在本地: `models/`")
        elif os.path.exists(hf_cache_path):
            cache_status.info(f"✅ 已缓存在 HF 默认目录")
        else:
            cache_status.warning(f"⬇️ 需要下载 ({model_name})")
```

- [ ] **Step 3: Extract components to ui/components.py**

Create file: `ui/components.py`
```python
"""Reusable UI components."""

import streamlit as st
from typing import List

from utils.logger import get_logger
logger = get_logger(__name__)


def render_generation_controls(
    prompt: str,
    batch_size: int,
    streaming_mode: bool,
    stream_batch_size: int,
    generation_complete: bool,
    is_generating: bool,
    streaming_pending: bool,
    tokens: List[str],
    current_position: int,
) -> tuple:
    """
    Render generation controls and return state.

    Returns:
        Tuple of (prompt, batch_size, streaming_mode, stream_batch_size, button_clicked)
    """
    col_btn1, col_btn2 = st.columns(2)

    status_text = "✓ 生成完成" if generation_complete else (
        "⏳ 生成中..." if is_generating else "▶ 开始生成" if not streaming_pending else "▶ 继续生成"
    )

    button_label = status_text

    with col_btn1:
        clicked = st.button(button_label, type="primary")

    with col_btn2:
        reset_clicked = st.button("🔄 重新生成")

    return prompt, batch_size, streaming_mode, stream_batch_size, clicked, reset_clicked


def render_debug_panel(extractor, simulator, model_loaded: bool):
    """Render debug information panel."""
    if model_loaded and extractor:
        with st.expander("🔧 调试信息"):
            history_len = len(extractor.kvcache_history)
            st.text(f"捕获的 KV Cache 条目数: {history_len}")
            st.text(f"Token 数量: {len(tokens) if 'tokens' in dir() else 0}")
            if simulator:
                st.text(f"Simulator 历史长度: {len(simulator.history)}")
```

- [ ] **Step 4: Create ui/layout.py**

Create file: `ui/layout.py`
```python
"""Layout components."""

import streamlit as st


def create_two_column_layout(left_ratio: int = 1, right_ratio: int = 1) -> tuple:
    """
    Create a two-column layout.

    Args:
        left_ratio: Relative width of left column
        right_ratio: Relative width of right column

    Returns:
        Tuple of (left_column, right_column)
    """
    return st.columns([left_ratio, right_ratio])


def create_three_column_layout(ratios: tuple = (1, 1, 1)) -> tuple:
    """Create a three-column layout."""
    return st.columns(ratios)
```

- [ ] **Step 5: Refactor app.py to use extracted modules**

This is the main refactoring task. Read app.py and:
1. Import from new modules
2. Call extracted functions
3. Remove duplicate code
4. Target: reduce from ~790 lines to <500 lines

**Note:** This step requires careful reading of app.py to understand which functions to extract. The goal is to:
- Move sidebar rendering to ui.sidebar.render_sidebar()
- Move generation controls to ui.components
- Keep main layout logic in app.py

- [ ] **Step 6: Commit**

```bash
git add ui/ app.py
git commit -m "refactor: extract UI components to ui/ module, simplify app.py"
```

---

## Task 6: Verify and test

**Files:**
- Test: `streamlit run app.py` (manual test)
- Run: `python -c "from utils.logger import setup_logger; print('OK')"`

- [ ] **Step 1: Verify all imports work**

Run: `python -c "from utils.logger import setup_logger; from core.model_base import ModelHandle; print('All imports OK')"`

- [ ] **Step 2: Count lines in app.py**

Run: `wc -l app.py`
Target: <500 lines

- [ ] **Step 3: Check for remaining debug prints**

Run: `grep -n "print\[" app.py || echo "No debug prints found"`
Run: `grep -n "print('" app.py | grep -v "OK'" || echo "No debug prints found"`

- [ ] **Step 4: Run basic pytest**

Run: `pytest tests/ -v --tb=short 2>&1 | head -50`

- [ ] **Step 5: Final commit for Phase 1**

```bash
git add -A
git commit -m "Phase 1: Complete code refactoring - modular structure, unified logging"
```

---

## Self-Review Checklist

After completing all tasks, verify:

1. **Spec coverage:**
   - [x] Unified logging system created (utils/logger.py)
   - [x] ModelHandle dataclass created (core/model_base.py)
   - [x] app.py reduced to under 500 lines
   - [x] All debug prints removed from app.py
   - [x] UI components extracted to ui/ module

2. **Placeholder scan:**
   - No "TBD" or "TODO" in implementation
   - All functions have complete code
   - No vague descriptions

3. **Type consistency:**
   - ModelHandle dataclass fields match usage in app.py
   - Logger function signatures consistent

4. **Line count:**
   - app.py should be under 500 lines after refactoring
