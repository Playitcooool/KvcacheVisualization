"""Sidebar UI components."""

import streamlit as st
import os

from device_utils import list_available_devices
from i18n import t


def render_sidebar():
    """Render the sidebar with model settings.

    Returns:
        tuple: (model_source, model_name, checkpoint_path, selected_device)
    """
    st.markdown("### ⚙️ 模型设置")

    # 设备选择
    available_devices = list_available_devices()
    device_options = ["auto"] + available_devices
    selected_device = st.selectbox(
        "计算设备",
        device_options,
        format_func=lambda x: f"{x} {'(自动)' if x == 'auto' else ''}"
    )

    # 主题选择
    theme_display = st.selectbox(
        t("theme", st.session_state.lang),
        ["🌞 亮色", "🌙 暗色"] if st.session_state.lang == "zh" else ["Light", "Dark"]
    )
    st.session_state.theme = "dark" if "暗" in theme_display or "Dark" in theme_display else "light"

    # 语言选择
    lang_display = st.selectbox(
        "Language",
        ["中文", "English"]
    )
    st.session_state.lang = "en" if lang_display == "English" else "zh"

    st.markdown("---")

    # 模型来源选择
    model_source = st.radio(
        "模型来源",
        ["🤖 HuggingFace", "📁 本地模型"],
        help="选择 HuggingFace 在线模型或本地模型"
    )

    quantization = None
    model_name = ""
    checkpoint_path = ""
    local_path = ""

    if model_source == "🤖 HuggingFace":
        # HuggingFace 模型选择
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
    else:
        # 本地模型
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
                model_name = local_path
                checkpoint_path = ""
            elif os.path.isfile(local_path):
                st.success("📄 检测到 checkpoint 文件")
                model_name = "custom"
                checkpoint_path = local_path
            else:
                st.error("❌ 路径不存在")
                model_name = ""
                checkpoint_path = ""
        else:
            model_name = "custom"
            checkpoint_path = ""

    st.markdown("---")

    # 模型缓存状态
    st.markdown("**📦 模型缓存状态:**")
    cache_status = st.empty()
    if model_source == "🤖 HuggingFace":
        # 检查本地缓存
        model_cache_path = os.path.join("models", model_name.replace("/", "--"))
        hf_cache_path = os.path.expanduser(f"~/.cache/huggingface/hub/models--{model_name.replace('/', '--')}")

        if os.path.exists(model_cache_path):
            cache_status.info(f"✅ 已缓存在本地: `models/`")
        elif os.path.exists(hf_cache_path):
            cache_status.info(f"✅ 已缓存在 HF 默认目录")
        else:
            cache_status.warning(f"⬇️ 需要下载 ({model_name})")

    # 加载按钮
    can_load = (
        st.session_state.get('model_loaded', False) or
        (model_source == "🤖 HuggingFace" and model_name) or
        (model_source == "📁 本地模型" and os.path.exists(local_path) if local_path else False)
    )

    # Load model button - returns values needed by main app
    if st.button("🚀 加载模型", type="primary", disabled=not can_load):
        return model_source, model_name, checkpoint_path, selected_device, local_path, True

    # 模型信息
    if st.session_state.model_loaded:
        st.markdown("---")
        st.markdown("**模型信息:**")
        config = st.session_state.model_config
        st.markdown(f"- 📊 层数: `{config.get('num_layers', 'N/A')}`")
        st.markdown(f"- 🔢 头数: `{config.get('num_heads', 'N/A')}`")
        st.markdown(f"- 📐 隐藏维度: `{config.get('hidden_size', 'N/A')}`")
        if config.get('quantization'):
            st.markdown(f"- 🔢 量化: `{config.get('quantization')}`")
        st.markdown(f"- 💻 设备: `{st.session_state.model_loader.current_device if st.session_state.model_loader else 'N/A'}`")

    st.markdown("---")
    st.markdown("**使用说明:**")
    st.markdown("""
    1. 选择设备并加载模型
    2. 输入 Prompt 并生成
    3. 查看 KV Cache 可视化
    4. 拖动进度条回放
    """)

    return model_source, model_name, checkpoint_path, selected_device, local_path, False
