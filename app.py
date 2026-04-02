# app.py
import streamlit as st
import torch
from typing import Optional, Tuple
import os
import signal
import sys

from model_loader import ModelLoader, HuggingFaceLoader
from kvcache_extractor import KVCacheExtractor
from kvcache_simulator import KVCacheSimulator
from visualizer import KVCacheVisualizer
from device_utils import get_available_device, list_available_devices
from i18n import t, get_text, TRANSLATIONS
from theme import get_theme, get_theme_css, get_plotly_template, THEMES
from exporter import export_to_json, export_to_csv
from prompts import PROMPT_TEMPLATES, get_template_names, get_template, fill_template
from utils.logger import setup_logger
from ui import (
    render_sidebar,
    render_generation_controls,
    render_debug_panel,
    render_template_selector,
    render_generation_result,
    render_replay_control,
    render_export_buttons,
    render_visualization_tabs,
    create_two_column_layout,
)

logger = setup_logger(__name__)

# 页面配置
st.set_page_config(
    page_title="KV Cache 可视化器",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化主题和语言
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "lang" not in st.session_state:
    st.session_state.lang = "zh"

st.markdown(get_theme_css(st.session_state.theme), unsafe_allow_html=True)

st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: 600; color: #1f77b4; margin-bottom: 1rem; }
    .token-display { font-family: 'Courier New', monospace; background-color: #f0f0f0; padding: 1rem; border-radius: 8px; font-size: 1.1rem; min-height: 60px; }
    .stButton>button { width: 100%; height: 3rem; font-size: 1.1rem; }
    .model-card { background-color: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """初始化 session state"""
    defaults = {
        'model_loader': None, 'model': None, 'tokenizer': None,
        'extractor': None, 'simulator': None, 'visualizer': None,
        'tokens': [], 'token_ids': [], 'current_position': 0,
        'generation_complete': False, 'is_generating': False,
        'model_loaded': False, 'model_config': {},
        'streaming_pending': False, 'streaming_prompt': '',
        'streaming_max_tokens': 50, 'streaming_batch_size': 5,
        'generated_text': '', 'generation_input_ids': None,
        'attention_mask': None, 'generation_thread': None,
        'max_history_length': 100,  # KV Cache 缓存大小
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clean_bpe_token(token: str) -> str:
    """将 BPE token 转换为可读文本"""
    return token.replace("Ġ", " ").replace("Ċ", "\n")


def load_model(model_type: str, model_name: str = "gpt2", checkpoint_path: str = "", device: str = "auto", quantization: str = None):
    """加载模型"""
    try:
        if model_type == "huggingface":
            loader = ModelLoader.create("huggingface", model_name=model_name, quantization=quantization)
            with st.spinner("正在加载模型，如果是首次下载可能需要几分钟..."):
                model, tokenizer, config = loader.load(device=device)
        else:
            loader = ModelLoader.create("pytorch", checkpoint_path=checkpoint_path)
            with st.spinner("正在加载模型..."):
                model, tokenizer, config = loader.load(device=device)

        st.session_state.model_loader = loader
        st.session_state.model = model
        st.session_state.tokenizer = tokenizer
        st.session_state.model_config = config
        st.session_state.model_loaded = True

        num_layers = config.get('num_layers', 12)
        num_heads = config.get('num_heads', 12)
        num_kv_heads = config.get('num_kv_heads', num_heads)
        head_dim = config.get('head_dim', 64)

        st.session_state.extractor = KVCacheExtractor(
            num_layers=num_layers, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim)
        st.session_state.simulator = KVCacheSimulator(
            num_layers=num_layers, num_heads=num_heads, head_dim=head_dim,
            max_history_length=st.session_state.get('max_history_length', 100))
        st.session_state.visualizer = KVCacheVisualizer(
            num_layers=num_layers, num_heads=num_heads, head_dim=head_dim)

        return True, f"模型加载成功 ({config.get('num_layers', '?')} 层, {config.get('num_heads', '?')} 头)"
    except Exception as e:
        return False, f"加载失败: {str(e)}"


def run_generation_step(prompt: str, max_new_tokens: int = 50):
    """执行一步生成"""
    if st.session_state.generation_complete or st.session_state.is_generating:
        return

    st.session_state.is_generating = True
    try:
        model = st.session_state.model
        tokenizer = st.session_state.tokenizer
        extractor = st.session_state.extractor
        simulator = st.session_state.simulator

        extractor.clear_history()
        simulator.reset()
        st.session_state.tokens = []
        st.session_state.token_ids = []
        st.session_state.current_position = 0

        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
        input_length = input_ids.shape[1]
        st.session_state.prompt_length = input_length
        attention_mask = torch.ones_like(input_ids)

        handles = extractor.register_hooks(model)

        with torch.no_grad():
            output = model.generate(
                input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=0.7,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id, use_cache=False)

        for handle in handles:
            handle.remove()

        generated_ids = output[0][input_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids)

        for i, (token_id, token_str) in enumerate(zip(generated_ids.tolist(), generated_tokens)):
            pos = i + 1
            if pos <= len(extractor.kvcache_history):
                entry = extractor.kvcache_history[pos - 1]
                simulator.add_entry(pos, entry.k_cache, entry.v_cache, token_id, token_str)
            else:
                k = torch.zeros(1, simulator.num_heads, 1, simulator.head_dim)
                v = torch.zeros(1, simulator.num_heads, 1, simulator.head_dim)
                simulator.add_entry(pos, k, v, token_id, token_str)

        st.session_state.tokens = generated_tokens
        st.session_state.generated_text = generated_text
        st.session_state.token_ids = generated_ids.tolist()
        st.session_state.current_position = len(generated_tokens)
        st.session_state.generation_complete = True

    except Exception as e:
        import traceback
        st.error(f"生成失败: {str(e)}")
        with st.expander("详细错误信息"):
            st.text(traceback.format_exc())
    finally:
        st.session_state.is_generating = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_generation_streaming(prompt: str, max_new_tokens: int = 50, batch_size: int = 5):
    """流式生成 token"""
    logger.debug("========== run_generation_streaming START ==========")
    logger.debug(f"is_generating={st.session_state.is_generating}, gen_complete={st.session_state.generation_complete}")

    if st.session_state.generation_complete:
        return
    if st.session_state.is_generating:
        return

    st.session_state.is_generating = True
    try:
        model = st.session_state.model
        tokenizer = st.session_state.tokenizer
        extractor = st.session_state.extractor
        simulator = st.session_state.simulator

        if (not st.session_state.tokens or st.session_state.streaming_prompt != prompt):
            extractor.clear_history()
            simulator.reset()
            st.session_state.tokens = []
            st.session_state.token_ids = []
            st.session_state.current_position = 0

            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
            attention_mask = torch.ones_like(input_ids)
            st.session_state.prompt_length = input_ids.shape[1]
            st.session_state.generation_input_ids = input_ids
            st.session_state.attention_mask = attention_mask
            st.session_state.streaming_prompt = prompt
        else:
            input_ids = st.session_state.generation_input_ids
            attention_mask = torch.ones_like(input_ids)

        if input_ids is None:
            st.error("输入状态异常，请刷新页面重试")
            st.session_state.is_generating = False
            return

        remaining = max_new_tokens - st.session_state.current_position
        if remaining <= 0:
            st.session_state.generation_complete = True
            return

        current_batch_size = min(batch_size, remaining)

        handles = extractor.register_hooks(model)
        logger.debug(f"Detected attention type: {extractor._attention_type}")

        with torch.no_grad():
            output = model.generate(
                input_ids, attention_mask=attention_mask, max_new_tokens=current_batch_size,
                do_sample=True, temperature=0.7,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id, use_cache=False)

        for handle in handles:
            handle.remove()

        start_pos = st.session_state.prompt_length + st.session_state.current_position
        if start_pos >= len(output[0]):
            st.warning(f"警告: start_pos={start_pos} 超出输出长度={len(output[0])}")
            st.session_state.generation_complete = True
            st.session_state.streaming_pending = False
            st.session_state.is_generating = False
            return

        new_ids = output[0][start_pos:start_pos + current_batch_size]
        new_tokens = tokenizer.convert_ids_to_tokens(new_ids)
        extractor_start_idx = st.session_state.prompt_length + st.session_state.current_position

        for i, (token_id, token_str) in enumerate(zip(new_ids.tolist(), new_tokens)):
            pos = st.session_state.current_position + i + 1
            hist_idx = extractor_start_idx + i

            if len(simulator.history) >= st.session_state.get('max_history_length', 100):
                st.session_state.generation_complete = True
                st.session_state.streaming_pending = False
                st.session_state.is_generating = False
                break

            if hist_idx < len(extractor.kvcache_history):
                entry = extractor.kvcache_history[hist_idx]
                simulator.add_entry(pos, entry.k_cache, entry.v_cache, token_id, token_str)
            else:
                k = torch.zeros(1, simulator.num_heads, 1, simulator.head_dim)
                v = torch.zeros(1, simulator.num_heads, 1, simulator.head_dim)
                simulator.add_entry(pos, k, v, token_id, token_str)

        st.session_state.token_ids.extend(new_ids.tolist())
        st.session_state.tokens.extend(new_tokens)
        st.session_state.current_position += len(new_tokens)
        st.session_state.generation_input_ids = output
        st.session_state.generated_text = tokenizer.decode(st.session_state.token_ids, skip_special_tokens=True)

        if st.session_state.current_position >= max_new_tokens:
            st.session_state.generation_complete = True
            st.session_state.streaming_pending = False
        else:
            st.session_state.is_generating = False

    except Exception as e:
        import traceback
        logger.debug(f"EXCEPTION: {str(e)}")
        st.error(f"流式生成失败: {str(e)}")
        with st.expander("详细错误信息"):
            st.text(traceback.format_exc())
        st.session_state.is_generating = False
    finally:
        st.session_state.is_generating = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def reset_simulation():
    """重置模拟状态"""
    st.session_state.tokens = []
    st.session_state.token_ids = []
    st.session_state.current_position = 0
    st.session_state.generation_complete = False
    st.session_state.streaming_pending = False
    st.session_state.generated_text = ''
    st.session_state.generation_input_ids = None
    st.session_state.attention_mask = None
    if st.session_state.simulator:
        st.session_state.simulator.reset()
    if st.session_state.extractor:
        st.session_state.extractor.clear_history()


init_session_state()

st.markdown('<h1 class="main-header">KV Cache 可视化器</h1>', unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    model_source, model_name, checkpoint_path, selected_device, local_path, should_load = render_sidebar()

    if should_load:
        with st.spinner("加载中，请稍候..."):
            if model_source == "🤖 HuggingFace":
                success, msg = load_model("huggingface", model_name=model_name, device=selected_device)
            else:
                if os.path.isdir(local_path):
                    success, msg = load_model("huggingface", model_name=local_path, device=selected_device)
                else:
                    success, msg = load_model("pytorch", checkpoint_path=checkpoint_path, device=selected_device)
        if success:
            st.success(msg)
            st.rerun()
        else:
            st.error(msg)

# 主内容区
if not st.session_state.model_loaded:
    st.info("👈 请在左侧选择并加载模型")
    st.markdown("### 🚀 快速开始")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🤖 HuggingFace 模型**")
        st.markdown("- gpt2: 最快，适合测试")
        st.markdown("- TinyLlama: 小型模型，效果好")
        st.markdown("- Qwen2-0.5B: 国产模型")
    with col2:
        st.markdown("**📁 本地模型**")
        st.markdown("- 支持 HuggingFace 格式目录")
        st.markdown("- 支持 .pt/.pth/.safetensors")
        st.markdown("- 自动检测模型架构")
else:
    col_left, col_right = create_two_column_layout(1, 1)

    with col_left:
        st.markdown("### LLM 生成区域")
        render_template_selector()

        prompt = st.text_input("输入 Prompt", value="Hello, how are you?")
        batch_size = st.slider("生成 Token 数", 10, 500, 50)

        streaming_mode = st.checkbox("流式生成模式", value=False, help="启用后分批生成 token，每批后更新可视化")
        stream_batch_size = 5
        if streaming_mode:
            stream_batch_size = st.slider("流式批次大小", 1, 10, 5, help="每批生成的 token 数")

        action = render_generation_controls()

        if action == "reset":
            reset_simulation()
            st.rerun()
        elif action:
            if streaming_mode:
                st.session_state.streaming_pending = True
                st.session_state.streaming_prompt = prompt
                st.session_state.streaming_max_tokens = batch_size
                st.session_state.streaming_batch_size = stream_batch_size
                run_generation_streaming(prompt, max_new_tokens=batch_size, batch_size=stream_batch_size)
                st.rerun()
            else:
                run_generation_step(prompt, max_new_tokens=batch_size)
                st.rerun()

        render_generation_result(clean_bpe_token)
        render_debug_panel()
        render_replay_control(clean_bpe_token)

    with col_right:
        st.markdown("### KV Cache 可视化区域")

        if not st.session_state.simulator or not st.session_state.simulator.history:
            st.info("先生成 token 后才能查看 KV Cache 可视化")
        else:
            k_cache_list = [h.k_cache for h in st.session_state.simulator.history]
            v_cache_list = [h.v_cache for h in st.session_state.simulator.history]

            stats = st.session_state.visualizer.calculate_cache_stats(
                k_cache_list[:st.session_state.current_position],
                v_cache_list[:st.session_state.current_position]
            )

            render_export_buttons(stats)
            render_visualization_tabs(k_cache_list, v_cache_list, stats, clean_bpe_token)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>"
    "KV Cache 可视化器 | PyTorch + Transformers + Streamlit + Plotly"
    "</div>",
    unsafe_allow_html=True
)
