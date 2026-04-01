# app.py
import streamlit as st
import torch
from typing import Optional, Tuple
import os

from model_loader import ModelLoader, HuggingFaceLoader
from kvcache_extractor import KVCacheExtractor
from kvcache_simulator import KVCacheSimulator, MAX_HISTORY_LENGTH
from visualizer import KVCacheVisualizer
from device_utils import get_available_device, list_available_devices
from i18n import t, get_text, TRANSLATIONS
from theme import get_theme, get_theme_css, get_plotly_template, THEMES
from exporter import export_to_json, export_to_csv
from prompts import PROMPT_TEMPLATES, get_template_names, get_template, fill_template

# 页面配置
st.set_page_config(
    page_title="KV Cache 可视化器",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化主题和语言（需要在页面配置之后）
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "lang" not in st.session_state:
    st.session_state.lang = "zh"

# 应用主题
st.markdown(get_theme_css(st.session_state.theme), unsafe_allow_html=True)

# 自定义 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .token-display {
        font-family: 'Courier New', monospace;
        background-color: #f0f0f0;
        padding: 1rem;
        border-radius: 8px;
        font-size: 1.1rem;
        min-height: 60px;
    }
    .stButton>button {
        width: 100%;
        height: 3rem;
        font-size: 1.1rem;
    }
    .model-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """初始化 session state"""
    defaults = {
        'model_loader': None,
        'model': None,
        'tokenizer': None,
        'extractor': None,
        'simulator': None,
        'visualizer': None,
        'tokens': [],
        'token_ids': [],
        'current_position': 0,
        'generation_complete': False,
        'is_generating': False,
        'model_loaded': False,
        'model_config': {},
        'streaming_pending': False,
        'streaming_prompt': '',
        'streaming_max_tokens': 50,
        'streaming_batch_size': 5,
        'generated_text': '',  # 存储解码后的完整文本
        'generation_input_ids': None,
        'attention_mask': None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clean_bpe_token(token: str) -> str:
    """将 BPE token 转换为可读文本（Ġ 替换为空格）"""
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

        # 初始化 extractor 和 simulator
        num_layers = config.get('num_layers', 12)
        num_heads = config.get('num_heads', 12)
        num_kv_heads = config.get('num_kv_heads', num_heads)
        head_dim = config.get('head_dim', 64)

        st.session_state.extractor = KVCacheExtractor(
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim
        )
        st.session_state.simulator = KVCacheSimulator(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim
        )
        st.session_state.visualizer = KVCacheVisualizer(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim
        )

        return True, f"模型加载成功 ({config.get('num_layers', '?')} 层, {config.get('num_heads', '?')} 头)"
    except Exception as e:
        return False, f"加载失败: {str(e)}"


def run_generation_step(prompt: str, max_new_tokens: int = 50):
    """执行一步生成（生成一批 token）"""
    if st.session_state.generation_complete or st.session_state.is_generating:
        return

    st.session_state.is_generating = True

    try:
        model = st.session_state.model
        tokenizer = st.session_state.tokenizer
        extractor = st.session_state.extractor
        simulator = st.session_state.simulator

        # 清空之前的历史
        extractor.clear_history()
        simulator.reset()
        st.session_state.tokens = []
        st.session_state.token_ids = []

        # 编码 prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
        input_length = input_ids.shape[1]

        # 创建 attention mask (全1，表示所有token都参与attention)
        attention_mask = torch.ones_like(input_ids)

        # 注册 hooks
        handles = extractor.register_hooks(model)

        # 生成
        with torch.no_grad():
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        # 移除 hooks
        for handle in handles:
            handle.remove()

        # 解码生成的 token
        generated_ids = output[0][input_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids)

        # 更新 simulator
        for i, (token_id, token_str) in enumerate(zip(generated_ids.tolist(), generated_tokens)):
            pos = i + 1
            # 从 extractor 历史获取 KV Cache
            if pos <= len(extractor.kvcache_history):
                entry = extractor.kvcache_history[pos - 1]
                simulator.add_entry(pos, entry.k_cache, entry.v_cache, token_id, token_str)
            else:
                # 如果没有捕获到，创建空的
                k = torch.zeros(1, simulator.num_heads, 1, simulator.head_dim)
                v = torch.zeros(1, simulator.num_heads, 1, simulator.head_dim)
                simulator.add_entry(pos, k, v, token_id, token_str)

        st.session_state.tokens = generated_tokens
        st.session_state.generated_text = generated_text
        st.session_state.token_ids = generated_ids.tolist()
        st.session_state.current_position = len(generated_tokens)
        st.session_state.generation_complete = True

    except Exception as e:
        st.error(f"生成失败: {str(e)}")
    finally:
        st.session_state.is_generating = False
        # GPU 内存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_generation_streaming(prompt: str, max_new_tokens: int = 50, batch_size: int = 5):
    """流式生成 token，按批次更新 UI

    Args:
        prompt: 输入提示词
        max_new_tokens: 最大生成 token 数
        batch_size: 每批生成的 token 数
    """
    # MAX_HISTORY_LENGTH imported at top of file

    if st.session_state.generation_complete or st.session_state.is_generating:
        return

    st.session_state.is_generating = True

    try:
        model = st.session_state.model
        tokenizer = st.session_state.tokenizer
        extractor = st.session_state.extractor
        simulator = st.session_state.simulator

        # 如果是新的生成，清空之前的历史
        if not st.session_state.tokens:
            extractor.clear_history()
            simulator.reset()
            st.session_state.tokens = []
            st.session_state.token_ids = []
            st.session_state.current_position = 0

            # 编码 prompt
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
            attention_mask = torch.ones_like(input_ids)
            st.session_state.prompt_length = input_ids.shape[1]
            st.session_state.generation_input_ids = input_ids
            st.session_state.attention_mask = attention_mask
        else:
            # 继续之前的生成，使用累积的 input_ids
            input_ids = st.session_state.generation_input_ids
            attention_mask = st.session_state.attention_mask

        # 计算剩余需要生成的 token 数
        remaining = max_new_tokens - st.session_state.current_position
        if remaining <= 0:
            st.session_state.generation_complete = True
            return

        # 确定本批生成的数量
        current_batch_size = min(batch_size, remaining)

        # 注册 hooks
        handles = extractor.register_hooks(model)

        # 生成一批 token
        with torch.no_grad():
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=current_batch_size,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        # 移除 hooks
        for handle in handles:
            handle.remove()

        # 解码新生成的 token（从上次结束位置开始）
        start_pos = st.session_state.prompt_length + st.session_state.current_position
        new_ids = output[0][start_pos:start_pos + current_batch_size]
        new_tokens = tokenizer.convert_ids_to_tokens(new_ids)

        # 更新 extractor 和 simulator（仅新 token）
        for i, (token_id, token_str) in enumerate(zip(new_ids.tolist(), new_tokens)):
            pos = st.session_state.current_position + i + 1

            # 检查历史长度限制
            if len(simulator.history) >= MAX_HISTORY_LENGTH:
                st.session_state.generation_complete = True
                st.session_state.is_generating = False
                break

            if pos <= len(extractor.kvcache_history):
                entry = extractor.kvcache_history[pos - 1]
                simulator.add_entry(pos, entry.k_cache, entry.v_cache, token_id, token_str)
            else:
                k = torch.zeros(1, simulator.num_heads, 1, simulator.head_dim)
                v = torch.zeros(1, simulator.num_heads, 1, simulator.head_dim)
                simulator.add_entry(pos, k, v, token_id, token_str)

        # 更新 session state
        st.session_state.token_ids.extend(new_ids.tolist())
        st.session_state.tokens.extend(new_tokens)
        st.session_state.current_position += len(new_tokens)

        # 更新累积的 input_ids 用于下次生成
        st.session_state.generation_input_ids = output

        # 更新解码文本
        st.session_state.generated_text = tokenizer.decode(st.session_state.token_ids, skip_special_tokens=True)

        # 检查是否完成
        if st.session_state.current_position >= max_new_tokens:
            st.session_state.generation_complete = True
            st.session_state.streaming_pending = False
        else:
            # 还有更多 token 需要生成，重置 hooks 以便下次继续
            st.session_state.is_generating = False

    except Exception as e:
        st.error(f"流式生成失败: {str(e)}")
        st.session_state.is_generating = False
    finally:
        # GPU 内存清理
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


# 初始化
init_session_state()

# 流式生成自动继续检查
# 如果有待处理的流式生成任务，继续执行
if (st.session_state.get('streaming_pending', False) and
    st.session_state.model_loaded and
    not st.session_state.generation_complete):
    run_generation_streaming(
        st.session_state.streaming_prompt,
        max_new_tokens=st.session_state.streaming_max_tokens,
        batch_size=st.session_state.streaming_batch_size
    )
    st.rerun()

# 主界面
st.markdown('<h1 class="main-header">KV Cache 可视化器</h1>', unsafe_allow_html=True)

# 侧边栏：模型选择
with st.sidebar:
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

        checkpoint_path = ""
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

    if st.button("🚀 加载模型", type="primary", disabled=not can_load):
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

# 主内容区
if not st.session_state.model_loaded:
    st.info("👈 请在左侧选择并加载模型")

    # 展示快速开始选项
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
    # 左右分栏布局
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### LLM 生成区域")

        # 预设模板
        lang = st.session_state.lang
        st.markdown(f"**{t('prompt_templates', lang)}**")
        template_names = get_template_names(lang)
        selected_template = st.selectbox(
            "选择模板" if lang == "zh" else "Select Template",
            ["-"] + template_names,
            label_visibility="collapsed"
        )

        if selected_template != "-" and selected_template in template_names:
            template = get_template(selected_template, lang)
            st.info(template.description_zh if lang == "zh" else template.description_en)

        prompt = st.text_input("输入 Prompt", value="Hello, how are you?")

        # 批量大小
        batch_size = st.slider("生成 Token 数", 5, 50, 20)

        # 流式生成模式
        streaming_mode = st.checkbox("流式生成模式", value=False, help="启用后分批生成 token，每批后更新可视化")

        # 流式生成批次大小
        stream_batch_size = 5
        if streaming_mode:
            stream_batch_size = st.slider("流式批次大小", 1, 10, 5, help="每批生成的 token 数")

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if not st.session_state.generation_complete:
                if st.button("▶ 开始生成" if not st.session_state.is_generating else "⏳ 生成中..."):
                    if streaming_mode:
                        # 设置流式生成状态
                        st.session_state.streaming_pending = True
                        st.session_state.streaming_prompt = prompt
                        st.session_state.streaming_max_tokens = batch_size
                        st.session_state.streaming_batch_size = stream_batch_size
                        st.session_state.is_generating = True
                        run_generation_streaming(prompt, max_new_tokens=batch_size, batch_size=stream_batch_size)
                        # 如果未完成，下次 rerun 时继续
                        if not st.session_state.generation_complete:
                            st.rerun()
                    else:
                        with st.spinner("生成中..."):
                            run_generation_step(prompt, max_new_tokens=batch_size)
                    st.rerun()
            else:
                st.success("✓ 生成完成")
        with col_btn2:
            if st.button("🔄 重新生成"):
                reset_simulation()
                st.rerun()

        st.markdown("**生成结果:**")
        if st.session_state.tokens:
            # 清理 BPE token 并显示
            cleaned_tokens = [clean_bpe_token(t) for t in st.session_state.tokens[:st.session_state.current_position]]
            tokens_display = "".join(cleaned_tokens)
            st.markdown(f'<div class="token-display">{tokens_display}</div>', unsafe_allow_html=True)
        else:
            st.info("点击「开始生成」按钮启动")

        # 进度
        if st.session_state.tokens:
            progress = st.session_state.current_position / max(len(st.session_state.tokens), 1)
            st.progress(progress, text=f"Token {st.session_state.current_position}/{len(st.session_state.tokens)}")

        # 调试信息
        if st.session_state.model_loaded and st.session_state.extractor:
            with st.expander("🔧 调试信息"):
                history_len = len(st.session_state.extractor.kvcache_history)
                st.text(f"捕获的 KV Cache 条目数: {history_len}")
                st.text(f"Token 数量: {len(st.session_state.tokens)}")
                if st.session_state.simulator:
                    st.text(f"Simulator 历史长度: {len(st.session_state.simulator.history)}")

        # 回放控制
        if st.session_state.generation_complete and st.session_state.simulator:
            st.markdown("---")
            st.markdown("### 🎚️ 回放控制")
            max_pos = len(st.session_state.simulator.history)
            slider_pos = st.slider(
                "拖动进度条回看任意位置",
                min_value=1,
                max_value=max_pos,
                value=max_pos
            )
            st.session_state.current_position = slider_pos

            if 1 <= slider_pos <= len(st.session_state.tokens):
                st.markdown(f"**位置:** Token {slider_pos} = \"{st.session_state.tokens[slider_pos-1]}\"")

    with col_right:
        st.markdown("### KV Cache 可视化区域")

        if not st.session_state.simulator or not st.session_state.simulator.history:
            st.info("先生成 token 后才能查看 KV Cache 可视化")
        else:
            k_cache_list = [h.k_cache for h in st.session_state.simulator.history]
            v_cache_list = [h.v_cache for h in st.session_state.simulator.history]

            # 计算统计数据（需要在导出按钮之前）
            stats = st.session_state.visualizer.calculate_cache_stats(
                k_cache_list[:st.session_state.current_position],
                v_cache_list[:st.session_state.current_position]
            )

            # 导出按钮
            lang = st.session_state.lang
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

            # Tab 选择
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 矩阵热力图", "📈 序列视图", "🔳 层级分布", "📐 统计数据", "🖥️ 综合仪表盘", "🔥 层级能量"])

            with tab1:
                if st.session_state.current_position > 0:
                    k_cache = st.session_state.simulator.history[st.session_state.current_position - 1].k_cache
                    fig = st.session_state.visualizer.create_heatmap(
                        k_cache,
                        title=f"KV Cache 热力图 (Token {st.session_state.current_position})"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # 调试信息
                    if k_cache.abs().sum() < 1e-6:
                        st.warning("⚠️ KV Cache 数据显示为空（可能是因为 Hook 未捕获到数据）")
                        with st.expander("调试信息"):
                            st.code(f"k_cache shape: {k_cache.shape}")
                            st.code(f"k_cache sum: {k_cache.abs().sum()}")
                            if st.session_state.extractor:
                                debug_info = st.session_state.extractor.get_debug_info()
                                if debug_info:
                                    st.text("Hook 捕获记录:\n" + "\n".join(debug_info[-10:]))

            with tab2:
                fig = st.session_state.visualizer.create_sequence_view(
                    st.session_state.tokens[:st.session_state.current_position],
                    k_cache_list[:st.session_state.current_position],
                    title="Token 序列生成视图"
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab3:
                if st.session_state.current_position > 0:
                    k_cache = st.session_state.simulator.history[st.session_state.current_position - 1].k_cache
                    fig = st.session_state.visualizer.create_layer_view(
                        k_cache,
                        title=f"层级注意力分布 (Token {st.session_state.current_position})"
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with tab4:
                # 统计数据展示
                col_stat1, col_stat2 = st.columns(2)
                with col_stat1:
                    st.metric("生成 Token 数", stats['num_generated_tokens'])
                    st.metric("缓存 Token 数", stats['num_cached_tokens'])
                    st.metric("Cache 效率", f"{stats['cache_efficiency']}%")
                with col_stat2:
                    st.metric("峰值内存", f"{stats['peak_memory_mb']} MB")
                    st.metric("平均层能量", f"{stats['avg_layer_energy']:.4f}")
                    st.metric("Attention 稀疏度", f"{stats['sparsity']}%")

                # 详细统计表格
                st.markdown("#### 详细统计")
                fig_stats = st.session_state.visualizer.create_stats_gauge(stats)
                st.plotly_chart(fig_stats, use_container_width=True)

            with tab5:
                fig = st.session_state.visualizer.create_dashboard(
                    st.session_state.tokens,
                    k_cache_list,
                    v_cache_list,
                    st.session_state.current_position,
                    title="KV Cache 综合仪表盘"
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab6:
                if st.session_state.current_position > 0:
                    fig = st.session_state.visualizer.create_layer_energy_heatmap(
                        k_cache_list[:st.session_state.current_position],
                        title="层级能量热力图"
                    )
                    st.plotly_chart(fig, use_container_width=True)

# 底部信息
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>"
    "KV Cache 可视化器 | PyTorch + Transformers + Streamlit + Plotly"
    "</div>",
    unsafe_allow_html=True
)
