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