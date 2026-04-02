"""Reusable UI components."""

import streamlit as st
import torch

from visualization.comparison import ComparisonVisualizer


def render_generation_controls():
    """Render generation buttons and status."""
    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        if st.session_state.generation_complete:
            st.success("✓ 生成完成")
        elif st.session_state.is_generating:
            st.info("⏳ 生成中...")
        else:
            if st.button("▶ 开始生成" if not st.session_state.streaming_pending else "▶ 继续生成"):
                return True
    with col_btn2:
        if st.button("🔄 重新生成"):
            return "reset"

    return None


def render_debug_panel():
    """Render debug information expander."""
    if st.session_state.model_loaded and st.session_state.extractor:
        with st.expander("🔧 调试信息"):
            history_len = len(st.session_state.extractor.kvcache_history)
            st.text(f"捕获的 KV Cache 条目数: {history_len}")
            st.text(f"Token 数量: {len(st.session_state.tokens)}")
            if st.session_state.simulator:
                st.text(f"Simulator 历史长度: {len(st.session_state.simulator.history)}")


def render_template_selector():
    """Render template selection dropdown."""
    lang = st.session_state.lang
    from prompts import get_template_names, get_template
    from i18n import t

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

    return selected_template


def render_generation_result(clean_bpe_token_func):
    """Render generation result display."""
    st.markdown("**生成结果:**")
    if st.session_state.tokens:
        cleaned_tokens = [clean_bpe_token_func(t) for t in st.session_state.tokens[:st.session_state.current_position]]
        tokens_display = "".join(cleaned_tokens)
        st.markdown(f'<div class="token-display">{tokens_display}</div>', unsafe_allow_html=True)
    else:
        st.info("点击「开始生成」按钮启动")

    if st.session_state.tokens:
        progress = st.session_state.current_position / max(len(st.session_state.tokens), 1)
        st.progress(progress, text=f"Token {st.session_state.current_position}/{len(st.session_state.tokens)}")


def render_replay_control(clean_bpe_token_func):
    """Render replay slider control."""
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
            cleaned = clean_bpe_token_func(st.session_state.tokens[slider_pos-1])
            st.markdown(f"**位置:** Token {slider_pos} = \"{cleaned}\"")


def render_export_buttons(stats):
    """Render JSON and CSV export buttons."""
    from i18n import t
    from exporter import export_to_json, export_to_csv

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


def render_visualization_tabs(k_cache_list, v_cache_list, stats, clean_bpe_token_func):
    """Render all visualization tabs."""
    cleaned_tokens = [clean_bpe_token_func(t) for t in st.session_state.tokens[:st.session_state.current_position]]
    visualizer = st.session_state.visualizer

    # 提取 attention weights 列表
    attn_weights_list = []
    if st.session_state.simulator:
        for entry in st.session_state.simulator.history:
            attn_weights_list.append(getattr(entry, 'attn_weights', None))

    tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
        ["📈 序列视图", "🔳 层级分布", "📐 统计数据", "🖥️ 综合仪表盘", "🔥 层级能量", "🧠 Attention", "⭐ Token重要性", "📊 层分析"]
    )

    with tab2:
        fig = visualizer.create_sequence_view(
            cleaned_tokens,
            k_cache_list[:st.session_state.current_position],
            title="Token 序列生成视图"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if st.session_state.current_position > 0:
            k_cache = st.session_state.simulator.history[st.session_state.current_position - 1].k_cache
            fig = visualizer.create_layer_view(
                k_cache,
                title=f"层级注意力分布 (Token {st.session_state.current_position})"
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("生成 Token 数", stats['num_generated_tokens'])
            st.metric("缓存 Token 数", stats['num_cached_tokens'])
            st.metric("Cache 效率", f"{stats['cache_efficiency']}%")
        with col_stat2:
            st.metric("峰值内存", f"{stats['peak_memory_mb']} MB")
            st.metric("平均层能量", f"{stats['avg_layer_energy']:.4f}")
            st.metric("Attention 稀疏度", f"{stats['sparsity']}%")

        st.markdown("#### 详细统计")
        fig_stats = visualizer.create_stats_gauge(stats)
        st.plotly_chart(fig_stats, use_container_width=True)

    with tab5:
        fig = visualizer.create_dashboard(
            cleaned_tokens,
            k_cache_list,
            v_cache_list,
            st.session_state.current_position,
            title="KV Cache 综合仪表盘"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab6:
        if st.session_state.current_position > 0:
            fig = visualizer.create_layer_energy_heatmap(
                k_cache_list[:st.session_state.current_position],
                title="层级能量热力图"
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab7:
        # Attention Pattern 可视化
        if attn_weights_list and st.session_state.current_position > 0:
            current_attn = attn_weights_list[st.session_state.current_position - 1]

            # 选择视图模式
            attn_view_mode = st.radio(
                "Attention 视图模式",
                ["整体视图", "Head 分布", "Summary"],
                horizontal=True
            )

            if current_attn is not None:
                if attn_view_mode == "整体视图":
                    fig = visualizer.create_attention_heatmap(
                        current_attn,
                        cleaned_tokens,
                        title=f"Attention Pattern (Token {st.session_state.current_position})"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                elif attn_view_mode == "Head 分布":
                    fig = visualizer.create_attention_per_head(
                        current_attn,
                        cleaned_tokens,
                        title=f"Attention per Head (Token {st.session_state.current_position})"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                else:  # Summary
                    fig = visualizer.create_attention_summary(
                        attn_weights_list[:st.session_state.current_position],
                        cleaned_tokens,
                        title="Attention Summary"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("当前 token 的 Attention 数据不可用")
        else:
            st.info("先生成 token 后才能查看 Attention 可视化")

    with tab8:
        # Token Importance 可视化
        if st.session_state.current_position > 0 and k_cache_list:
            # 选择视图模式
            importance_view_mode = st.radio(
                "Token 重要性视图模式",
                ["排序柱状图", "热力图"],
                horizontal=True
            )

            if importance_view_mode == "排序柱状图":
                fig = visualizer.create_token_importance(
                    k_cache_list[:st.session_state.current_position],
                    cleaned_tokens,
                    title="Token 重要性排序 (KV Energy)"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = visualizer.create_token_importance_heatmap(
                    k_cache_list[:st.session_state.current_position],
                    cleaned_tokens,
                    title="Token 重要性热力图"
                )
                st.plotly_chart(fig, use_container_width=True)

            # 显示统计信息
            st.markdown("#### Top 5 最重要 Token:")
            energies = []
            for k in k_cache_list[:st.session_state.current_position]:
                if k is not None and k.numel() > 0:
                    energies.append(torch.norm(k).item())
                else:
                    energies.append(0)

            token_energies = list(zip(cleaned_tokens, energies))
            token_energies.sort(key=lambda x: x[1], reverse=True)

            for i, (token, energy) in enumerate(token_energies[:5]):
                st.markdown(f"**{i+1}.** `{token}` - Energy: `{energy:.4f}`")
        else:
            st.info("先生成 token 后才能查看 Token 重要性")

    with tab9:
        render_layer_analysis_tab(
            k_cache_list,
            attn_weights_list,
            visualizer
        )


def render_layer_analysis_tab(k_cache_list, attn_weights_list, visualizer):
    """Render the Layer Analysis tab with energy evolution and attention stats."""
    st.markdown("### 📊 层能量变化")
    if k_cache_list:
        fig_evo = visualizer.create_layer_energy_evolution(
            k_cache_list,
            title="各层能量随生成位置的变化"
        )
        st.plotly_chart(fig_evo, use_container_width=True)
    else:
        st.info("先生成 token 后才能查看层能量变化")

    st.markdown("---")
    st.markdown("### 🔄 Attention 层间统计")

    metric = st.radio(
        "选择统计指标",
        ["coverage", "sparsity", "max_val"],
        format_func=lambda x: {
            "coverage": "覆盖度",
            "sparsity": "稀疏度",
            "max_val": "最大值"
        }[x],
        horizontal=True
    )

    stats = visualizer.calculate_attention_stats_by_layer(attn_weights_list)
    if stats:
        fig_stats = visualizer.create_attention_layer_stats(
            stats,
            metric=metric,
            title=f"Attention {metric} by Layer"
        )
        st.plotly_chart(fig_stats, use_container_width=True)
    else:
        st.info("先生成 token 后才能查看 Attention 层间统计")


def render_comparison_panel(clean_bpe_token_func):
    """Render multi-model comparison panel.

    This panel shows comparison visualizations when multiple models are loaded.
    For single model, shows a placeholder message.
    """
    from i18n import t

    st.markdown("### 📊 模型对比")

    # 获取模型管理器（如果存在）
    model_manager = getattr(st.session_state, 'model_manager', None)

    if model_manager is None or model_manager.num_models < 2:
        # 单模型模式 - 显示提示
        st.info("⚠️ 对比模式需要加载至少 2 个模型")

        if model_manager and model_manager.num_models == 1:
            model = model_manager.active_model
            if model:
                st.markdown(f"**当前已加载:** `{model.name}` ({model.num_layers} 层, {model.num_heads} 头)")

        st.markdown("""
        **多模型对比模式:**

        1. 在侧边栏加载第一个模型
        2. 使用「添加对比模型」加载第二个模型
        3. 返回此处查看对比可视化

        **支持三种对比模式:**
        - 📐 **分屏对比** - 左右并排显示
        - 🔄 **叠加对比** - 同一图表叠加曲线
        - 📊 **统计对比** - 柱状图对比各项指标
        """)
        return

    # 多模型模式 - 显示对比可视化
    comparison_visualizer = ComparisonVisualizer()

    # 选择对比模式
    comparison_mode = st.radio(
        "对比模式",
        ["📐 分屏对比", "🔄 叠加对比", "📊 统计对比"],
        horizontal=True
    )

    # 获取所有模型的数据
    model_data = {}
    model_stats = {}

    for model_id in model_manager.model_ids:
        model = model_manager.get_model(model_id)
        if model and model.simulator:
            # 获取 token 和 cache 数据
            tokens = model.simulator.tokens or []
            k_cache_list = [h.k_cache for h in model.simulator.history]
            v_cache_list = [h.v_cache for h in model.simulator.history]
            attn_weights = [getattr(h, 'attn_weights', None) for h in model.simulator.history]

            model_data[model_id] = {
                "tokens": tokens,
                "k_cache_list": k_cache_list,
                "v_cache_list": v_cache_list,
                "attn_weights": attn_weights,
            }

            # 计算统计
            if model.visualizer and k_cache_list and v_cache_list:
                stats = model.visualizer.calculate_cache_stats(k_cache_list, v_cache_list)
                model_stats[model_id] = stats

    if comparison_mode == "📐 分屏对比":
        view_type = st.selectbox(
            "分屏视图类型",
            ["sequence", "layer"],
            format_func=lambda x: {"sequence": "序列视图", "layer": "层级分布"}.get(x, x)
        )

        fig = comparison_visualizer.create_split_view(
            model_data,
            view_type=view_type,
            title=f"Model Comparison - {view_type}"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif comparison_mode == "🔄 叠加对比":
        view_type = st.selectbox(
            "叠加视图类型",
            ["sequence", "layer"],
            format_func=lambda x: {"sequence": "序列视图", "layer": "层级分布"}.get(x, x)
        )

        fig = comparison_visualizer.create_overlay_view(
            model_data,
            view_type=view_type,
            title=f"Model Comparison - Overlay"
        )
        st.plotly_chart(fig, use_container_width=True)

    else:  # 统计对比
        # 显示模型信息
        st.markdown("#### 模型统计指标对比:")
        for model_id, stats in model_stats.items():
            with st.expander(f"📌 {model_id}: {stats.get('num_generated_tokens', 0)} tokens"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Cache 效率", f"{stats.get('cache_efficiency', 0):.1f}%")
                    st.metric("稀疏度", f"{stats.get('sparsity', 0):.1f}%")
                with col2:
                    st.metric("峰值内存", f"{stats.get('peak_memory_mb', 0):.2f} MB")
                    st.metric("平均层能量", f"{stats.get('avg_layer_energy', 0):.4f}")
                with col3:
                    st.metric("生成 Token", stats.get('num_generated_tokens', 0))
                    st.metric("缓存 Token", stats.get('num_cached_tokens', 0))

        # 统计柱状图对比
        fig = comparison_visualizer.create_stats_comparison(
            model_stats,
            title="Statistics Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)

        # 层级能量热力图对比
        st.markdown("#### 层级能量热力图对比:")
        fig = comparison_visualizer.create_layer_energy_comparison_heatmap(
            model_data,
            title="Layer Energy Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)
