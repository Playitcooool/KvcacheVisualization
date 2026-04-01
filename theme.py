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