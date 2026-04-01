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