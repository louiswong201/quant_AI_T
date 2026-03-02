"""
内置 Prompt 模板：将 query + 检索上下文格式化为常见 LLM 输入格式
"""

from typing import Optional

# 模板名 -> (system 可选, user 模板，占位符 {query} {context})
_TEMPLATES = {
    "plain": (None, "基于以下参考内容回答问题。\n\n参考：\n{context}\n\n问题：{query}"),
    "alpaca": (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
        "### Instruction:\nBased on the context below, answer the question.\n\n### Context:\n{context}\n\n### Question:\n{query}\n\n### Response:\n",
    ),
    "chatml": (
        "You are a helpful assistant. Answer the question using only the provided context.",
        "<|im_start|>user\nContext:\n{context}\n\nQuestion: {query}<|im_end|>\n<|im_start|>assistant\n",
    ),
}


def format_prompt(
    template_name: str = "plain",
    query: str = "",
    context: str = "",
    system: Optional[str] = None,
    max_context_chars: int = 0,
) -> str:
    """
    将 query 与 context 按预设模板拼成单段或两段（system + user）字符串。

    Args:
        template_name: "plain" | "alpaca" | "chatml"
        query: 用户问题
        context: 检索得到的上下文字符串
        system: 若提供则覆盖模板的 system 段
        max_context_chars: 若 > 0 则截断 context 到该长度

    Returns:
        格式化后的字符串；若为 chatml/alpaca 且模板含 system，返回 "system\\n\\nuser_part"。
    """
    if max_context_chars > 0 and len(context) > max_context_chars:
        context = context[:max_context_chars] + "..."
    t = _TEMPLATES.get(template_name) or _TEMPLATES["plain"]
    sys_part, user_tpl = t[0], t[1]
    if system is not None:
        sys_part = system
    user_part = user_tpl.format(query=query, context=context or "(无)")
    if sys_part:
        return sys_part.strip() + "\n\n" + user_part.strip()
    return user_part.strip()


def list_templates() -> list:
    """返回内置模板名称列表。"""
    return list(_TEMPLATES.keys())
