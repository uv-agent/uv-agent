from __future__ import annotations

from uv_agent.plugins.i18n import LocalizedText


TEXTS: dict[str, LocalizedText] = {
    "skills": {"en": "Skills", "zh": "Skills"},
    "mention_skills_hint": {
        "en": "Search and Enter to insert a skill mention",
        "zh": "搜索后按 Enter 插入 skill 引用",
    },
    "no_skills": {
        "en": "no .agents/skills entries discovered",
        "zh": "没有发现 .agents/skills 条目",
    },
}
