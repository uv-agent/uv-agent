from __future__ import annotations

from uv_agent.plugins.i18n import LocalizedText


TEXTS: dict[str, LocalizedText] = {
    "goal": {"en": "Goal", "zh": "Goal"},
    "goal_panel_hint": {
        "en": "Manage plugin-stored goal state for this thread",
        "zh": "管理当前线程的插件目标状态",
    },
    "goal_enable": {"en": "Enable goal mode", "zh": "开启 goal 模式"},
    "goal_enable_hint": {
        "en": "Mark Goal for the next send; plugin state is created or reused then",
        "zh": "标记下一次发送启用 Goal；届时才创建或复用插件状态",
    },
    "goal_disable": {"en": "Disable goal mode", "zh": "关闭 goal 模式"},
    "goal_disable_hint": {
        "en": "Preserve state and notify the model on the next send",
        "zh": "保留状态，并在下次发送前通知模型",
    },
    "goal_disable_requires_completed": {
        "en": "Can disable only after the model has sent a final reply",
        "zh": "只能在模型已经发送最终回复后关闭",
    },
    "goal_state": {"en": "Goal state", "zh": "Goal 状态"},
    "goal_state_hint": {
        "en": "Show plugin-stored goal state",
        "zh": "查看插件存储的 goal 状态",
    },
    "goal_state_pending": {
        "en": "Goal state will be created when you send the next message",
        "zh": "发送下一条消息时才会创建 Goal 状态",
    },
    "goal_state_missing": {"en": "Not created yet", "zh": "尚未创建"},
    "goal_state_empty": {"en": "Empty state", "zh": "状态为空"},
    "goal_state_read_error": {"en": "Could not read state", "zh": "无法读取状态"},
    "goal_state_truncated": {
        "en": "Preview truncated after {limit} characters",
        "zh": "预览已在 {limit} 个字符后截断",
    },
    "goal_reset": {"en": "Reset goal state", "zh": "重置目标状态"},
    "goal_reset_hint": {
        "en": "Allowed only while goal mode is disabled",
        "zh": "仅在 goal 模式关闭时允许",
    },
    "goal_reset_disabled_active": {
        "en": "Disable goal mode before resetting state",
        "zh": "请先关闭 goal 模式再重置状态",
    },
    "goal_enabled": {"en": "enabled", "zh": "已开启"},
    "goal_disabled": {"en": "disabled", "zh": "已关闭"},
    "goal_enabled_flash": {"en": "Goal mode enabled", "zh": "Goal 模式已开启"},
    "goal_disabled_flash": {"en": "Goal mode disabled", "zh": "Goal 模式已关闭"},
    "goal_reset_flash": {"en": "Goal state reset", "zh": "Goal 状态已重置"},
    "goal_already_disabled": {
        "en": "Goal mode is already disabled",
        "zh": "Goal 模式已经关闭",
    },
}
