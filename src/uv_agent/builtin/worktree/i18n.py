from __future__ import annotations

from uv_agent.plugins.i18n import LocalizedText


TEXTS: dict[str, LocalizedText] = {
    "worktree": {"en": "worktree", "zh": "worktree"},
    "worktree_title": {"en": "Worktree", "zh": "Worktree"},
    "worktree_panel_hint": {
        "en": "Create or manage a branch worktree",
        "zh": "创建或管理分支 worktree",
    },
    "worktree_create": {"en": "Create new branch worktree", "zh": "创建新分支 worktree"},
    "worktree_create_hint": {
        "en": "Enter a branch name in the next panel",
        "zh": "在下一级面板输入分支名",
    },
    "worktree_create_title": {"en": "Create Worktree", "zh": "创建 Worktree"},
    "worktree_branch_placeholder": {"en": "branch-name", "zh": "branch-name"},
    "worktree_branch_hint": {
        "en": "Enter a Git branch name without / or path separators · Enter creates · Esc closes",
        "zh": "输入不含 / 或路径分隔符的 Git 分支名 · Enter 创建 · Esc 关闭",
    },
    "worktree_current": {"en": "Current worktree", "zh": "当前 worktree"},
    "worktree_merge": {"en": "Append merge-back prompt", "zh": "追加合并提示词"},
    "worktree_merge_hint": {
        "en": "Append instructions to the composer without sending",
        "zh": "追加到输入框但不自动发送",
    },
    "worktree_delete": {"en": "Delete worktree and branch", "zh": "删除 worktree 和分支"},
    "worktree_delete_hint": {
        "en": "Force removes the worktree and deletes the local branch",
        "zh": "强制移除 worktree 并删除本地分支",
    },
    "worktree_delete_confirm": {
        "en": "Confirm delete worktree and local branch",
        "zh": "确认删除 worktree 和本地分支",
    },
    "worktree_delete_confirm_hint": {
        "en": "This runs git worktree remove --force and git branch -D",
        "zh": "将执行 git worktree remove --force 和 git branch -D",
    },
    "worktree_none": {"en": "Current thread is not in a worktree", "zh": "当前线程不在 worktree 中"},
    "worktree_created": {"en": "Worktree created", "zh": "Worktree 已创建"},
    "worktree_prompt_appended": {"en": "Merge prompt appended", "zh": "已追加合并提示词"},
    "worktree_deleted": {"en": "Worktree and branch deleted", "zh": "Worktree 和分支已删除"},
    "worktree_invalid_branch": {"en": "Invalid branch name", "zh": "分支名无效"},
    "worktree_error": {"en": "Worktree error", "zh": "Worktree 错误"},
}
