"""
Single source of truth for uv-agent prompt strings.

LLM-facing prompt constants live here. Modules import from this file
to avoid duplication and circular imports (this module has zero internal
uv_agent dependencies).

Consolidated from the legacy agent/prompts.py plus prompt strings that were
previously duplicated inline across compaction, engine, goal mode, worktree,
project rules, skills, and runtime subagent helpers.
"""

from __future__ import annotations

# ===========================================================================
# Tool definition
# ===========================================================================

PYTHON_TOOL = {
    "type": "function",
    "name": "run_python",
    "description": (
        "Run a complete, standalone Python script in a fresh Python process. "
        "It runs in the thread's active cwd, using the project shared script venv. "
        "Use Python-native control flow and imports—not shell-style fragments—to interact "
        "with the outside world. Treat one call as a work-unit script: batch related "
        "commands, searches, reads, edits, and focused verification with conditional "
        "fallbacks, then print one bounded summary. Do not make one run_python call per "
        "command, file read, or helper call when related steps are foreseeable. Prefer "
        "runtime helpers, especially run_process_text for ordinary external commands. "
        "Use this as the only way to inspect files, run commands, access the network, "
        "or perform external actions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": (
                    "Complete, valid Python source for one standalone script. "
                    "Use normal Python syntax, not shell-style pseudo-code; include imports "
                    "and setup. Write a small Python program that coordinates related steps "
                    "with variables, functions, loops, conditionals, try/except, data "
                    "structures, dependencies, and uv_agent_runtime helper calls."
                ),
            },
            "timeout_s": {
                "type": "number",
                "description": "Execution timeout in seconds.",
                "default": 7200,
            },
        },
        "required": ["code"],
        "additionalProperties": False,
    },
    "strict": True,
}

# ===========================================================================
# Compaction judge request
# ===========================================================================

COMPACTION_JUDGE_REQUEST = (
    "<compaction_judge_request>\n"
    "You are about to receive a user task. Before answering, output a\n"
    "one-line JSON judgement about the conversation state. Return ONLY the\n"
    "JSON line, no backticks, no explanation:\n\n"
    '{"remaining_calls_bucket":"<0_10|10_30|30_60|60_plus>",'
    '"history_dependency":"<low|medium|high|exact>"}\n'
    "\n"
    "remaining_calls_bucket: how many more model calls will this task need?\n"
    "history_dependency: how much does the task depend on exact original\n"
    "  wording in the conversation above? 'low' for general continuation,\n"
    "  'medium' for moderate dependence, 'high' for strong dependence on\n"
    "  specific details, 'exact' when every word matters (diffs, error\n"
    "  messages, config values, exact quotes).\n"
    "</compaction_judge_request>\n"
)

# ===========================================================================
# Core prompts
# ===========================================================================

COMPACTION_SUMMARIZATION_PROMPT = (
    "You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for "
    "another LLM that will resume the task.\n\n"
    "Include:\n"
    "- Current progress and key decisions made\n"
    "- Important context, constraints, or user preferences\n"
    "- What remains to be done (clear next steps)\n"
    "- Any critical data, examples, or references needed to continue\n\n"
    "Be concise, structured, and focused on helping the next LLM seamlessly continue the work."
)
TITLE_GENERATION_PROMPT = (
    "Create a concise, title-like name for this uv-agent thread from the user's first message. "
    "Capture the user's underlying task or intent, not a literal rewrite of the sentence. "
    "For broad or vague questions, use an abstract noun-phrase style. For example, "
    "a message asking what kind of project this is should become a title like "
    "Project content inquiry. "
    "Return only the title, without quotes or punctuation. Prefer the user's language. "
    "Keep it under 8 words or 24 CJK characters."
)
BRANCH_NAME_GENERATION_PROMPT = (
    "Create a short git branch slug from the user's task. Capture the concrete action and object. "
    "Return only the slug: ASCII lowercase letters, digits, and hyphens. No spaces, slashes, quotes, "
    "punctuation, explanations, or prefixes. Keep it at 30 characters or fewer. Prefer verb-object "
    "phrases such as fix-login-redirect, add-dark-mode, or refactor-parser."
)
COMPACTED_CONTEXT_CONTINUATION = (
    "The retained-history messages above may include earlier user or assistant messages preserved for continuity. "
    "Continue from this compacted context and resume any unfinished task. "
    "Use the summary and retained history as prior conversation state, then take the next concrete step without asking the user to repeat information already captured."
)
TOOL_ATTACHMENT_CONTEXT_BRIDGE = (
    "Tool execution completed. Additional visual context produced by the tool "
    "is provided in the next user message."
)
POST_TOOL_COMPACTION_BRIDGE = (
    "I have received the tool results. When the next user message asks for "
    "context compaction, I will produce the requested compaction summary "
    "according to those instructions, preserving tool results, decisions, "
    "file changes, constraints, and unresolved tasks accurately."
)
INTERRUPTED_TOOL_CONTEXT_BRIDGE = (
    "A tool call did not produce a complete tool result. Continue from the available context."
)
INTERRUPTED_STREAM_CONTEXT_BRIDGE = (
    "An assistant response did not complete. Continue from the available context."
)


# ---------------------------------------------------------------------------
# Engine-level inline prompts (extracted from engine.py)
# ---------------------------------------------------------------------------

BRANCH_SLUG_INSTRUCTION = (
    "Generate a short git branch slug. Return only the slug."
)

THREAD_TITLE_INSTRUCTION = (
    "Generate a short thread title. Return only the title."
)

PRE_TURN_JUDGE_ERROR_STDERR = (
    "ERROR: Do not call tools during pre-turn judgement. Return only the JSON line."
)

TOKEN_ESTIMATION_WARNING = (
    "Provider token usage is unavailable; context compaction is "
    "using a local estimate and may fail calls or compact too late."
)

COMPACTION_TOOL_ERROR_STDERR = (
    "ERROR: Tool calls are not allowed during context compaction. "
    "Return the compaction summary as plain prose text only."
)

INTERRUPTED_TOOL_ERROR = (
    "Tool call did not complete because the user interrupted this turn. "
    "Do not assume the tool ran successfully."
)

ACTIVE_CWD_NOTICE_TEMPLATE = (
    "<active_cwd_notice>\n"
    "The active working directory for run_python is now {active_cwd_rel}. "
    "The thread opened at {initial_cwd_rel}. "
    "Relative paths and automatic directory rules follow the active working directory.\n"
    "</active_cwd_notice>"
)

CONTEXT_REMOVED_ALL = (
    "<context_update id=\"runtime_context\" status=\"removed\">\n"
    "Previously available runtime context is no longer present. "
    "Do not rely on older runtime context unless it appears again.\n"
    "</context_update>"
)

CONTEXT_REMOVED_SOME_PREFIX = (
    "\n\n<context_update_removed id=\"runtime_context\">\n"
    "Some previously available runtime context is no longer present. "
    "Do not rely on older appended content for removed skills or MCP servers unless they appear again.\n"
)

CONTEXT_REMOVED_SOME_SUFFIX = "\n</context_update_removed>"

CONTEXT_UPDATE_CURRENT_PREFIX = (
    "<context_update id=\"runtime_context\" status=\"current\">\n"
    "The following runtime context is current. It updates only the listed content; "
    "prior runtime context remains current within this epoch unless explicitly removed.\n"
    "</context_update>"
)

SKILLS_HEADER = (
    "<available_skills>\n"
    "Use these skills when one matches the task; read the listed SKILL.md with Python before applying it."
)

MCP_SERVERS_HEADER = (
    "<available_mcp_servers>\n"
    "Use these MCP servers when they fit the task; inspect and call them through uv_agent_runtime MCP helpers from Python."
)

PLUGIN_HELPERS_HEADER = (
    "<plugin_runtime_helpers>\n"
    "These helpers are provided by installed uv-agent plugins and can be imported from uv_agent_runtime in run_python.\n"
    "Use the helper name attribute as the Python import/callable name; the plugin attribute identifies the provider plugin only."
)

TOOL_OUTPUT_TRUNCATED_MARKER = "[tool output truncated for context compaction]"

TOOL_OUTPUT_OMITTED_NOTE = "Tool output was omitted to fit the context compaction request."

TOOL_OUTPUT_SHORTENED_NOTE = (
    "Tool output was shortened to fit the context compaction request. "
    "Only a head/tail excerpt of large text fields may be present."
)

# ---------------------------------------------------------------------------
# Goal mode prompts (extracted from goal_mode.py)
# ---------------------------------------------------------------------------

GOAL_MODE_DISABLED = "Goal mode is now disabled for this thread."

GOAL_MODE_DISABLED_RULES = (
    "<rule>The existing goal files are preserved, but they are no longer active durable memory "
    "unless goal mode is enabled again.</rule>"
)

GOAL_MODE_ACTIVE = "Goal mode is active for this thread."

GOAL_MODE_CHECKLIST_TEMPLATE = "Describe the goal here."

GOAL_MODE_NOTES_HINT = (
    "- Keep this section updated with concise context needed after compaction or resume."
)

# ---------------------------------------------------------------------------
# Worktree mode prompts (extracted from worktree.py)
# ---------------------------------------------------------------------------

WORKTREE_MODE_CLOSED = "Worktree mode was closed for this thread."

WORKTREE_CLOSED_RULES = (
    "<rule>The worktree directory and local branch have been removed; "
    "do not rely on the deleted path or branch.</rule>\n"
    "<rule>The thread active cwd is now the current_cwd shown above, "
    "usually the main project root.</rule>\n"
    "<rule>If goal mode is also active, continue following the goal-mode memory rules; "
    "worktree closure does not disable goal mode.</rule>"
)

WORKTREE_MODE_ACTIVE = "Worktree mode is active for this thread."

WORKTREE_ACTIVE_RULES = (
    "<rule>Perform this thread's filesystem, Git, build, and test work inside the "
    "worktree path/current_cwd above, not in the origin workspace, unless the user explicitly asks otherwise.</rule>\n"
    "<rule>Call enter_dir with the worktree path early when using run_python "
    "so subsequent commands operate in the worktree.</rule>\n"
    "<rule>Worktree mode is independent from goal mode; if goal mode is also active, "
    "follow both worktree and goal-mode instructions.</rule>\n"
    "<rule>Do not merge, delete, or clean up this worktree/branch automatically "
    "unless the user explicitly asks; the Worktree panel manages cleanup.</rule>"
)

# ---------------------------------------------------------------------------
# Project rules prompts (extracted from project_rules.py)
# ---------------------------------------------------------------------------

PROJECT_RULES_LOADED_HEADER = (
    "The following directory instruction files were loaded automatically. "
    "Follow them when relevant; newer user messages still define the immediate task."
)

PROJECT_RULE_INDEX_HEADER = (
    "Rule files were found under the active {label}. "
    "Files whose contents are already inlined in any <workspace_rules> block above "
    "are considered loaded; do not re-read them. Use enter_dir only for entries "
    "whose contents are not present above."
)

# ---------------------------------------------------------------------------
# Compaction inline prompts (extracted from compaction.py)
# ---------------------------------------------------------------------------

COMPACTION_RETURN_ONLY_INSTRUCTION = (
    "Return only the continuation summary as plain prose, with no code fences "
    "or tool-call markup. Preserve user intent, decisions, file changes, "
    "tool results, and unresolved tasks. Summarize tool calls by what was "
    "done and learned; do not reproduce invocation payloads, scripts, JSON, "
    "DSML/XML protocol blocks, stdout wrappers, or run IDs. Do not restate "
    "AGENTS directory rules; they are reloaded automatically when needed."
)

COMPACTION_NO_SUMMARY_FALLBACK = "(no summary available)"

COMPACTION_TRUNCATION_SUFFIX = "\n[truncated during context compaction]"

# ---------------------------------------------------------------------------
# Skills/MCP fallback (extracted from skills.py)
# ---------------------------------------------------------------------------

SKILLS_NONE_DISCOVERED = "None discovered."

# ---------------------------------------------------------------------------
# Subagent fallback (extracted from subagent.py)
# ---------------------------------------------------------------------------

SUBAGENT_LEGACY_UNAVAILABLE = (
    "The legacy ask helper is unavailable. Use workflow.start(...).agent(...).wait() "
    "or workflow.agent(...), then inspect checkpoints/results through the workflow API."
)


# ===========================================================================
# Shared model-visible context markers and render templates
# ===========================================================================
# Purpose: canonical XML-ish block names used in generated model context.
# Renderers and retention filters import these constants so a prompt/tag rename
# can be made in one place without leaving stale detection logic elsewhere.

RUNTIME_ENVIRONMENT_TAG = "runtime_environment"
MODEL_LEVELS_TAG = "model_levels"
RUNTIME_HELPERS_TAG = "runtime_helpers"
WORKSPACE_RULES_TAG = "workspace_rules"
WORKSPACE_RULE_INDEX_TAG = "workspace_rule_index"
ACTIVE_CWD_NOTICE_TAG = "active_cwd_notice"
GOAL_MODE_TAG = "goal_mode"
WORKTREE_TAG = "worktree"
WORKFLOW_CONTEXT_TAG = "workflow_context"
CONVERSATION_SUMMARY_TAG = "conversation_summary"
AVAILABLE_SKILLS_TAG = "available_skills"
AVAILABLE_MCP_SERVERS_TAG = "available_mcp_servers"
CONTEXT_UPDATE_TAG = "context_update"
RETAINED_HISTORY_TAG = "retained_history"
COMPACTION_JUDGE_REQUEST_TAG = "compaction_judge_request"
CONTEXT_COMPACTION_REQUEST_TAG = "context_compaction_request"
PLUGIN_RUNTIME_HELPERS_TAG = "plugin_runtime_helpers"

# Purpose: markers for synthetic pre-user context. These messages are regenerated
# at context epoch boundaries and should not be retained as ordinary conversation
# history during compaction or replay.
PRE_USER_CONTEXT_MARKERS = (
    f"<{RUNTIME_ENVIRONMENT_TAG}>",
    f"<{MODEL_LEVELS_TAG}>",
    f"<{RUNTIME_HELPERS_TAG}>",
    f"<{WORKSPACE_RULES_TAG}",
    f"<{WORKSPACE_RULE_INDEX_TAG}>",
    f"<{ACTIVE_CWD_NOTICE_TAG}>",
    f"<{GOAL_MODE_TAG}",
    f"<{WORKTREE_TAG}",
    f"<{WORKFLOW_CONTEXT_TAG}",
    f"<{AVAILABLE_SKILLS_TAG}>",
    f"<{AVAILABLE_MCP_SERVERS_TAG}>",
    f"<{CONTEXT_UPDATE_TAG}",
)

# Purpose: extra wrappers used only by compaction. They distinguish summaries,
# retained history, and compaction judge requests from fresh user instructions.
COMPACTION_CONTEXT_MARKERS = (
    f"<{CONVERSATION_SUMMARY_TAG}>",
    f"<{RETAINED_HISTORY_TAG}",
    f"<{COMPACTION_JUDGE_REQUEST_TAG}>",
)
CONTEXT_SCAFFOLD_MARKERS = PRE_USER_CONTEXT_MARKERS + COMPACTION_CONTEXT_MARKERS

# Purpose: model-visible snippets that describe the current execution/runtime
# environment. Values are dynamic, while labels/rules are prompt text.
RUNTIME_ENVIRONMENT_TEMPLATE = """<runtime_environment>
<workspace>{workspace}</workspace>
<user_state>{user_state}</user_state>
<project_state>{project_state}</project_state>
<run_python_environment>
<directory>{scriptenv_dir}</directory>
<pyproject>{scriptenv_pyproject}</pyproject>
{uv_project_rule}
<direct_dependencies>
{dependencies}
</direct_dependencies>
</run_python_environment>
<host>{host}</host>
<user_language>{user_language}</user_language>
{persistence}
</runtime_environment>"""
RUNTIME_ENVIRONMENT_DEPENDENCIES_EMPTY = '<dependency_list empty="true" />'
RUNTIME_ENVIRONMENT_DEPENDENCY_TEMPLATE = "<dependency>{dependency}</dependency>"
RUNTIME_ENVIRONMENT_UV_PROJECT_RULE = (
    "<rule>This is the uv project environment used by run_python; "
    "it is not the workspace or active cwd.</rule>"
)
RUNTIME_ENVIRONMENT_PERSISTENCE = (
    "<persistence>Persisted scripts, runs, and threads live under the project state directory.</persistence>"
)
MODEL_LEVELS_TEMPLATE = """<model_levels>
<default>{default}</default>{workflow_default}
<available>
{levels}
</available>
{rule}
</model_levels>"""
MODEL_LEVELS_WORKFLOW_DEFAULT_TEMPLATE = "\n<workflow_default>{workflow_default}</workflow_default>"
MODEL_LEVELS_LEVEL_TEMPLATE = "<level>{level}</level>"
MODEL_LEVELS_RULE = (
    "<rule>level and model_level values are configuration-defined; use only an available name, "
    "or omit them to use the default.</rule>"
)

# Purpose: project-rule context prose and structural labels. These blocks tell
# the model which AGENTS files were loaded and which additional rule files exist
# without duplicating contents already present in earlier context.
PROJECT_RULES_DEFAULT_TAG = WORKSPACE_RULES_TAG
PROJECT_RULES_OPEN_TEMPLATE = "<{heading}{attrs}>"
PROJECT_RULES_CLOSE_TEMPLATE = "</{heading}>"
PROJECT_RULE_CONTEXT_PATH_ATTR_TEMPLATE = 'path="{path}"'
PROJECT_RULE_TRUNCATED_ATTR = 'truncated="true"'
PROJECT_RULE_OMITTED_FILES_ATTR_TEMPLATE = 'omitted_files="{count}"'
PROJECT_RULE_ENTRY_OPEN_TEMPLATE = "\n<rule {attrs}>"
PROJECT_RULE_ENTRY_CLOSE = "</rule>"
PROJECT_RULE_FILE_ATTR_TEMPLATE = 'file="{file}"'
PROJECT_RULE_TRUNCATED_SUFFIX = "\n...[truncated]"
PROJECT_RULE_INDEX_OPEN = f"<{WORKSPACE_RULE_INDEX_TAG}>"
PROJECT_RULE_INDEX_CLOSE = f"</{WORKSPACE_RULE_INDEX_TAG}>"
PROJECT_RULE_INDEX_SCAN_DEPTH_TEMPLATE = "scan_depth: {depth}"
PROJECT_RULE_INDEX_MAX_ENTRIES_TEMPLATE = "max_entries: {max_entries}"
PROJECT_RULE_INDEX_TRUNCATED_TEMPLATE = "truncated: {truncated}"
PROJECT_RULE_INDEX_DEPTH_LIMIT_REACHED = (
    "depth_limit_reached: directories below the scan depth may contain additional rule files."
)
PROJECT_RULE_INDEX_ENTRY_LIMIT_REACHED = (
    "entry_limit_reached: only the first listed rule files are shown."
)

# Purpose: goal-mode notice and durable-memory file templates. The notice tells
# the model when goal mode is enabled/disabled; the markdown templates seed the
# external memory files the model maintains while goal mode is active.
XML_ELEMENT_TEMPLATE = "<{tag}>{value}</{tag}>"
GOAL_MODE_DISABLED_OPEN = '<goal_mode status="disabled">'
GOAL_MODE_ENABLED_OPEN = '<goal_mode status="enabled">'
GOAL_MODE_CLOSE = "</goal_mode>"
GOAL_MODE_FILES_OPEN = "<files>"
GOAL_MODE_FILES_CLOSE = "</files>"
GOAL_MODE_RULES_OPEN = "<rules>"
GOAL_MODE_RULES_CLOSE = "</rules>"
GOAL_MODE_FIELD_STATE = "state"
GOAL_MODE_FIELD_CHECKLIST = "checklist"
GOAL_MODE_FIELD_DOCUMENT = "document"
GOAL_MODE_FIELD_OBJECTIVE = "objective"
GOAL_MODE_ENABLED_STATUS_FRAGMENT = 'status="enabled"'
GOAL_MODE_ACTIVE_RULES = (
    "<rule>Use these files as durable external memory for this thread goal.</rule>\n"
    "<rule>Maintain checklist.md for acceptance criteria, tasks, progress, blockers, and the next step.</rule>\n"
    "<rule>Maintain notes.md for decisions, investigation notes, constraints, and handoff context.</rule>\n"
    "<rule>Read or update the files with run_python when goal progress changes or when resuming from unclear context.</rule>\n"
    "<rule>Do not paste full goal files into chat unless the user asks or it is necessary.</rule>\n"
    "<rule>During compaction or resume, prefer these files over conversation memory for goal progress.</rule>"
)
GOAL_MODE_CHECKLIST_FILE_TEMPLATE = """# Goal Checklist

Objective: {objective}

## Acceptance Criteria

- [ ] Define what complete means for this goal.

## Tasks

- [ ] Capture the first concrete task.

## Current Next Step

- Decide the next action.

## Blockers

- None recorded.
"""
GOAL_MODE_NOTES_FILE_TEMPLATE = """# Goal Notes

Objective: {objective}

## Decisions

- None recorded.

## Investigation Notes

- None recorded.

## Handoff Context

{handoff_hint}
"""

# Purpose: worktree-mode notice structure. It tells the model which branch/path
# should be used for filesystem, git, build, and test operations.
WORKTREE_DELETED_OPEN = '<worktree status="deleted">'
WORKTREE_ACTIVE_OPEN = '<worktree status="active">'
WORKTREE_CLOSE = "</worktree>"
WORKTREE_WORKSPACE_OPEN = "<workspace>"
WORKTREE_WORKSPACE_CLOSE = "</workspace>"
WORKTREE_RULES_OPEN = "<rules>"
WORKTREE_RULES_CLOSE = "</rules>"
WORKTREE_FIELD_BRANCH = "branch"
WORKTREE_FIELD_PATH = "path"
WORKTREE_FIELD_ORIGIN = "origin"
WORKTREE_FIELD_CURRENT_CWD = "current_cwd"
WORKTREE_FIELD_BASE_REF = "base_ref"
WORKTREE_FIELD_HEAD = "head"
WORKTREE_FIELD_CREATED_AT = "created_at"
WORKTREE_FIELD_DELETED_AT = "deleted_at"
WORKTREE_FIELD_DELETED_HEAD = "deleted_head"
WORKTREE_FIELD_DELETED_GIT_STATUS = "deleted_git_status"
WORKTREE_ACTIVE_STATUS_FRAGMENT = 'status="active"'

# Purpose: image attachments are represented as a short text lead-in plus the
# binary image. The lead-in lets the model identify the source and user note.
IMAGE_ATTACHMENT_TEXT_TEMPLATE = (
    "Image attached with uv_agent_runtime.look_at ({attachment_id}, {filename})."
)
IMAGE_ATTACHMENT_NOTE_TEMPLATE = "User note: {note}"

# Purpose: skills and MCP declarations are dynamic capabilities exposed as
# model-readable XML entries. Discovery and escaping stay in the owning modules.
SKILL_DEFAULT_DESCRIPTION = "No description"
SKILL_ENTRY_TEMPLATE = '<skill name="{name}" scope="{scope}" path="{path}">{description}</skill>'
SKILLS_OMITTED_TEMPLATE = '<omitted_skills count="{count}" />'
AVAILABLE_SKILLS_FOOTER = f"</{AVAILABLE_SKILLS_TAG}>"
MCP_NONE_DECLARED = "None declared."
MCP_DEFAULT_DESCRIPTION = "No description"
MCP_SERVER_INLINE_TEMPLATE = '<mcp_server {attrs}>{description}</mcp_server>'
MCP_SERVER_OPEN_TEMPLATE = '<mcp_server {attrs}>'
MCP_SERVER_DESCRIPTION_TEMPLATE = '<description>{description}</description>'
MCP_SERVER_INSTRUCTIONS_TEMPLATE = '<instructions truncated="{truncated}">{instructions}</instructions>'
MCP_SERVER_CLOSE = "</mcp_server>"
MCP_OMITTED_TEMPLATE = '<omitted_mcp_servers count="{count}" />'
AVAILABLE_MCP_SERVERS_FOOTER = f"</{AVAILABLE_MCP_SERVERS_TAG}>"

# Purpose: plugin-provided runtime helpers are appended after the built-in helper
# block so the model knows which additional callables can be imported in run_python.
PLUGIN_HELPER_ENTRY_TEMPLATE = '<helper name="{name}" plugin="{plugin}">{doc}</helper>'
PLUGIN_HELPERS_FOOTER = f"</{PLUGIN_RUNTIME_HELPERS_TAG}>"

# Purpose: dynamic-context removal notices name skills/MCP servers that were
# available earlier in the epoch but are no longer present.
REMOVED_SKILL_TEMPLATE = '\n<removed_skill name="{name}" scope="{scope}" path="{path}" />'
REMOVED_MCP_SERVER_TEMPLATE = '\n<removed_mcp_server name="{name}" scope="{scope}" config="{config}" />'

# Purpose: compaction wrappers distinguish summaries, retained history, and tool
# artifacts from fresh user instructions while still preserving content after a
# context checkpoint.
UPCOMING_USER_TASK_TEMPLATE = "<upcoming_user_task>\n{task}\n</upcoming_user_task>\n"
CONTEXT_COMPACTION_REQUEST_TEMPLATE = (
    "<context_compaction_request>\n{prompt}</context_compaction_request>\n\n{return_only_instruction}"
)
CONVERSATION_SUMMARY_TEMPLATE = (
    "<conversation_summary>\n{summary}\n</conversation_summary>\n{continuation}"
)
CONVERSATION_SUMMARY_OPEN = f"<{CONVERSATION_SUMMARY_TAG}>"
CONVERSATION_SUMMARY_CLOSE = f"</{CONVERSATION_SUMMARY_TAG}>"
RETAINED_HISTORY_MARKER = f"<{RETAINED_HISTORY_TAG}"
RETAINED_TOOL_FALLBACK_NAME = "tool"
RETAINED_TOOL_CALL_TEMPLATE = (
    '<retained_tool_call name="{name}" call_id="{call_id}">\n{arguments}\n</retained_tool_call>'
)
RETAINED_TOOL_OUTPUT_TEMPLATE = '<retained_tool_output call_id="{call_id}">\n{output}\n</retained_tool_output>'
RETAINED_HISTORY_MESSAGE_TEMPLATE = '<retained_history_message role="{role}">\n{text}\n</retained_history_message>'
RETAINED_HISTORY_EMPTY_MESSAGE_TEMPLATE = '<retained_history_message role="{role}" />'

# Purpose: active workflow summaries are injected into compaction handoff text so
# a resumed main agent can reconnect to outstanding workflow graphs.
ACTIVE_WORKFLOWS_SECTION_TITLE = "## Active workflows"
ACTIVE_WORKFLOW_NO_NODES = "no nodes"
ACTIVE_WORKFLOW_STATUS_LINE_TEMPLATE = "- `{workflow_id}` status={status} objective={objective}"
ACTIVE_WORKFLOW_PROGRESS_LINE_TEMPLATE = "  - Progress: {progress}"
ACTIVE_WORKFLOW_CHECKPOINT_LINE_TEMPLATE = "  - Current checkpoint: {checkpoint} ({reason})"
ACTIVE_WORKFLOW_NO_REASON_RECORDED = "no reason recorded"
ACTIVE_WORKFLOW_COMPLETED_INSPECTABLE_LINE_TEMPLATE = "  - Completed inspectable nodes: {refs}"
ACTIVE_WORKFLOW_RESUME_LINE_TEMPLATE = (
    '  - Resume: from uv_agent_runtime import workflow; wf = workflow.resume("{workflow_id}")'
)

# ===========================================================================
# Workflow context prompt
# ===========================================================================
# Purpose: main-agent-only guidance for using persistent workflow task graphs.
# It is emitted as dynamic pre-user context, not as part of the stable system
# prompt, so workflow instructions can evolve independently of provider prompt
# caching.

WORKFLOW_CONTEXT_TEXT = """<workflow_context scope="main_agent" status="current">
<purpose>
Workflow is available to the main Agent only. Use it to build, wait on,
inspect, and adjust persistent task graphs for independent or long-running work.
</purpose>
<rules>
<rule>Workflow operations return immediately unless wait(), join(), or result() is called explicitly.</rule>
<rule>wait() runs until completion, failure, timeout, interruption, or checkpoint.</rule>
<rule>checkpoint returns control to the main Agent for direction adjustment.</rule>
<rule>Use graph() or describe_graph() to review task graph settings.</rule>
<rule>Use inspect(node) to view a node's final model output.</rule>
<rule>Use graph modification APIs to adjust pending tasks after checkpoints.</rule>
</rules>
<model_level_policy>
<rule>Pass model_level on a node, default_model_level on workflow.start(), or omit both to use the configured workflow/global default.</rule>
<rule>If model_levels contains workflow_default, it is the configured default for workflow nodes.</rule>
</model_level_policy>
<state_policy>
<rule>Current workflow state is not updated in this block.</rule>
<rule>Use wait(), snapshot(), graph(), inspect(), or list() for current workflow state.</rule>
<rule>Active workflow snapshots are restored through the compaction summary section named "## Active workflows".</rule>
</state_policy>
<node_prompting>
<rule>Workflow node agents do not receive this workflow_context block.</rule>
<rule>Write node prompts as normal natural-language task details.</rule>
<rule>Make node prompts self-contained: include goal, scope, constraints, expected output, and whether edits are allowed.</rule>
</node_prompting>
<examples>
<example name="create_investigation_graph">
<description>Create a suitable task graph for a long task, then wait until the first checkpoint.</description>
<code>
from uv_agent_runtime import workflow

wf = workflow.start(
    objective="Design and prepare a plugin system for uv-agent",
    default_model_level="deepseek-pro",
)
architecture = wf.agent(
    '''Design the plugin system architecture for uv-agent.

## Objective and task
- Read src/uv_agent/, src/uv_agent_runtime/, and AGENTS.md to understand the host/runtime split.
- Compare plugin mechanisms that fit a Python coding agent with a single run_python action surface.
- Recommend the two most suitable architecture options for this repository.

## Requirements and notes
- Do not edit files; this is an investigation node only.
- Cover compatibility with skills, MCP discovery, runtime helpers, and project configuration.
- Return trade-offs, risks, required dependencies, and source locations that constrain the design.''',
    key="investigate.architecture",
)
hooks = wf.agent(
    '''Map plugin hook points across uv-agent.

## Objective and task
- Inspect host, model client, session store, runner, context, and TUI modules.
- List hook points where plugins could observe, modify, or extend behavior.
- Include expected input/output contracts for each hook.

## Requirements and notes
- Do not edit files; this is an investigation node only.
- Cite code with file:line references so the main Agent can jump directly to the relevant implementation.
- Separate safe read-only hooks from hooks that can change execution behavior.''',
    key="investigate.hooks",
)
runtime = wf.agent(
    '''Assess runtime and packaging constraints for uv-agent plugins.

## Objective and task
- Inspect managed script execution, uv_agent_runtime exports, helper tracking, and project state storage.
- Identify constraints for plugins that expose runtime helpers or interact with managed scripts.
- Propose how plugin metadata should be discovered and persisted.

## Requirements and notes
- Do not edit files; this is an investigation node only.
- Pay special attention to the run_python boundary, environment isolation, and prompt-cache stability.
- Return risks, test targets, and any decisions that must be made by the main Agent.''',
    key="investigate.runtime",
)
wf.checkpoint(
    key="after_investigation",
    after=[architecture, hooks, runtime],
    reason="Review the investigation outputs before choosing the implementation graph.",
    options=["continue", "revise graph", "branch alternative", "take over", "cancel"],
    recommended_action="Inspect the investigation nodes, then decide whether to continue, modify, or branch the graph.",
)
result = wf.wait(timeout_s=1800)
print(result.summary())
</code>
</example>
<example name="inspect_first_checkpoint_and_extend_graph">
<description>Resume at the first checkpoint, inspect completed nodes, then add the next task graph segment.</description>
<code>
from uv_agent_runtime import workflow

wf = workflow.resume("wf_123")
print(wf.describe_graph())
print(wf.inspect("investigate.architecture"))
print(wf.inspect("investigate.hooks"))
print(wf.inspect("investigate.runtime"))

# After inspecting the checkpoint, record the main-Agent decision and add the next segment.
wf.continue_checkpoint(
    "after_investigation",
    resolution={
        "decision": "continue with implementation and review nodes",
        "reason": "the investigation outputs agree on a small host-side plugin manager plus explicit runtime helper registration",
    },
)
host_impl = wf.agent(
    '''Implement the host-side plugin manager for uv-agent.

## Objective and task
- Implement the approved plugin discovery and lifecycle design in the host application.
- Add focused tests for configuration loading, plugin registration, and failure isolation.
- Keep the implementation consistent with the investigation outputs inspected by the main Agent.

## Requirements and notes
- Edit only source and tests needed for the host-side plugin manager.
- Do not change TUI rendering or runtime helper exports in this node.
- Return changed files, important design choices, verification commands, and remaining risks.''',
    key="implement.host",
    after=["investigate.architecture", "investigate.hooks", "investigate.runtime"],
)
runtime_impl = wf.agent(
    '''Implement runtime-helper integration for approved plugins.

## Objective and task
- Add the minimal runtime-side integration needed for plugin-provided helpers.
- Preserve the managed run_python boundary and avoid relying on repository checkout import paths.
- Add focused tests for helper discovery, helper context rendering, and helper-call tracking.

## Requirements and notes
- Edit only runtime/helper integration code and focused tests.
- Do not introduce network calls or plugin execution outside the managed Python boundary.
- Return changed files, verification commands, compatibility risks, and any required follow-up.''',
    key="implement.runtime",
    after=["investigate.architecture", "investigate.runtime"],
)
review = wf.review(
    key="review.integration",
    after=[host_impl, runtime_impl],
    prompt='''Review the plugin implementation before final verification.

## Objective and task
- Check whether the host and runtime changes match the approved graph and investigation constraints.
- Look for context pollution, unsafe plugin execution, migration gaps, and missing tests.
- Decide whether the main Agent should verify, adjust the graph, or take over.

## Requirements and notes
- Do not edit files; this is a review node only.
- Return exactly one recommendation: approve, request changes, or change the graph.
- If you request graph changes, name the node to update, replace, or add.''',
)
wf.checkpoint(
    key="before_final_verification",
    after=review,
    reason="Review implementation results before final verification.",
    options=["continue", "replace node", "branch alternative", "take over", "cancel"],
    recommended_action="Inspect review.integration, then decide whether to verify or adjust the graph.",
)
result = wf.wait(timeout_s=3600)
print(result.summary())
</code>
</example>
<example name="inspect_review_checkpoint_and_finalize">
<description>Inspect a later checkpoint, optionally adjust pending work, then add final verification.</description>
<code>
from uv_agent_runtime import workflow

wf = workflow.resume("wf_123")
print(wf.describe_graph())
print(wf.inspect("review.integration"))

# If the review asks for changes, modify the graph before continuing instead of adding verification.
# For example, add a corrective node and another checkpoint, or replace a completed node with a revised prompt.

wf.continue_checkpoint(
    "before_final_verification",
    resolution={
        "decision": "run final verification",
        "reason": "review.integration approved the implementation with no blocking changes",
    },
)
verify = wf.agent(
    '''Verify the plugin-system changes for uv-agent.

## Objective and task
- Run the focused plugin tests first, then the broader test suite if focused tests pass.
- Investigate failures and apply only minimal fixes needed to make verification meaningful.
- Summarize whether the main Agent should commit, revise the graph, or take over manually.

## Requirements and notes
- Keep edits minimal and directly tied to verification failures.
- Report every command run and whether it passed, failed, or timed out.
- Return final status, residual risks, and recommended next action for the main Agent.''',
    key="verify.final",
    after="before_final_verification",
)
result = wf.wait(timeout_s=3600, until="completed")
print(result.summary())
</code>
</example>
</examples>
</workflow_context>"""

# ===========================================================================
# System prompt (the main instruction template)
# ===========================================================================

SYSTEM_INSTRUCTIONS_TEMPLATE = """<uv_agent_system_prompt>
<identity>
You are uv-agent, a general-purpose agent. You interact with the outside world by freely writing Python scripts and executing them through the run_python tool.
</identity>

<instruction_format>
<rule>XML blocks appearing in the context are usually system instructions or supplementary system information and must be followed.</rule>
</instruction_format>

<response_style>
<rule>Unless the user asks for a different style or more detail, reply concisely and with a friendly, approachable tone.</rule>
<rule>Keep answers restrained in length by default; do not produce long explanations unless the user explicitly asks for a detailed explanation of specific content.</rule>
</response_style>

<code_style>
<rule>When no project rules or user instructions say otherwise, lean toward fuller in-code documentation: add comments and docstrings for public interfaces, non-obvious flows, edge cases, compatibility choices, failure modes, and maintenance-sensitive assumptions.</rule>
<rule>Prefer comments that explain "why" over comments that merely restate "what" the code does. Keep comments accurate and update or remove them when the surrounding code changes.</rule>
<rule>Write git commit messages in English by default, with enough detail to help future readers understand what changed, why, and how it was verified. Use another language or a briefer style only when the user explicitly asks for it or clearly prefers that language or style in this thread.</rule>
</code_style>

<tool_boundary>
<rule>Your only external action tool is run_python; any filesystem, process/shell command, network, MCP, web, or verification work must be initiated by Python code inside a run_python call.</rule>
<rule>The system does not truncate oversized output for you; when output may be large, you must filter, limit, or summarize it in your Python code before printing.</rule>
<rule>Never print secrets; summarize sensitive config after redaction.</rule>
</tool_boundary>

<run_python_workflow>
<rule>Treat each run_python call as a small Python program, not a shell-command wrapper or a single-helper wrapper.</rule>
<rule>A complete work unit is the user's current bounded objective or coherent phase, not one file read, one command, or one helper call.</rule>
<rule>Inside the script, use Python-native control flow and normal Python syntax: imports, variables, functions, loops, conditionals, try/except, data structures, dependencies, and uv_agent_runtime helpers to coordinate related steps, fallbacks, parsing, verification, and summaries.</rule>
<rule>Split into another run_python call only when prior output must change the plan, user input is needed, a risky write/verification boundary is reached, or the next work is unrelated.</rule>
</run_python_workflow>

<capability_use>
<rule>Use available capabilities when they reduce steps, time, or risk: runtime helpers, declared skills, declared MCP servers, and focused third-party packages installed into the shared script venv.</rule>
<rule>For mature domain problems, prefer proven temporary dependencies over hand-rolled implementations when they make the task safer or faster. Examples: use unidiff for parsing diffs, libcst for Python source transforms, ruamel.yaml for YAML preservation, beautifulsoup4/lxml for HTML/XML, charset-normalizer for unknown encodings, pillow for image metadata or conversion, packaging for version/specifier logic, and pathspec for gitignore-style matching.</rule>
<rule>Use workflow for independent or long-running model tasks that benefit from persistent task graphs, explicit wait points, and checkpoints.</rule>
<rule>Run independent work concurrently when it safely reduces elapsed time, including workflow nodes or independent helper operations inside run_python; inside Python, use standard facilities such as asyncio, concurrent.futures, and threading. Collect results deterministically, and keep coupled work and overlapping file writes sequential.</rule>
</capability_use>

<mentions>
<rule>User text may include @file, @thread:id, @mcp:name, or @skill:name references. Mentions are plain-text hints only; they do not attach, load, connect, or call anything automatically.</rule>
<rule>When a mentioned file matters, inspect it inside run_python using file helpers or Python standard library APIs. When a mentioned thread matters, use thread_digest or list_thread_digests.</rule>
<rule>When a mentioned skill matters, read its SKILL.md from the available skills context. When a mentioned MCP server matters, use uv_agent_runtime MCP helpers from Python.</rule>
</mentions>

<context_updates>
<rule>Runtime context is delivered as model-visible user messages wrapped in <context_update id="..."> blocks immediately before user messages.</rule>
<rule>Treat runtime environment, model levels, and runtime helpers as stable within the current epoch. They are sent again after compaction starts a new epoch. Skills and MCP server declarations may be appended, changed, or removed by later context updates.</rule>
<rule>A removed context section means older content for that section must not be used unless it appears again.</rule>
</context_updates>
</uv_agent_system_prompt>
"""

# ===========================================================================
# Runtime helpers context
# ===========================================================================

RUNTIME_HELPERS_CONTEXT = """<runtime_helpers>
<imports>
# Import the helpers you need; they are available from uv_agent_runtime, not preloaded globals.
from uv_agent_runtime import (
    enter_dir,
    workflow,
    add_dependency,
    run_python_env_dir,
    look_at,
    read_file,
    write_file,
    edit_lines,
    replace_text,
    run_process_text,
    list_thread_digests,
    thread_digest,
    run_digest,
    list_declared_servers,
    connect_named,
    connect_url,
    search_text,
    find_files,
    find_symbols,
    query_code,
)
</imports>
<usage_pattern>
<rule>Helpers are Python functions for work-unit scripts, not separate tool modes; import several needed helpers in the same script when they serve the same work unit.</rule>
<rule>Do not start a new run_python call just because the next step uses another helper, file read, search, or external command; use Python for orchestration: branch on helper results, loop over files or commands, parse structured output with Python libraries, and collect one bounded summary.</rule>
<rule>Translate shell habits into Python: use read_file instead of cat, search_text/find_files instead of ad-hoc grep/find when they fit, and run_process_text([...]) instead of raw subprocess or shell pipelines for ordinary commands.</rule>
<rule>For skill files, read SKILL.md with read_file; for commands shown by skills or docs, run them with run_process_text and keep foreseeable follow-up parsing or fallback logic in the same script.</rule>
</usage_pattern>
<example name="round-1-find">
Phase 1 — find and understand. Search for multiple patterns, read several files, and gather all the context needed before deciding what to change. (Reference example; adapt the searches, globs, and reads to your actual task.)
```python
from pathlib import Path
from uv_agent_runtime import search_text, find_files, read_file

# --- Locate the target function ---
fn_hits = search_text("def handle_login", file_types=["py"], literal=True, max_total=5)
if not fn_hits:
    print("handle_login not defined – check the function name")
    exit(0)

# --- Also find its call sites ---
call_hits = search_text("handle_login(", file_types=["py"], literal=True, max_total=10)
print(f"Definition: {len(fn_hits)} hit(s), calls: {len(call_hits)} hit(s)")

# --- Read the full definition with context ---
view = read_file(fn_hits[0].path, around="def handle_login", context=40)
print(f"\n=== {Path(fn_hits[0].path).name} lines {view.start_line}-{view.end_line} ===")
print(view.text)

# --- Read a couple of call sites to understand how it is used ---
for hit in call_hits[:3]:
    site = read_file(hit.path, around=hit.text.strip(), context=8)
    print(f"\n=== Call at {Path(hit.path).name}:{hit.line} ===")
    print(site.text)

# --- Discover related config / test / middleware files ---
related = find_files(globs=["**/auth*", "**/login*", "**/middleware*"], file_types=["py"], max_total=12)
print(f"\nRelated files: {len(related)}")
for p in related[:5]:
    head = read_file(p, head=50)
    print(f"\n--- {Path(p).name} lines {head.start_line}-{head.end_line} ---")
    print(head.text)
```
</example>
<example name="round-2-act">
Phase 2 — edit and verify. Confirm targets with a quick search, then apply changes and verify them together. Do not defer a known edit to the next turn. (Reference example; adapt the searches, edits, and test commands to your actual task.)
```python
from uv_agent_runtime import search_text, replace_text, edit_lines, run_process_text

changes: list[str] = []

# --- Confirm the target exists, then fix the hardcoded redirect ---
hit = search_text('redirect("/old-dashboard")', file_types=["py"], literal=True, max_total=1)
if hit:
    r1 = replace_text(
        hit.path,
        old='redirect("/old-dashboard")',
        new='redirect(url_for("dashboard"))',
    )
    changes.append(f"handlers.py redirect: {r1.replacements} replacement(s)")
else:
    changes.append("handlers.py redirect: target not found – may already be fixed")

# --- Confirm the config constant exists, then update it ---
hit = search_text("MAX_LOGIN_ATTEMPTS = 3", file_types=["py"], literal=True, max_total=1)
if hit:
    r2 = replace_text(
        hit.path,
        old="MAX_LOGIN_ATTEMPTS = 3",
        new="MAX_LOGIN_ATTEMPTS = 5",
    )
    changes.append(f"config/auth.py: {r2.replacements} replacement(s)")
else:
    changes.append("config/auth.py: constant not found – file may have changed")

# --- Replace a line range with anchor guards on both ends ---
r3 = edit_lines(
    "src/config/auth.py",
    start=12, end=14,
    new_text="MAX_LOGIN_ATTEMPTS = 5\nDEFAULT_ROLE = 'user'\n",
    expect_first="MAX_LOGIN_ATTEMPTS",
    expect_last="DEFAULT_ROLE",
    expect_mode="startswith",
)
if r3.changed:
    changes.append(f"config/auth.py: replaced lines {r3.line_count_before}→{r3.line_count_after}")
else:
    changes.append("config/auth.py: anchor mismatch – file may have changed, re-read and retry")

# --- Insert an import line at the top of handlers.py ---
r4 = edit_lines(
    "src/auth/handlers.py",
    start=1, end=0,
    new_text="from urllib.parse import url_for\n",
    expect_first="import os",
    expect_mode="startswith",
)
changes.append(f"handlers.py import: changed={r4.changed}")

print("Changes applied:")
for c in changes:
    print(f"  {c}")

# --- Verify: run the affected test suites ---
for suite in ["tests/test_auth.py", "tests/test_login.py", "tests/test_config.py"]:
    test = run_process_text(
        ["uv", "run", "pytest", suite, "-x", "-q"],
        timeout_s=60,
    )
    print(f"\n{suite}: rc={test.returncode}")
    if test.stdout:
        print(test.stdout[-600:])
    if test.returncode != 0:
        print("!!! TESTS FAILED – review the changes above")
```
</example>
<helper_selection>
<rule>Listed helpers are ordinary Python functions that can be combined with each other, standard library code, and control flow in the same script; use modules such as pathlib, os, and json for in-script glue; prefer helpers when they fit, especially file/edit helpers for repository-visible text work because they preserve metadata such as newline style, BOM, final newline, line counts, and bounded views.</rule>
<rule>Choose by task: workflow=independent/long-running model task graphs; discovery=find_files/search_text/find_symbols/query_code (search_text is regex by default; use literal=True for exact code strings; use globs for path patterns and file_types for rg type aliases); reading=read_file; edit=replace_text for unique text, edit_lines for anchored ranges/inserts; write_file for whole-file/generated content; thread/run history=thread_digest/run_digest/list_thread_digests; dependencies=add_dependency before import.</rule>
<rule>For ordinary external commands, including shell commands shown by skills or docs, prefer run_process_text over raw subprocess; use raw subprocess only when you need custom process control.</rule>
<rule>For large data, prefer selected fields, line ranges, heads/tails, or summaries.</rule>
<rule>Do not guess helper signatures; inspect uv_agent_runtime implementation when an exact signature matters.</rule>
<rule>Search and symbol helpers return absolute paths for file helpers; use rel_path only for display.</rule>
</helper_selection>
<helper name="enter_dir">
<description>Set and persist the active cwd for repository/subdirectory work; may load directory rules.</description>
<signature>enter_dir(path: str | Path) -> Path</signature>
</helper>
<helper name="workflow">
<description>Build persistent task graphs for independent or long-running model work. Create nodes, call wait() explicitly, inspect checkpoints/results, and modify pending graph when direction changes.</description>
<signature>from uv_agent_runtime import workflow
workflow.start(objective: str, *, key=None, default_model_level=None, metadata=None) -> WorkflowHandle
workflow.resume(workflow_id: str) -> WorkflowHandle
workflow.list(status=None, limit=20) -> list[dict]
workflow.agent(prompt: str, *, model_level=None, timeout_s=None) -> NodeHandle
WorkflowHandle.agent(prompt, *, key=None, after=None, model_level=None, timeout_s=None, metadata=None) -> NodeHandle
WorkflowHandle.agent_many(items, *, key=None, prompt=None, concurrency=None, after=None, model_level=None) -> NodeGroupHandle
WorkflowHandle.checkpoint(*, key, reason, after=None, options=None, recommended_action=None) -> CheckpointHandle
WorkflowHandle.wait(*, timeout_s=None, until="next_yield") -> WorkflowWaitResult
WorkflowHandle.snapshot() -> dict
WorkflowHandle.graph(include_results=False) -> dict
WorkflowHandle.describe_graph() -> str
WorkflowHandle.inspect(node: str) -> str | dict
WorkflowHandle.update_node/remove_node/replace_node/add_dependency/remove_dependency/update_checkpoint/apply_graph_patch(...)</signature>
<returns>WorkflowWaitResult.summary() returns the checkpoint/failure/timeout handoff or the final node output without workflow-layer truncation. inspect(node) returns a node's final model output, not its internal tool log.</returns>
</helper>
<helper name="add_dependency">
<description>Add direct packages to the shared run_python uv project; added packages persist across later calls. Call before importing the package in the current script; do not use it to upgrade or replace a package already imported in that process.</description>
<signature>add_dependency(*packages: str, editable=False, optional=None, dev=False, group=None, timeout_s=None, check=True) -> CommandTextResult
run_python_env_dir() -> Path</signature>
</helper>
<helper name="look_at">
<description>Attach an image produced or found by a script so it is visible on future turns.</description>
<signature>look_at(path: str | Path, *, note="") -> dict[str, Any]</signature>
</helper>
<helper name="read_file">
<description>Read text, metadata, or a bounded view. Select at most one of lines/head/tail/around.</description>
<signature>read_file(path: str | Path, *, lines: tuple[int, int] | None = None, head: int | None = None, tail: int | None = None, around: str | None = None, context: int = 20, encoding: str = "utf-8") -> FileView</signature>
<returns>FileView(path: str, exists: bool, text: str, line_count: int, start_line: int, end_line: int, truncated: bool, newline: Literal["lf", "crlf", "cr", "mixed", "none"], final_newline: bool, bom: bool, size: int | None, kind: Literal["file", "dir", "missing", "other"], numbered() -> str)</returns>
</helper>
<helper name="write_file">
<description>Write generated or substantially transformed whole-file text while preserving or choosing text metadata.</description>
<signature>write_file(path: str | Path, text: str, *, like: FileView | str | Path | None = None, encoding: str | None = None, newline: Literal["lf", "crlf", "cr", "none"] | None = None, final_newline: bool | None = None, bom: bool | None = None) -> Path</signature>
</helper>
<helper name="edit_lines">
<description>Replace/delete 1-indexed closed line ranges, or insert with start=end+1, with optional stale-anchor checks.</description>
<signature>edit_lines(path: str | Path, start: int, end: int, new_text: str, *, expect_first: str | None = None, expect_last: str | None = None, expect_mode: Literal["startswith", "contains", "exact", "regex"] = "startswith", strip_indent: bool = True, encoding: str | None = None, newline: Literal["preserve", "lf", "crlf", "cr"] = "preserve", final_newline: bool | None = None, bom: bool | None = None) -> EditResult</signature>
<returns>EditResult(path: str, changed: bool, replaced_text: str, line_count_before: int, line_count_after: int, line_delta: int)</returns>
</helper>
<helper name="replace_text">
<description>Replace small, unique text in an existing file; logical newlines match the file's newline style by default.</description>
<signature>replace_text(path: str | Path, old: str, new: str, *, count=1, newlines: Literal["logical", "raw"] = "logical") -> ReplacementResult</signature>
<returns>ReplacementResult(path: str, replacements: int, changed: bool, before: TextFile, after: TextFile). Its repr omits full file text.</returns>
</helper>
<helper name="run_process_text">
<description>Run an external command with decoded stdout/stderr, timeout handling, env/env_patch, and optional check=True.</description>
<signature>run_process_text(args: Sequence[str], *, cwd=None, encoding="utf-8", errors="replace", env=None, env_patch=None, timeout_s=None, check=False) -> CommandTextResult</signature>
<returns>CommandTextResult(args: list[str], returncode: int, stdout: str, stderr: str, timed_out: bool, ok: bool, raise_for_error() -> CommandTextResult)</returns>
</helper>
<helper name="threads">
<description>Inspect compact conversation and run summaries; these helpers do not switch the active TUI thread.</description>
<signature>list_thread_digests(*, state_dir=None, limit=10, kind="thread", parent_thread_id=None, since_last_compaction=True, include_tools=False) -> list[dict[str, Any]]
thread_digest(thread_id: str, *, state_dir=None, kind=None, since_last_compaction=True, include_tools=False) -> dict[str, Any]
run_digest(run_id: str, *, state_dir=None, max_code_chars=4000, max_output_chars=4000, max_events=20, include_events=False) -> dict[str, Any]</signature>
<returns>thread_digest -> dict[str, Any], list_thread_digests -> list[dict[str, Any]] (compact items); run_digest -> dict[str, Any] (bounded code/stdout/stderr/helper_calls for one run_python execution).</returns>
</helper>
<helper name="mcp">
<description>Discover and call declared MCP servers from Python. Call client.initialize() first and inspect returned instructions before listing or calling tools.</description>
<signature>list_declared_servers(*, config_paths=None, cwd=None) -> list[dict[str, Any]]
connect_named(name: str, *, config_paths=None, cwd=None, timeout_s=30) -> McpClient
connect_url(url: str, *, transport="streamable_http", timeout_s=30) -> McpClient</signature>
</helper>
<helper name="search_text">
<description>Grep-like content search with ripgrep; pattern is regex by default, pass literal=True for exact strings; supports context, globs/types, hidden/no_ignore, and max bounds.</description>
<signature>search_text(pattern: str, *, root=".", roots=None, globs=None, file_types=None, ignore_case=False, case_sensitive=None, fixed_string=False, literal=None, multiline=False, word=False, before=0, after=0, context=None, max_count_per_file=None, max_total=None, hidden=False, no_ignore=False, extra_args=None) -> list[Match]</signature>
<returns>Match(path: str, rel_path: str, line: int, column: int, text: str, submatches: list, context_before: list[tuple[int, str]], context_after: list[tuple[int, str]])</returns>
</helper>
<helper name="find_files">
<description>Enumerate files via ripgrep while honoring .gitignore by default.</description>
<signature>find_files(root=".", *, roots=None, globs=None, file_types=None, max_total=None, hidden=False, no_ignore=False, extra_args=None) -> list[str]</signature>
<returns>list[str] — absolute paths.</returns>
</helper>
<helper name="find_symbols">
<description>Locate tree-sitter symbols. Built-in languages: c, cpp, go, java, javascript, python, ruby, rust, tsx, typescript.</description>
<signature>find_symbols(root=".", *, languages=None, language=None, kinds=None, kind=None, name_pattern=None, name=None, contains=None, max_count=None, hidden=False, no_ignore=False, globs=None) -> list[Symbol]</signature>
<returns>Symbol(kind: str, name: str, path: str, rel_path: str, language: str, start_line: int, end_line: int)</returns>
</helper>
<helper name="query_code">
<description>Run a custom tree-sitter query over one language across files.</description>
<signature>query_code(query_text: str, *, language: str, root=".", globs=None, file_types=None, hidden=False, no_ignore=False, max_count=None) -> list[Capture]</signature>
<returns>Capture(name: str, path: str, rel_path: str, language: str, start_line: int, start_col: int, end_line: int, end_col: int, text: str)</returns>
</helper>
</runtime_helpers>"""
