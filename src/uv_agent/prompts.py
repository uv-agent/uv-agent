"""
Core uv-agent prompt strings.

Stable system prompt, tool definition, runtime context, project rules, and
compaction scaffolding live here. Builtin/domain plugins own their own
model-visible context and helper wording.
"""

from __future__ import annotations

# ===========================================================================
# Tool definition
# ===========================================================================

PYTHON_TOOL = {
    "type": "function",
    "name": "run_python",
    "description": (
        # "不要为了单个命令、单次读取或单个 helper 发起一次 run_python；"
        # "也不要把单次搜索、读取、编辑或验证拆成碎片调用。"
        # "每次 run_python 必须是一个完整工作单元脚本；"
        # "把一次调用视为一个完整的工作单元脚本："
        # "用 Python 原生控制流把服务于同一目标的搜索、读取、计算、编辑、验证和条件回退编排在一起，"
        # "最后只输出一份摘要。"
        # "只有需要用户确认、操作有破坏风险、或结果会改变整体方向时，才拆成下一次调用。"
        "在全新的 Python 进程中运行一个完整、独立的 Python 脚本。"
        "它在该线程的活动 cwd 中运行，并使用项目共享的脚本虚拟环境。"
        "与外部世界交互时，使用 Python 原生控制流和 import，"
        "而不是 shell 风格片段。"
        "优先使用上下文中载入的 helper 函数，对于普通外部命令，尤其优先使用 rt.run。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": (
                    "一个完整、有效的 Python 源码，表示单个独立脚本。"
                    "使用常规 Python 语法，不要使用 shell 风格伪代码；"
                    "包含 import 和必要设置。编写一个小型 Python 程序，"
                    "通过变量、函数、循环、条件、try/except、数据结构、依赖项以及 "
                    "上下文中载入的 helper 函数调用来协调相关步骤。"
                ),
            },
            "timeout_s": {
                "type": "number",
                "description": "执行超时时间（秒）。通常无须设置，可以选择在脚本内控制更细粒度的超时。",
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

COMPACTION_JUDGE_REQUEST = """<agent_compaction_judge_request>
你即将收到一个用户任务。回答前，请先输出一行关于对话状态的 JSON 判断。只返回 JSON 行，不要反引号，不要解释：

{"remaining_calls_bucket":"<0_10|10_30|30_60|60_plus>","history_dependency":"<low|medium|high|exact>"}

remaining_calls_bucket: 这个任务还需要多少次模型调用？
history_dependency: 任务对上面对话原始措辞的依赖程度如何？'low' 表示一般续接，'medium' 表示中等依赖，'high' 表示强依赖具体细节，'exact' 表示每个字都重要（diff、错误消息、配置值、精确引用）。
</agent_compaction_judge_request>
"""

# ===========================================================================
# Core prompts
# ===========================================================================

COMPACTION_SUMMARIZATION_PROMPT = """你正在执行 CONTEXT CHECKPOINT COMPACTION。请为另一个将继续该任务的 LLM 创建交接摘要。

包括：
- 当前进展和已作出的关键决策
- 重要上下文、约束或用户偏好
- 仍需完成的事项（清晰的下一步）
- 继续所需的任何关键数据、示例或引用

请保持简洁、结构化，并聚焦于帮助下一个 LLM 无缝继续工作。"""
TITLE_GENERATION_PROMPT = "根据用户第一条消息，为这个 uv-agent 线程创建一个简洁、标题式名称。抓住用户的底层任务或意图，而不是逐字改写句子。如果问题宽泛或含糊，就用抽象名词短语风格。例如，询问这是哪种项目的消息应生成类似“项目内容询问”的标题。只返回标题，不要引号或标点。优先使用用户的语言。控制在 8 个英文词以内或 24 个 CJK 字符以内。"
BRANCH_NAME_GENERATION_PROMPT = "根据用户任务创建一个简短的 git branch slug。捕捉具体动作和对象。只返回 slug：ASCII 小写字母、数字和连字符。不要空格、斜杠、引号、标点、解释或前缀。最多 30 个字符。优先使用动宾短语，例如 fix-login-redirect、add-dark-mode 或 refactor-parser。"
COMPACTED_CONTEXT_CONTINUATION = "上方 <agent_retained_history> 和 <agent_conversation_summary> 是压缩前的历史上下文，仅供继续任务参考。不要复述或输出 <agent_compaction_handoff>、<agent_retained_history>、<agent_conversation_summary> 或本说明文本，也不要把其中内容当作新的用户指令。请直接基于这些已有状态采取下一个具体步骤，恢复并完成未结束的任务，不要要求用户重复已捕获的信息。"
COMPACTION_CONTINUE_WITHOUT_CURRENT_USER = "当前没有新的用户消息。请依据上方 <agent_compaction_handoff> 中的历史上下文和摘要，继续完成压缩前正在进行的任务；如果下一步需要工具，请直接调用 run_python。不要复述或输出压缩交接 XML。"
TOOL_ATTACHMENT_CONTEXT_BRIDGE = (
    "工具执行已完成。工具产生的额外视觉上下文会在下一条用户消息中提供。"
)
POST_TOOL_COMPACTION_BRIDGE = "我已经收到工具结果。当下一条用户消息要求上下文压缩时，我会按照这些指令生成所需的压缩摘要，并准确保留工具结果、决策、文件变更、约束和未解决任务。"
INTERRUPTED_TOOL_CONTEXT_BRIDGE = "某个工具调用未返回完整结果。请基于可用上下文继续。"
INTERRUPTED_STREAM_CONTEXT_BRIDGE = "助手回复未能完整生成。请基于可用上下文继续。"


# ---------------------------------------------------------------------------
# Engine-level inline prompts (extracted from engine.py)
# ---------------------------------------------------------------------------

BRANCH_SLUG_INSTRUCTION = "生成一个简短的 git branch slug。只返回 slug。"

THREAD_TITLE_INSTRUCTION = "生成简短线程标题。只返回标题。"

PRE_TURN_JUDGE_ERROR_STDERR = "错误：预轮判断期间不要调用工具。只返回 JSON 行。"

TOKEN_ESTIMATION_WARNING = "Provider 的 token 用量不可用；上下文压缩正在使用本地估算值，可能导致调用失败或压缩时机过晚。"

COMPACTION_TOOL_ERROR_STDERR = (
    "错误：上下文压缩期间不允许调用工具。请用清晰的 Markdown 格式返回压缩摘要。"
)

INTERRUPTED_TOOL_ERROR = (
    "工具调用未完成，因为用户中断了本轮。不要假定该工具已成功运行。"
)
GUIDED_INPUT_CONTEXT_BRIDGE = "A newer user message arrived while this task was running. This turn stopped at a safe boundary so the next turn can continue from the newer user message."

ACTIVE_CWD_NOTICE_TEMPLATE = """<agent_active_cwd_notice>
run_python 的活动工作目录现在是 {active_cwd_rel}（线程打开时位于 {initial_cwd_rel}）。相对路径与目录规则均跟随活动工作目录。
</agent_active_cwd_notice>"""

TOOL_OUTPUT_TRUNCATED_MARKER = "[工具输出因上下文压缩被截断]"

TOOL_OUTPUT_OMITTED_NOTE = "为适配上下文压缩请求，工具输出已省略。"

TOOL_OUTPUT_SHORTENED_NOTE = (
    "为适配上下文压缩请求，工具输出已缩短。大型文本字段可能仅保留首尾摘录。"
)

# ---------------------------------------------------------------------------
# Project rules prompts (extracted from project_rules.py)
# ---------------------------------------------------------------------------

PROJECT_RULES_LOADED_HEADER = (
    "以下目录规则文件已自动加载。相关时请遵循；较新的用户消息仍定义当前任务。"
)

PROJECT_RULE_INDEX_HEADER = "在活动 {label} 下发现了规则文件。内容已内联于上方任意 <agent_workspace_rules> 块中的文件，视为已加载；不要重新读取。仅对内容未在上方出现的条目使用 rt.cd。"

# ---------------------------------------------------------------------------
# Compaction inline prompts (extracted from compaction.py)
# ---------------------------------------------------------------------------

COMPACTION_RETURN_ONLY_INSTRUCTION = "只返回交接摘要，使用清晰的 Markdown 格式，不要代码块或工具调用标记。保留用户意图、决策、文件变更、工具结果和未解决任务。从做了什么、学到了什么的角度总结工具调用；不要复现调用 payload、脚本、JSON、DSML/XML 协议块、stdout 包装或 run ID。不要复述 AGENTS 目录规则，系统会自动重新加载必要内容。"

COMPACTION_NO_SUMMARY_FALLBACK = "（无可用摘要）"

COMPACTION_TRUNCATION_SUFFIX = """
[上下文压缩期间被截断]"""

# ===========================================================================
# Shared model-visible context markers and render templates
# ===========================================================================
# Purpose: canonical XML-ish block names used in generated model context.
# Renderers and retention filters import these constants so a prompt/tag rename
# can be made in one place without leaving stale detection logic elsewhere.

RUNTIME_ENVIRONMENT_TAG = "agent_runtime_environment"
MODEL_LEVELS_TAG = "agent_model_levels"
RUNTIME_HELPERS_TAG = "agent_runtime_helpers"
WORKSPACE_RULES_TAG = "agent_workspace_rules"
WORKSPACE_RULE_INDEX_TAG = "agent_workspace_rule_index"
ACTIVE_CWD_NOTICE_TAG = "agent_active_cwd_notice"
COMPACTION_HANDOFF_TAG = "agent_compaction_handoff"
CONVERSATION_SUMMARY_TAG = "agent_conversation_summary"
RETAINED_HISTORY_TAG = "agent_retained_history"
COMPACTION_JUDGE_REQUEST_TAG = "agent_compaction_judge_request"
CONTEXT_COMPACTION_REQUEST_TAG = "agent_context_compaction_request"
COMPACTION_CONTINUATION_TAG = "agent_compaction_continuation"

# Purpose: markers for synthetic pre-user context. These messages are model-visible
# context blocks rather than ordinary user conversation and should not be retained
# as conversation history during compaction.
PRE_USER_CONTEXT_MARKERS = (
    f"<{RUNTIME_ENVIRONMENT_TAG}>",
    f"<{MODEL_LEVELS_TAG}>",
    f"<{RUNTIME_HELPERS_TAG}>",
    f"<{WORKSPACE_RULES_TAG}",
    f"<{WORKSPACE_RULE_INDEX_TAG}>",
    f"<{ACTIVE_CWD_NOTICE_TAG}>",
    "<agent_epoch_context",
    "<agent_epoch_context_update",
    "<agent_turn_context",
)

# Purpose: extra wrappers used only by compaction. They distinguish summaries,
# retained history, and compaction judge requests from fresh user instructions.
COMPACTION_CONTEXT_MARKERS = (
    f"<{COMPACTION_HANDOFF_TAG}>",
    f"<{CONVERSATION_SUMMARY_TAG}>",
    f"<{RETAINED_HISTORY_TAG}",
    f"<{COMPACTION_JUDGE_REQUEST_TAG}>",
    f"<{COMPACTION_CONTINUATION_TAG}>",
)
CONTEXT_SCAFFOLD_MARKERS = PRE_USER_CONTEXT_MARKERS + COMPACTION_CONTEXT_MARKERS

# Purpose: model-visible snippets that describe the current execution/runtime
# environment. Values are dynamic, while labels/rules are prompt text.
RUNTIME_ENVIRONMENT_TEMPLATE = """<agent_runtime_environment>
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
</agent_runtime_environment>"""
RUNTIME_ENVIRONMENT_DEPENDENCIES_EMPTY = '<dependency_list empty="true" />'
RUNTIME_ENVIRONMENT_DEPENDENCY_TEMPLATE = "<dependency>{dependency}</dependency>"
RUNTIME_ENVIRONMENT_UV_PROJECT_RULE = "<rule>这是 run_python 使用的 uv project 环境；它不是 workspace，也不是活动 cwd。</rule>"
RUNTIME_ENVIRONMENT_PERSISTENCE = (
    "<persistence>持久化的脚本、runs 和 threads 位于项目状态目录下。</persistence>"
)
MODEL_LEVELS_TEMPLATE = """<agent_model_levels>
<default>{default}</default>
<available>
{levels}
</available>
{rule}
</agent_model_levels>"""
MODEL_LEVELS_LEVEL_TEMPLATE = "<level>{level}</level>"
MODEL_LEVELS_RULE = "<rule>level 和 model_level 的取值由配置定义；只能使用可用名称，或省略以使用默认值。</rule>"

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
    "depth_limit_reached: 超出扫描深度的目录可能还包含其他规则文件。"
)
PROJECT_RULE_INDEX_ENTRY_LIMIT_REACHED = (
    "entry_limit_reached: 仅显示前几个列出的规则文件。"
)

# Purpose: image attachments are represented as a short text lead-in plus the
# binary image. The lead-in lets the model identify the source and user note.
IMAGE_ATTACHMENT_TEXT_TEMPLATE = (
    "通过 uv_agent_runtime.look_at 附加的图片（{attachment_id}, {filename}）。"
)
IMAGE_ATTACHMENT_NOTE_TEMPLATE = "用户备注：{note}"

# Purpose: compaction wrappers distinguish summaries, retained history, and tool
# artifacts from fresh user instructions while still preserving content after a
# context checkpoint.
UPCOMING_USER_TASK_TEMPLATE = (
    "<agent_upcoming_user_task>\n{task}\n</agent_upcoming_user_task>\n"
)
CONTEXT_COMPACTION_REQUEST_TEMPLATE = "<agent_context_compaction_request>\n{prompt}</agent_context_compaction_request>\n\n{return_only_instruction}"
CONVERSATION_SUMMARY_TEMPLATE = (
    "<agent_conversation_summary>\n{summary}\n</agent_conversation_summary>"
)
COMPACTION_CONTINUATION_TEMPLATE = (
    "<agent_compaction_continuation>\n{continuation}\n</agent_compaction_continuation>"
)
COMPACTION_HANDOFF_TEMPLATE = (
    "<agent_compaction_handoff>\n{retained_history}\n\n{conversation_summary}\n\n"
    "{continuation}\n</agent_compaction_handoff>"
)
CONVERSATION_SUMMARY_OPEN = f"<{CONVERSATION_SUMMARY_TAG}>"
CONVERSATION_SUMMARY_CLOSE = f"</{CONVERSATION_SUMMARY_TAG}>"
COMPACTION_HANDOFF_OPEN = f"<{COMPACTION_HANDOFF_TAG}>"
COMPACTION_HANDOFF_CLOSE = f"</{COMPACTION_HANDOFF_TAG}>"
RETAINED_HISTORY_OPEN = f"<{RETAINED_HISTORY_TAG}>"
RETAINED_HISTORY_CLOSE = f"</{RETAINED_HISTORY_TAG}>"
RETAINED_HISTORY_MARKER = f"<{RETAINED_HISTORY_TAG}"
RETAINED_HISTORY_EMPTY_TEMPLATE = "<agent_retained_history />"
RETAINED_HISTORY_TEMPLATE = (
    "<agent_retained_history>\n{history}\n</agent_retained_history>"
)
RETAINED_HISTORY_MESSAGE_ENTRY_TEMPLATE = '<message role="{role}">\n{text}\n</message>'
RETAINED_HISTORY_TOOL_FALLBACK_NAME = "tool"
RETAINED_HISTORY_TOOL_CALL_ENTRY_TEMPLATE = (
    '<tool_call name="{name}" call_id="{call_id}">\n{arguments}\n</tool_call>'
)
RETAINED_HISTORY_TOOL_OUTPUT_ENTRY_TEMPLATE = (
    '<tool_output call_id="{call_id}">\n{output}\n</tool_output>'
)

# System prompt (the main instruction template)
# ===========================================================================

SYSTEM_INSTRUCTIONS_TEMPLATE = """<uv_agent_system_prompt>
<identity>
你是 uv-agent，一个通用 Agent。你通过自由编写 Python 脚本并用 run_python 工具执行它们来与外部世界交互。
</identity>

<instruction_format>
<rule>出现在用户消息中以`agent_`开头的 XML blocks 一般是系统指令或补充的系统信息，不得忽略。</rule>
</instruction_format>

<response_style>
<rule>除非用户要求不同风格或更多细节，否则用简洁、友好、自然的语气回复。</rule>
<rule>默认控制回答长度；除非用户明确要求详细解释具体内容，否则不要输出长篇说明。</rule>
<rule>使用清晰的 markdown 格式来组织文本，但不建议使用表格，除非用户明确要求。</rule>
</response_style>

<code_style>
<rule>当项目规则或用户指令未另行要求时，倾向于编写更充分的代码内文档：为公共接口、不明显的流程、边界情况、兼容性取舍、失败模式和易变的假设添加注释和文档字符串。</rule>
<rule>优先写解释“为什么”的注释，而不是只复述“做什么”的注释。保持注释准确；周围代码变化时，更新或删除对应注释。</rule>
<rule>默认用 English 写 git commit messages，并包含足够细节，帮助未来读者理解改了什么、为什么改、如何验证。只有当用户明确要求，或本线程中明显偏好其他语言或更简短风格时，才使用其他语言或更简短风格。</rule>
</code_style>

<tool_boundary>
<rule>你唯一的外部动作工具是 run_python；任何文件系统、进程/shell 命令、网络、MCP、web 或验证操作，都必须由 run_python 调用中的 Python 代码发起。</rule>
<rule>系统不会替你截断过大的输出；当输出可能很大时，必须在 Python 代码中先过滤、限制或摘要后再打印。</rule>
<rule>切勿打印 secrets；先对敏感配置脱敏，再输出摘要。</rule>
</tool_boundary>

<run_python_workflow>
<rule>在单次的脚本编写中，编写尽可能多的步骤：搜索、读取、计算、编辑、验证和条件回退都在同一个脚本内用 Python 原生控制流编排。</rule>
<rule>在脚本内使用常规 Python 语法，借助 Python 强大的特性、上下文中载入的 helper 函数以及其他能力，来同时处理多文件、多步骤、可预见的分支或失败。</rule>
<rule>在探索阶段，在单脚本中一次性收集足够信息：并行搜索、查找多个 pattern，同时借助搜索的返回值来读取多个相关文件、运行多条命令获取信息，最后再解析结构化输出，然后返回摘要；同样，执行和验证可以一并完成，无须拆成多轮。</rule>
</run_python_workflow>

<capability_use>
<rule>如果某项能力能减少步骤、节省时间或降低风险，就优先使用，包括：runtime helpers、插件提供的上下文能力，以及安装到共享脚本虚拟环境中、目标明确的第三方包。</rule>
<rule>在成熟领域，临时引入可靠的第三方依赖往往比手写实现更安全、更高效。</rule>
<rule>可以并发运行相互独立的任务，包括插件提供的后台能力或 run_python 内的独立 helper operations；在 Python 中可使用 asyncio、concurrent.futures 和 threading 等标准设施。按确定顺序收集结果，并让相互依赖的任务以及对同一文件的写入保持顺序执行。</rule>
</capability_use>

<mentions>
<rule>用户文本可能包含 @file、@thread:id 或插件提供的其他 mentions。这些 mentions 只是纯文本提示，不会自动加载任何东西。</rule>
<rule>当文件或者 thread 被提及时，在 run_python 中使用对应上下文中载入的 helper 函数读取并检查它。</rule>
</mentions>
</uv_agent_system_prompt>
"""

# ===========================================================================
# Runtime helpers context
# ===========================================================================

RUNTIME_HELPERS_CONTEXT = """<agent_runtime_helpers>
<imports>
import uv_agent_runtime as rt
</imports>

<common_types>
<type name="CollectionResult[T]">
<signature>class CollectionResult[T]:
    ok: bool
    def __bool__(self) -> bool: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[T]: ...
    def __getitem__(self, index: int | slice) -> T | list[T]: ...
    def first(self) -> T | None: ...
    def one(self) -> T: ...  # raises SelectionError unless exactly one item
    def all(self) -> list[T]: ...
    def summary(self) -> str: ...
    def print(self) -> None: ...</signature>
</type>
<type name="FileView">
<signature>FileView(path: str, exists: bool, text: str, line_count: int, start_line: int, end_line: int, truncated: bool, encoding: str, newline: Literal["lf", "crlf", "cr", "mixed", "none"], final_newline: bool, bom: bool, size: int | None, kind: Literal["file", "dir", "missing", "other"])
FileView.header() -> str
FileView.numbered() -> str
FileView.summary() -> str
FileView.print(*, numbered: bool = False) -> None</signature>
</type>
<type name="Resource">
<signature>Resource(uri: str, kind: Literal["text", "bytes", "path"], mime_type: str, metadata: dict[str, Any])
Resource.read(*, lines: tuple[int, int] | None = None, head: int | None = None, tail: int | None = None, around: str | None = None, context: int = 20, encoding: str = "utf-8") -> FileView
Resource.text(*, encoding: str = "utf-8") -> str
Resource.bytes() -> bytes
Resource.path() -> Path
Resource.data: bytes | None</signature>
</type>
<type name="TextFile">
<signature>TextFile(path: str, text: str, encoding: str, newline: Literal["lf", "crlf", "cr", "mixed", "none"], final_newline: bool, bom: bool)</signature>
</type>
<type name="EditResult">
<signature>EditResult(path: str, changed: bool, replaced_text: str, line_count_before: int, line_count_after: int, line_delta: int)</signature>
</type>
<type name="ReplacementResult">
<signature>ReplacementResult(path: str, replacements: int, before: TextFile, after: TextFile)
ReplacementResult.changed: bool</signature>
</type>
<type name="CommandTextResult">
<signature>CommandTextResult(args: list[str], returncode: int, stdout: str, stderr: str, timed_out: bool = False)
CommandTextResult.ok: bool
CommandTextResult.head(lines: int = 20, *, stream: Literal["stdout", "stderr", "both"] = "both") -> str
CommandTextResult.tail(lines: int = 20, *, stream: Literal["stdout", "stderr", "both"] = "both") -> str
CommandTextResult.summary() -> str
CommandTextResult.print(*, lines: int = 20) -> None
CommandTextResult.raise_for_error() -> CommandTextResult</signature>
</type>
<type name="Match">
<signature>Match(path: str, rel_path: str, line: int, column: int, text: str, submatches: list[Submatch], context_before: list[tuple[int, str]], context_after: list[tuple[int, str]])
Match.file() -> File
Match.line_range(*, context: int = 0) -> tuple[int, int]
Match.view(*, context: int = 8) -> FileView</signature>
</type>
<type name="SearchResults">
<signature>class SearchResults(CollectionResult[Match]):
    def grouped(self) -> dict[str, list[Match]]: ...
    def views(self, *, context: int = 8, limit: int | None = None) -> list[FileView]: ...</signature>
</type>
<type name="Symbol">
<signature>Symbol(kind: str, name: str, path: str, rel_path: str, language: str, start_line: int, end_line: int)
Symbol.file() -> File
Symbol.view(*, context: int = 12) -> FileView</signature>
</type>
<type name="SymbolResults">
<signature>class SymbolResults(CollectionResult[Symbol]):
    def views(self, *, context: int = 12, limit: int | None = None) -> list[FileView]: ...</signature>
</type>
<type name="Capture">
<signature>Capture(name: str, path: str, rel_path: str, language: str, start_line: int, start_col: int, end_line: int, end_col: int, text: str)
Capture.file() -> File
Capture.view(*, context: int = 8) -> FileView</signature>
</type>
<type name="CaptureResults">
<signature>class CaptureResults(CollectionResult[Capture]):
    def views(self, *, context: int = 8, limit: int | None = None) -> list[FileView]: ...</signature>
</type>
<type name="FileSet">
<signature>class FileSet(CollectionResult[str]):
    def files(self) -> list[File]: ...
    def views(*, head: int | None = None, tail: int | None = None, lines: tuple[int, int] | None = None, around: str | None = None, context: int = 20, limit: int | None = None) -> list[FileView]: ...
    def search(query: str, *, globs=None, types=None, mode="text", limit=None, ...) -> SearchResults: ...
    def symbols(*, language=None, kind=None, name=None, limit=None, ...) -> SymbolResults: ...</signature>
</type>
<type name="Other result types">
<signature>PathInfo(path: str, exists: bool, kind: Literal["file", "dir", "missing", "other"], size: int | None, cwd: str, base: str | None, is_absolute: bool, is_relative_to_base: bool | None)
TextComparison(equal: bool, kind: Literal["equal", "content", "eol", "final_newline"], message: str, first_difference_line: int | None = None, left: str | None = None, right: str | None = None)
PatchResult(returncode: int, stdout: str, stderr: str, changed_files: list[str])
Snapshot(root: str, files: dict[str, bytes | None])</signature>
</type>
<type name="Thread types">
<signature>ThreadDigest = {thread_id: str, title: str, created_at: str | None, updated_at: str | None, last_text: str, turn_count: int, interrupted_turn_count: int, latest_compaction: ThreadCompactionSummary | None, items: list[ThreadDigestItem]}
ThreadView = {thread_id: str, kind: str, title: str, created_at: str | None, updated_at: str | None, selected_epochs: list[str], epochs: list[ThreadEpoch], turns: list[ThreadTurn], truncated: bool}
ThreadTurn = {id: "turn:&lt;turn_id&gt;", turn_id: str, epoch_id: str, status: str, user_messages: list[ConversationMessage], assistant_messages: list[ConversationMessage], process_refs: list[ProcessRef]}
ProcessRef = {id: "run:&lt;run_id&gt;" | "event:N", kind: str, event_ref: "event:N", event_id: int, turn_id: str, status: str, summary: str, related_ids?: list[str], helper_names?: list[str]}
ThreadDetailResult = {thread_id: str | None, requested_ids: list[str], requested_turn_ids: list[str], details: list[ProcessDetail], missing: list[str], truncated: bool}
ProcessDetail = {id: str, kind: str, status: str, summary: str, thread_id: str | None, turn_id: str | None, event_id: int | None, event_ref: str | None, run_id?: str, returncode?: int | None, timed_out?: bool, interrupted?: bool, code?: BoundedText, stdout?: BoundedText, stderr?: BoundedText, helper_calls?: list[HelperCall], structured_events?: list[dict], events?: list[RunEventDetail], output?: BoundedText, raw_event?: dict}
BoundedText = {text: str, chars: int, truncated: bool, limit: int}</signature>
</type>
</common_types>

<function name="get">
<description>获取本地文件或已注册 URI 资源；非 URI 返回 File，URI 返回 Resource。具体 URI scheme 由插件上下文说明。文件对象负责读取、写入、JSON、替换、行编辑、插入、删除、metadata 和 diff；资源对象负责读取只读文本、二进制或路径资源。</description>
<signature>rt.get(target: str | Path, *, max_bytes: int | None = None) -> File | Resource
File.read(*, lines: tuple[int, int] | None = None, head: int | None = None, tail: int | None = None, around: str | None = None, context: int = 20, encoding: str = "utf-8") -> FileView
File.text(*, encoding: str = "utf-8") -> str
File.json(*, encoding: str = "utf-8") -> Any
File.write(text: str, *, like: FileView | TextFile | str | Path | None = None, encoding: str | None = None, newline: Literal["lf", "crlf", "cr", "none"] | None = None, final_newline: bool | None = None, bom: bool | None = None) -> Path
File.write_text(text: str, *, like: FileView | TextFile | str | Path | None = None, encoding: str | None = None, newline: Literal["lf", "crlf", "cr", "none"] | None = None, final_newline: bool | None = None, bom: bool | None = None) -> Path
File.write_json(value: object, *, encoding: str = "utf-8", indent: int = 2) -> Path
File.replace(old: str, new: str, *, count: int = 1, newlines: Literal["logical", "raw"] = "logical") -> ReplacementResult
File.edit(start: int, end: int, new_text: str, *, expect_first: str | None = None, expect_last: str | None = None, expect_mode: Literal["startswith", "contains", "exact", "regex"] = "startswith", strip_indent: bool = True, encoding: str | None = None, newline: Literal["preserve", "lf", "crlf", "cr"] = "preserve", final_newline: bool | None = None, bom: bool | None = None) -> EditResult
File.insert_after(line: int, text: str, *, expect_line: str | None = None, expect_mode: Literal["startswith", "contains", "exact", "regex"] = "startswith", encoding: str | None = None) -> EditResult
File.insert_before(line: int, text: str, *, expect_line: str | None = None, expect_mode: Literal["startswith", "contains", "exact", "regex"] = "startswith", encoding: str | None = None) -> EditResult
File.delete_lines(start: int, end: int, *, expect_first: str | None = None, expect_last: str | None = None, expect_mode: Literal["startswith", "contains", "exact", "regex"] = "startswith", encoding: str | None = None) -> EditResult
File.info(*, base: str | Path | None = None) -> PathInfo
File.compare(other: str | Path, *, ignore_eol: bool = False, ignore_final_newline: bool = False) -> TextComparison
File.diff(other: str | Path, *, context: int = 3) -> str</signature>
</function>
<function name="search">
<description>FFF 原生索引内容搜索；默认 mode="text" 是精确文本搜索（不用转义正则），需要正则时用 mode="regex"，需要容错行搜索时用 mode="fuzzy"。返回可迭代 SearchResults，hit 可直接 `.view()`。</description>
<signature>rt.search(query: str, *, root: str | Path = ".", roots: str | Path | Sequence[str | Path] | None = None, globs: str | Sequence[str] | None = None, types: str | Sequence[str] | None = None, mode: Literal["text", "regex", "fuzzy"] = "text", ignore_case: bool = False, before: int = 0, after: int = 0, context: int | None = None, limit: int | None = None, refresh: bool = False) -> SearchResults</signature>
</function>
<function name="files">
<description>通过 FFF 原生索引枚举或模糊查找文件，默认遵循 ignore/.gitignore；query 做文件名模糊搜索，globs/types 做路径过滤。返回绝对路径集合，可继续 `.search()` 或 `.views()`。</description>
<signature>rt.files(root: str | Path = ".", *, roots: str | Path | Sequence[str | Path] | None = None, query: str = "", globs: str | Sequence[str] | None = None, types: str | Sequence[str] | None = None, limit: int | None = None, refresh: bool = False) -> FileSet</signature>
</function>
<function name="symbols">
<description>定位 tree-sitter symbols。内置语言：c、cpp、go、java、javascript、python、ruby、rust、tsx、typescript。</description>
<signature>rt.symbols(root: str | Path = ".", *, languages: str | Sequence[str] | None = None, language: str | None = None, kinds: str | Sequence[str] | None = None, kind: str | None = None, name_pattern: str | None = None, name: str | None = None, contains: str | None = None, limit: int | None = None, hidden: bool = False, no_ignore: bool = False, globs: str | Sequence[str] | None = None) -> SymbolResults</signature>
</function>
<function name="query">
<description>在一个 language 上跨文件运行自定义 tree-sitter query。</description>
<signature>rt.query(query_text: str, *, language: str, root: str | Path = ".", globs: str | Sequence[str] | None = None, types: str | Sequence[str] | None = None, hidden: bool = False, no_ignore: bool = False, limit: int | None = None) -> CaptureResults</signature>
</function>
<function name="run">
<description>运行普通外部命令。参数按 subprocess argv 分开传，也可传单个 list/tuple；不要传 shell 字符串；默认 check=False 便于读取失败输出。</description>
<signature>rt.run(*args: str | os.PathLike[str] | Sequence[str | os.PathLike[str]], cwd: str | Path | None = None, encoding: str = "utf-8", errors: str = "replace", env: Mapping[str, str] | None = None, env_patch: Mapping[str, str | None] | None = None, timeout: float | None = None, check: bool = False) -> CommandTextResult</signature>
</function>
<function name="deps">
<description>向共享 run_python uv project 添加 direct packages；添加的 packages 会在后续调用中持续存在。在当前脚本 import 该 package 前调用。</description>
<signature>rt.deps.add(*packages: str, editable: bool = False, optional: str | None = None, dev: bool = False, group: str | None = None, timeout: float | None = None, check: bool = True) -> CommandTextResult
rt.deps.env_dir() -> Path</signature>
</function>
<function name="threads">
<description>检查存储的线程历史；list 只负责找线程，view 默认只展开对话文本，detail 再按 id/turn_id 展开工具和 run_python 明细。这些 helpers 不会切换活动 TUI thread，也不会默认读取当前线程；必须显式传 thread_id。</description>
<signature>rt.threads.list(*, state_dir: str | Path | None = None, limit: int = 10, kind: str = "thread", parent_thread_id: str | None = None, since_last_compaction: bool = True, include_tools: bool = False) -> list[ThreadDigest]
rt.threads.digest(thread_id: str, *, state_dir: str | Path | None = None, kind: str | None = None, since_last_compaction: bool = True, include_tools: bool = False) -> ThreadDigest
rt.threads.view(thread_id: str, *, state_dir: str | Path | None = None, kind: str | None = None, epoch: Literal["latest", "all"] | int | Sequence[int | str] = "latest", max_turns: int | None = None, max_text_chars: int = 12000, max_item_chars: int = 4000, max_process_refs: int = 500) -> ThreadView
rt.threads.detail(*, state_dir: str | Path | None = None, thread_id: str | None = None, ids: str | Sequence[str] | None = None, turn_ids: str | Sequence[str] | None = None, max_code_chars: int = 4000, max_output_chars: int = 4000, max_events: int = 100, include_raw_events: bool = False) -> ThreadDetailResult</signature>
</function>
<function name="events">
<description>向 host 发送结构化事件、进度、结果或图片附件。</description>
<signature>rt.events.emit(kind: str, **payload: Any) -> dict[str, Any]
rt.events.progress(message: str, **payload: Any) -> dict[str, Any]
rt.events.result(**payload: Any) -> dict[str, Any]
rt.events.look_at(target: str | Path | bytes | Resource, *, note: str = "", mime_type: str | None = None, filename: str | None = None) -> dict[str, Any]
rt.look_at(target: str | Path | bytes | Resource, *, note: str = "", mime_type: str | None = None, filename: str | None = None) -> dict[str, Any]</signature>
</function>
<function name="blob">
<description>检查或取得 blob 逃生路径；普通资源读取和图片查看通常不需要直接使用。</description>
<signature>rt.blob.info(blob_id: str) -> dict[str, Any]
rt.blob.path(blob_id: str) -> Path</signature>
</function>
<function name="ui">
<description>向用户界面发送运行中可见的 Markdown 消息；适合脚本等待用户授权、外部确认或需要展示链接时使用。消息会作为 UI 事件发布。</description>
<signature>rt.ui.message(markdown: str) -> dict[str, Any]</signature>
</function>
<function name="misc">
<description>目录、路径、patch/diff/snapshot 和文本工具。</description>
<signature>rt.cd(path: str | Path) -> Path
rt.pwd() -> Path
rt.path(path: str | Path, *, base: str | Path | None = None) -> PathInfo
rt.patch(patch_text: str, *, cwd: str | Path | None = None, format: Literal["auto", "apply_patch", "unified"] = "auto", dry_run: bool = False, check: bool = True) -> PatchResult
rt.apply_patch(patch_text: str, *, cwd: str | Path | None = None, check: bool = True) -> PatchResult
rt.dry_run_patch(patch_text: str, *, cwd: str | Path | None = None, check: bool = True) -> PatchResult
rt.convert_patch(patch_text: str, *, from_format: Literal["apply_patch", "unified"], to_format: Literal["apply_patch", "unified"]) -> str
rt.diff(before: str, after: str, *, path: str | None = None, context: int = 3) -> str
rt.compare(left: str, right: str, *, ignore_eol: bool = False, ignore_final_newline: bool = False) -> TextComparison
rt.normalize(text: str, *, eol: Literal["lf", "crlf", "cr"] | None = "lf", final_newline: bool | None = None) -> str
rt.snapshot(paths: Sequence[str | Path] | None = None, *, root: str | Path = ".") -> Snapshot
rt.restore(snapshot: Snapshot) -> list[str]
rt.transaction(paths: Sequence[str | Path] | None = None, *, root: str | Path = ".") -> Iterator[Snapshot]</signature>
</function>

<instructions>
以上是内置的runtime helpers，现在查看例子，理解函数、类是如何使用的，在脚本中优先使用它们并组合使用来完成任务
一些插件提供的 helper 可能在某些场景下更加便利，注意搭配使用内置和不同插件 helper
</instructions>

<example name="round-1-find">
阶段 1 — 查找并理解。并行搜索多个 pattern、一次读取多个相关文件，在决定修改前收集上下文。（参考示例；根据实际任务调整 searches、globs 和 reads。）
```python
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import uv_agent_runtime as rt

# --- 一次性发起相互独立的定位任务 ---
jobs = {
    "definitions": lambda: rt.search("def render_invoice", types="py", limit=8),
    "call_sites": lambda: rt.search("render_invoice(", types="py", limit=20),
    "tests": lambda: rt.files(roots="tests", query="invoice", types="py", limit=12),
    "symbols": lambda: rt.symbols(language="python", name="InvoiceRenderer", limit=8),
}
with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
    futures = {name: pool.submit(fn) for name, fn in jobs.items()}
    found = {name: future.result() for name, future in futures.items()}

summary = {name: len(result) for name, result in found.items()}
print(f"定位摘要: {summary}")
if not found["definitions"] and not found["symbols"]:
    print("未找到 render_invoice / InvoiceRenderer；请检查命名或扩大搜索范围")
    raise SystemExit(0)

# --- 根据搜索结果继续读取上下文，不把读取推迟到下一轮 ---
for label, result in [("定义", found["definitions"]), ("调用点", found["call_sites"])]:
    for hit in result[:4]:
        view = hit.view(context=12)
        print(f"\n=== {label} {Path(hit.path).as_posix()}:{view.start_line}-{view.end_line} ===")
        print(view.text)

for symbol in found["symbols"][:3]:
    view = symbol.view(context=18)
    print(f"\n=== symbol {symbol.kind} {symbol.name} {Path(symbol.path).as_posix()} ===")
    print(view.text)

for path in found["tests"][:4]:
    view = rt.get(path).read(around="invoice", context=25)
    print(f"\n=== 相关测试 {Path(path).as_posix()}:{view.start_line}-{view.end_line} ===")
    print(view.text)
```
</example>
<example name="round-2-act">
阶段 2 — 编辑并验证。在目标、位置和修改方式已经明确后，先快速搜索确认目标，再一起应用变更并验证；不要把已知编辑推迟到下一轮。（参考示例；根据实际任务调整 searches、edits 和 test commands。）
```python
import uv_agent_runtime as rt

changes: list[str] = []
failed = False

# --- 确认目标仍然存在，然后在事务中完成多文件编辑 ---
with rt.transaction(["src/billing", "tests"], root=".") as snapshot:
    formula_hits = rt.search("return subtotal + tax", roots="src/billing", types="py", limit=5)
    test_hits = rt.search("expected_total = 110", roots="tests", types="py", limit=5)
    helper_hits = rt.search("def apply_discount", roots="src/billing", types="py", limit=3)

    if len(formula_hits) == 1:
        r1 = formula_hits.one().file().replace(
            old="return subtotal + tax",
            new="return apply_discount(subtotal + tax, discount)",
        )
        changes.append(f"total formula: {r1.replacements} 次替换")
    elif formula_hits:
        failed = True
        changes.append(f"total formula: 命中 {len(formula_hits)} 处，请先消歧")
        for view in formula_hits.views(context=6, limit=3):
            print(view.header())
            print(view.text)
    else:
        changes.append("total formula: 未找到目标，可能已经修复")

    if len(test_hits) == 1:
        r2 = test_hits.one().file().replace(
            old="expected_total = 110",
            new="expected_total = 95",
        )
        changes.append(f"expected total: {r2.replacements} 次替换")
    elif test_hits:
        failed = True
        changes.append(f"expected total: 命中 {len(test_hits)} 处，请先消歧")
        for view in test_hits.views(context=6, limit=3):
            print(view.header())
            print(view.text)
    else:
        changes.append("expected total: 未找到旧断言，请重读测试")

    if not failed and not helper_hits:
        calc_symbols = rt.symbols(language="python", name="calculate_total", limit=3)
        if len(calc_symbols) == 1:
            symbol = calc_symbols.one()
            r3 = rt.get(symbol.path).insert_after(
                symbol.end_line,
                "\n\ndef apply_discount(total, discount):\n    return max(0, total - discount)\n",
                expect_line="    return apply_discount(",
            )
            changes.append(f"apply_discount helper: changed={r3.changed}")
        else:
            failed = True
            changes.append(f"calculate_total symbol: 命中 {len(calc_symbols)} 处，请先消歧")
    elif helper_hits:
        changes.append("apply_discount helper: 已存在")

    if not failed:
        for suite in ["tests/test_billing.py", "tests/test_invoice.py"]:
            test = rt.run("uv", "run", "pytest", suite, "-x", "-q", timeout=60)
            print(f"\n{suite}: {test.summary()}")
            if not test.ok:
                failed = True
                print(test.tail(40))
                break

    if failed:
        rt.restore(snapshot)
        changes.append("验证失败：已恢复事务快照，请根据输出重新判断")

print("\n已应用变更:")
for c in changes:
    print(f"  {c}")

if not failed:
    # --- 验证通过后继续收尾：跑格式/更宽测试，不另开一轮 ---
    for cmd in [
        ("uv", "run", "ruff", "check", "src/billing", "tests/test_billing.py"),
        ("uv", "run", "pytest", "tests/test_billing.py", "tests/test_invoice.py", "-q"),
    ]:
        result = rt.run(*cmd, timeout=120)
        print(f"\n{' '.join(cmd)}: {result.summary()}")
        if not result.ok:
            print(result.tail(60))
            break
```
</example>
<example name="anti-pattern-one-helper-per-call">
反例 — 不要把一个清晰的工作单元拆成多次 run_python，每次只调用一个 helper。下面这种“偷懒式串行”会浪费往返、丢失前一次的返回值，也让后续步骤无法在同一个 Python 脚本里根据结果分支。
```python
# **不推荐**：第一轮只搜索，然后停下来等下一轮。
import uv_agent_runtime as rt
hits = rt.search("render_invoice", types="py", limit=10)
print(hits)
---
# **不推荐**：第二轮才读取文件。
import uv_agent_runtime as rt
print(rt.get("src/billing/invoice.py").read(around="render_invoice", context=30).text)
---
# **不推荐**：第三轮才编辑，第四轮才验证，失败后又要重新搜索。
import uv_agent_runtime as rt
print(rt.run("uv", "run", "pytest", "tests/test_billing.py", "-q", timeout=60).tail(40))
---
# **应改为**：在一次 run_python 脚本中导入并组合 rt.search/rt.get/rt.files/rt.symbols/rt.run 等 helpers，
# 用循环、条件、事务快照和数据结构衔接结果；能预见的编辑、验证、失败摘要和回退都在脚本里完成。
```
</example>
</agent_runtime_helpers>"""
