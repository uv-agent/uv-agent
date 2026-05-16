# uv-agent 设计目标

## 项目定位

`uv-agent` 是一个实验性 coding agent。它提供 Textual TUI，但核心能力必须能作为 Python 库被调用。TUI 是交互外壳，agent runtime、会话管理、配置读取、Python runner 和记录系统都应独立于界面存在。

## 核心原则

- Agent 与外界交互的唯一工具形态是提交 Python 脚本给 runner 执行。
- 不给 agent 直接暴露 shell、文件系统、浏览器或网络工具。
- Python 脚本内部允许使用 `subprocess`。
- runner 使用 `uv run` 执行脚本，但实现必须构造 argv 列表，不做 shell 字符串拼接。
- 临时脚本不能依赖当前项目源码、当前 `.venv` 或本地工作区 import 路径。

## Runtime 包

临时脚本需要的快捷能力来自独立可分发包：

- 发布包名：`uv-agent-runtime`
- import 名：`uv_agent_runtime`
- 本仓库源码位置：`src/uv_agent_runtime/`

该包提供给 agent 使用的快捷 API，例如文件读写封装、JSON 读写、结构化输出、JSONL 事件、路径处理、subagent 快捷启动入口、MCP stdio client 等。runner 必须确保最终执行脚本的 PEP 723 inline metadata 中包含 `uv-agent-runtime` 的版本约束。

开发期可以通过配置把 runtime 依赖 spec 指向本地源码，例如 `uv-agent-runtime @ file:///...`；发布后默认使用版本范围，例如 `uv-agent-runtime>=0.1,<0.2`。无论使用哪种 spec，临时脚本都必须通过 inline metadata 获得 runtime，不能依赖当前项目 `.venv` 或源码路径。

## 临时脚本依赖

依赖只写在脚本自己的 PEP 723 inline metadata 中。runner 不提供单独的 `dependencies` 参数。

```python
# /// script
# dependencies = [
#   "requests<3",
#   "rich",
#   "uv-agent-runtime>=0.1,<0.2",
# ]
# ///

from uv_agent_runtime import event
```

如果原始脚本没有声明 `uv-agent-runtime`，runner 在执行前生成规范化后的最终脚本，把 runtime 依赖合并进 inline metadata。JSONL 中同时记录原始脚本和最终脚本。

## Runner API 语义

runner 第一版至少表达这些输入：

```python
PythonRunRequest(
    code: str,
    uv_args: list[str] = [],
    script_args: list[str] = [],
    cwd: Path | None = None,
    timeout_s: float = 60,
)
```

- `code`：agent 提交的 Python 脚本源码。
- `uv_args`：追加到 `uv run` 的异常逃生口，例如 `--refresh-package`、`--upgrade-package`、`--index-url`、`--python`。
- `script_args`：传给临时脚本自身的参数。
- `cwd`：运行目录。
- `timeout_s`：超时。

`uv_args` 只用于棘手情况，必须完整记录到 run JSONL。常规依赖不通过 `uv_args --with` 注入，而是写入 inline metadata。

当前实现入口：

- CLI/TUI：`uv run uv-agent`
- 单轮命令：`uv run uv-agent ask "..."`
- runner：`uv_agent.runner.PythonRunner`
- runtime：`uv_agent_runtime`

`uv_agent_runtime.look_at(path, note="")` 用于把图片追加进后续模型上下文。
宿主会复制图片到用户级 project state 的 attachments 目录，thread JSONL 只保存
路径、hash、mime、备注等元数据，不直接保存大块 base64。MCP 也是 runtime 能力：
提示词只列出可用声明，agent 必须在 Python 脚本中使用 runtime helper 连接和调用。
`run_python` 也可以接收 `script_id` 或 `run_id` 来重跑受管理脚本。

## 受管理脚本与重跑

临时脚本是受管理 artifact，不是运行完即删除的系统临时文件。

默认结构位于用户级目录，避免把历史和脚本 artifact 混入项目工作树：

```text
~/.uv-agent/
  config.json
  projects/
    <project-id>/
      scripts/
        <script_id>/
          script.py
          metadata.json
      runs/
        <run_id>.jsonl
      threads/
        <thread_id>.jsonl
      subthreads/
        <thread_id>.jsonl
```

项目内 `.uv-agent/config.json` 只作为覆盖配置；`.uv-agent/` 被 git 忽略。

每次新脚本执行：

- 分配 `script_id`
- 保存原始脚本和最终脚本
- 分配 `run_id`
- 写入 run JSONL

runner 必须支持重跑：

- 按 `script_id` 重跑某个已保存脚本
- 按 `run_id` 尽量复现某次运行

每个项目最多保留最近使用的 32 组 managed scripts，默认值可由
`runner.max_saved_scripts` 调整。保留策略按最近 run 时间排序，删除旧 script
artifact；run JSONL 仍保留，便于审计。runtime 提供 `saved_scripts(limit=32)` 给
临时脚本读取近期脚本摘要，帮助 agent 用较少轮次选择重跑目标。

重跑默认复用原脚本内容。是否复用原 `uv_args`、`script_args`、cwd 和 timeout，必须由 API 明确表达。若需要严格依赖复现，后续可引入 `uv lock --script` 生成脚本锁文件；未引入锁文件前，不承诺跨时间的依赖版本完全一致。

## 会话模型

参考 Codex 的 thread / turn / item 分层，但实现保持轻量：

- `thread`：一段可切换、可继续的顶层会话。
- `subthread`：由 runtime subagent 启动的子会话，独立保存，关联父
  `thread_id` / `turn_id` / `run_id` / `script_id`，默认不参与父 thread
  的上下文重建。
- `turn`：一次用户输入到 agent 完成响应的过程。
- `item`：消息、推理摘要、runner 调用、runner 输出、压缩摘要等可持久化事件。

TUI 允许多个顶层 thread 同时运行。前台 thread 负责实时渲染；后台 thread
继续写入自己的 JSONL，用户可通过 thread 列表切回查看。并发不改变“唯一外部
工具是 Python runner”的边界。

当前 subagent 快捷入口位于 `uv_agent_runtime.ask`，通过 Python 脚本里的 subprocess 调用 `uv-agent ask`；MCP 快捷入口位于 `uv_agent_runtime.connect_stdio` / `connect_declared`，也必须从 Python runner 内部使用。

subagent 启动参数应优先暴露模型等级而不是具体模型，例如
`ask("review this failure", level="small")` 或 `model_level="large"`。等级到具体模型
的映射由用户级配置决定，主 agent 不需要知道 provider/model 细节。
默认情况下，runtime subagent 使用同一个 project state，但创建 `kind=subagent`
的子线程并保存到 `subthreads/`；如果没有 host state，则退回临时 state。

## JSONL 记录

会话与执行记录先使用 JSONL，不引入数据库。

run JSONL 至少记录：

- run started / finished / failed
- script_id、run_id、thread_id、turn_id
- cwd、timeout、uv argv、uv_args、script_args
- 原始脚本和最终脚本引用或内容
- stdout / stderr 流式事件
- exit code、开始/结束时间、错误信息

thread JSONL 至少记录：

- 用户输入
- agent 消息
- runner 调用和结果引用
- 压缩摘要
- 中断、重跑、恢复等生命周期事件

subthread JSONL 记录格式与 thread JSONL 相同，但 `thread.created` 带
`kind=subagent` 和父级关联字段。父 thread 只记录 runner/subagent 结构化事件
和结果摘要；完整子线程内容需要显式查看或引用，不自动进入父 thread 上下文。

中断不会回滚历史或文件系统副作用。已记录的用户输入、部分 agent 输出和已完成 runner 结果保留；后续上下文重建把中断轮次作为部分上下文摘要处理，不能假设未完成工具调用成功。

## 自动压缩

自动压缩是 thread 级能力：

- 当上下文接近模型限制时触发
- 用户可以手动触发
- 压缩摘要必须作为 item 写入 thread JSONL
- 压缩请求只包含真实对话、模型输出、runner 输出和图片引用；动态 workspace context 不作为普通对话内容参与压缩
- 后续上下文重建从最近一次压缩摘要开始，并追加压缩后的用户/agent 输入与完成的工具结果

压缩摘要至少保留：

- 当前目标
- 已确认的项目约束
- 未完成任务
- 最近关键 runner 结果
- 可重跑脚本引用
- 用户明确偏好或决策

压缩后继续对话时，系统提示词和配置必须按最新配置重新生成。AGENTS 规则、
skills 摘要和 MCP 声明也会重新计算；即使内容与压缩前相同，也需要在压缩后的
下一轮作为新的 workspace context update 追加，避免压缩摘要携带旧配置造成误导。

## 动态 Workspace Context

`AGENTS.md` / `AGENTS.*.md`、skills 摘要和 MCP 声明不写死进稳定 system
prompt。它们按 thread 记录指纹，只在首次出现、内容变化、被移除或压缩后恢复时
追加到模型输入。

- 如果内容没变，不重复追加，保护 prompt cache 和上下文预算。
- 如果某类上下文被移除，追加明确的移除提醒，要求 agent 不再依赖旧的追加内容。
- `item.context_update` 会写入 thread JSONL 用于比较指纹，但不会作为普通历史消息重建，也不会进入压缩输入。
- 长运行 TUI/CLI 每轮前刷新配置；压缩后的下一轮使用最新 system prompt、模型配置和 runner 配置。

## 配置读取

配置读取参考 Codex 的分层思想，但保持 Python 项目简洁。建议优先级：

1. 内置默认值
2. 用户配置
3. 项目配置
4. 会话覆盖

具体文件名、schema、合并规则和敏感信息处理在根目录 `AGENTS.md` 和相关实现文档中维护。OpenAI 凭据不得写入仓库、JSONL 或脚本 artifact。

默认配置放在用户级 `~/.uv-agent/config.json`。项目级覆盖配置可以放在 `.uv-agent/config.json`，该路径被 git 忽略。配置需要支持 provider、model、level、runtime、runner 等块，并在日志和错误展示中对 `api_key` 这类敏感字段脱敏。

配置是本项目自己的 schema，不照搬外部工具配置。provider 只描述共享的 base URL、认证、headers 和 endpoint，model 决定使用 `responses` 还是 `chat_completions` API。

UI 语言通过 `ui.language` 配置，默认 `auto`，并从 `UV_AGENT_LANGUAGE`、locale 等
环境推断。系统提示词必须包含当前用户语言和稳定 host 元数据，例如 OS、shell、
Python 版本、路径分隔符，帮助 agent 生成跨平台但符合当前环境的 Python 脚本。

模型客户端必须支持：

- Responses API
- Chat Completions API
- Anthropic Messages API
- SSE 流式输出
- 非流式调用，用于压缩等后台任务

## TUI 目标

TUI 使用 Textual。

- 默认界面参考 Codex：单一主时间线 + 底部 composer，不把屏幕切成固定三栏。
- 主时间线显示用户输入、assistant 流式输出、Python runner 状态和结构化事件摘要。
- 底部 composer 保持轻量：输入框承担编辑，旁边/下方只显示短状态、模型等级、上下文比例和线程短 id。
- 选中 composer 文本后短暂延迟自动复制，并显示通知；`Ctrl+C` 只用于双击打断运行中的 turn，空闲时双击退出。
- 临时面板用于聚焦查看脚本内容、完整 runner 日志、thread 列表、配置状态等，默认作为全屏 overlay 打开，必须支持滚动、过滤、选择和 Esc 关闭。
- Python runner 结果在主时间线默认折叠 stdout/stderr 详情，只展示状态和结构化
  events；点击事件 cell 可以展开完整 stdout/stderr、events 和 run log 路径。
- 窄终端下优先保留 transcript 和 composer，附属信息降级为短状态文本。

TUI 只消费 core/session/runner 提供的状态和事件，不把业务规则写死在 UI 层。

## 错误与事件呈现

参考 Codex、Gemini CLI、Qwen Code、opencode 的共同模式：

- 错误在 core 层归一成可读标题、摘要、hint 和可选详情；TUI 显示短错误卡片，详细 provider/run 输出放面板或日志。
- 事件作为 transcript timeline 的紧凑条目呈现，而不是塞满底部状态栏。
- thinking/reasoning、tool started、tool result、compaction、image attachment 都应有轻量可扫读的历史项。
- runtime `emit_event` 产生的结构化事件应在 tool result 内呈现。`progress`、
  `result`、`look_at`、`subagent.started`、`subagent.completed` 等事件有紧凑
  展示；未知事件显示为 JSON 摘要。
- 非零退出、超时、截断输出需要在 tool cell 上直接显色；完整 stdout/stderr 可通过展开 tool cell 或 `/runs` 查看。
