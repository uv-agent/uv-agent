# Core + Builtin Plugins 改造计划

> 目标：把非核心能力从 Engine/TUI 中拆出为 builtin plugins，让第三方插件使用同一套一等公民接口。本文件记录已确认的设计决策；当前不要求兼容旧插件/旧内部 API。

## 0. 本轮已确认约束

- **不做旧插件 API 兼容层**：可以直接替换 `pluggy` hook、`TurnContextBlock(text=...)`、旧 `plugins.disabled/config` 等内部 API。
- **builtin 插件位置固定**：内置插件放在 `src/uv_agent/builtin/`，由 `src/uv_agent/builtin/__init__.py` 声明 builtin 插件列表和加载顺序。
- **第三方插件仍需自发现**：third-party 插件继续通过 entry points 等安装后自发现机制加载。
- **core 前缀稳定**：stable system prompt 和 `run_python` tool 定义不随插件启停变化；插件内容进入动态 context，并在同一个 epoch 内保持已落盘前缀序列不被改写。
- **外部动作边界不变**：模型仍只有 `run_python`；插件通过 runtime helper RPC、host registries、storage/action/command/UI contribution 暴露能力。
- **暂不保留 Agent View**：`/agents`、`/bg` 本轮隐藏或移除；后续如果需要，由插件重新提供类似能力。
- **旧 workflow/scheduler 专属配置迁移**：旧 `runtime.workflow_default_level`、顶层 `scheduler` 等专属配置迁移到对应 builtin plugin config，不做兼容。

## 1. Core 边界

Core 保留：

- 线程、turn、历史重建、compaction、模型调用与 `run_python` tool loop orchestration。
- 稳定的 context 基础机制：core context、插件 epoch/turn context 落盘、批量渲染、compaction 后恢复、内部 context message 分组。
- 插件宿主能力：发现/加载、manifest、生命周期、配置、存储、runtime helper registry、command registry、UI contribution registry、action registry、事件。
- 基础 TUI 框架：transcript、composer、command palette 容器、status line 容器、通用 panel/list/table/todo/progress renderer。
- Core runtime helpers：文件/搜索/patch/进程/cwd/deps/events/look_at/thread history 等基础工作能力。
- Core context：workspace rules、runtime environment、core runtime helpers。**不再向模型发送 model levels 列表。**

从 core 迁出为 builtin plugins：

- Goal mode。
- Skills discovery/context/picker。
- MCP discovery/context/runtime namespace。
- Workflow context/runtime namespace/status/UI。
- Scheduler runtime namespace/status/UI，并通过 action registry 触发动作。
- Worktree notice 或后续类似状态提示。
- 其他非核心状态栏 badge、提示块、辅助面板。

## 2. 插件 API 与加载

### 2.1 Manifest + setup API

新插件 API 不再以 `pluggy` hook 为主要接口。插件返回 manifest，并在 setup 阶段显式注册能力：

```python
def plugin() -> Plugin:
    return Plugin(
        manifest=PluginManifest(
            id="builtin.goal",
            version="0.1.0",
            display_name="Goal Mode",
            description="Persistent goal/task tracking for one thread.",
            builtin=True,
            default_enabled=True,
            priority=100,
            dependencies=[],
            optional_dependencies=[],
            capabilities=[
                "epoch_context",
                "turn_context",
                "commands",
                "ui",
                "storage",
                "runtime_helpers",
                "actions",
            ],
            config_schema={...},
            storage_schema={...},
        ),
        setup=setup,
    )


def setup(ctx: PluginContext) -> None:
    ctx.commands.register(...)
    ctx.runtime.register_namespace(...)
    ctx.actions.register(...)
    ctx.ui.status.register(...)
```

### 2.2 Builtin 与 third-party 加载顺序

- builtin 插件由 `src/uv_agent/builtin/__init__.py` 返回固定列表和顺序。
- third-party 插件通过 entry points 等机制自发现，并按 plugin id 稳定排序。
- builtin 和 third-party 都进入同一个 `PluginManager`，使用同一套 manifest/setup/config/storage/context/command/UI/action/runtime helper API。
- 插件依赖只通过 manifest 与公开 registry 表达；不鼓励跨插件直接 import 对方实现。

### 2.3 依赖与错误处理

- required dependency 缺失/禁用：插件不启动。
- optional dependency 缺失：插件可启动，但相关能力降级并记录 warning。
- 硬错误禁用插件：manifest 无效、id 冲突、config schema 硬失败、command/helper/action namespace 冲突、setup 抛异常。
- 软错误保留插件但标记 warning：外部资源暂不可用、optional dependency 缺失、非核心贡献注册失败等。
- 插件 warning 不自动污染模型上下文；只有插件主动 enqueue turn context 时才告知模型。

## 3. 配置系统

### 3.1 插件配置结构

`plugins` 顶层是 plugin id map，不再使用旧 `plugins.disabled` / `plugins.config`：

```json
{
  "plugins": {
    "builtin.goal": {
      "enabled": true,
      "config": {}
    },
    "builtin.workflow": {
      "enabled": true,
      "config": {
        "default_level": "fast",
        "node_timeout_s": 3600,
        "default_concurrency": 4
      }
    }
  }
}
```

未来插件全局设置放在 `plugin_settings`，避免和 plugin id 冲突：

```json
{
  "plugin_settings": {
    "auto_discover": true,
    "plugin_dirs": []
  }
}
```

### 3.2 Merge 规则

插件配置合并顺序：

1. manifest/schema defaults
2. user config: `~/.uv-agent/config.json`
3. project config: `.uv-agent/config.json`

`config` 做 deep merge；`enabled` 是单独 bool override，不 deep merge。项目配置只覆盖写到的字段，不抹掉用户配置中的其他插件字段。

### 3.3 Schema

插件 config schema 使用 JSON Schema 子集，并支持少量扩展：

- `format: "level_ref"`：字段值引用现有 `levels` 名称。
- `format: "secret"` 或 `x-redact: true`：脱敏。
- `x-ui`：UI 编辑提示，如 select/textarea。

不新增 `model_roles`。`levels` 仍是唯一模型预设命名空间；插件需要模型时，在自己的 `config` 中声明 level 字段，host/plugin 按该字段解析。模型可见上下文不再暴露可用 levels 列表。

配置错误分级处理：有安全默认值时可回退并 warning；无安全默认值的 schema 错误禁用该插件。

## 4. Context 机制

### 4.1 两类主动 context

Context 不再使用 provider 轮询、不做 fingerprint、不自动 diff。插件主动声明上下文变化。

1. **Epoch context**
   - 插件主动 `publish/update/remove`。
   - 每个插件每个 epoch 只允许首次 `publish` 一个 epoch document。
   - 后续只能 `update/remove`。
   - 插件自己决定首次发送时机和 update delta 内容。
   - Core 只负责渲染、落盘、合并当前 state、批量发送和 compaction 后恢复。

2. **Turn context**
   - 插件主动 `enqueue` 一次性通知/提示。
   - 下一次模型请求前发送，发送后默认消费。
   - 不存在“每 turn 自动调用 provider”。

### 4.2 Epoch 边界

- epoch = 一个 thread 中两次 compaction 之间的上下文周期。
- 新 thread 开始进入第一个 epoch。
- compaction 完成后进入新 epoch。
- 新 epoch 第一次模型请求前，core 根据已保存的“当前 epoch documents”自动重发 full epoch context；不要求插件重新 publish。

### 4.3 Turn context 压缩后重发

Turn context 默认一次性消费。插件可标记某条 turn context 在 compaction 后重发一次最后内容：

```python
ctx.context.turn.enqueue(
    thread_id=thread_id,
    tag="goal_mode",
    attrs={"status": "enabled"},
    body={...},
    replay_after_compaction=True,
    replay_key="goal_mode_state",
)
```

规则：

- `replay_after_compaction=False` 为默认。
- 同一 `plugin_id + replay_key` 只保留最后一条。
- compaction 后在新 epoch 下一次模型请求前重发一次，然后清除 replay pending 状态。
- 插件可通过 `clear_replay(thread_id, replay_key=...)` 清理。

### 4.4 Message 分组规则

内部 agent context 与外部用户输入必须分开处理。

同一次模型请求前，内部 XML context 按阶段合并为最多几个 `role=user` items：

1. **epoch full context item**：core context + 插件首次/full epoch documents，多个 XML 块用空行隔开。
2. **epoch update item**：一个 `<agent_epoch_context_update>...</agent_epoch_context_update>` 块。
3. **turn context item**：插件主动 enqueue 的一次性 turn notices，多个 XML 块用空行隔开。
4. **外部 user items**：每个用户输入独立一个 `role=user` item，不合并。

外部用户输入不与内部 context 合并，也不再用 `---` 拼接。

### 4.5 Epoch publish/update/remove 渲染

首次/full epoch context 不加统一外层包裹，直接发送语义块：

```xml
<agent_runtime_environment>
...
</agent_runtime_environment>

<agent_available_skills>
...
</agent_available_skills>
```

后续多个 update/remove 在下一次模型请求前合并为一个批量 envelope：

```xml
<agent_epoch_context_update>
<agent_available_skills operation="update">
<skill>
<name>wenyan</name>
<path>...</path>
<description>...</description>
</skill>
</agent_available_skills>

<agent_available_mcp_servers operation="remove">
<reason>MCP config removed</reason>
</agent_available_mcp_servers>
</agent_epoch_context_update>
```

要求：

- 不使用 `status="current"`。
- 不使用无意义属性，例如 `epoch="..."`、`plugin="..."`、`fingerprint="..."`。
- `operation="update|remove|publish"` 这类属性可以保留，因为它有模型语义。
- update body 是插件提交的 delta；core 不计算 diff。
- core 内部可落盘 plugin id、document id、event id、当前 full state 等 metadata，但默认不进入模型可见 XML。

## 5. XML 渲染规则

- 插件只提交结构化 dict/list/text，不允许默认手写 XML。
- Core 统一渲染 XML、转义、校验 tag、添加顶层 `agent_` 前缀。
- 所有 agent 生成、模型可见的顶层 XML tag 自动加 `agent_` 前缀。
- 插件声明 `tag="goal_mode"`，core 渲染为 `<agent_goal_mode>`。
- **只有顶层 contribution tag 加 `agent_`；内部字段不加前缀。**
- 用户原文、文件内容、工具输出中的 XML 不自动改写。
- attrs 默认只来自 contribution 显式声明；core 不自动添加 `plugin/id/epoch/fingerprint` 等内部属性。
- dict key 默认保持插入顺序；attrs key 稳定排序。
- list 渲染为重复 `<item>`。
- `None` 默认省略；bool 渲染为 `true/false`。
- tag name 只允许安全 XML 名称；非法 tag 默认报错，尽早暴露插件问题。

示例：

```python
dict_to_xml("goal_mode", {
    "objective": "重构插件系统",
    "files": {"state": "..."},
    "rules": ["保持任务列表更新", "关键进展后更新 notes"],
})
```

渲染为：

```xml
<agent_goal_mode>
<objective>重构插件系统</objective>
<files>
<state>...</state>
</files>
<rules>
<item>保持任务列表更新</item>
<item>关键进展后更新 notes</item>
</rules>
</agent_goal_mode>
```

## 6. Runtime helpers 划分

### 6.1 Core helpers

Core 仅保留 run_python 基础工作能力：

- 文件/搜索/语义查询：`rt.file`、`rt.files`、`rt.search`、`rt.symbols`、`rt.query`
- patch/diff/snapshot：`rt.patch`、`rt.apply_patch`、`rt.diff`、`rt.compare`、`rt.snapshot`、`rt.restore`、`rt.transaction`
- 进程/依赖/cwd/path：`rt.run`、`rt.deps`、`rt.cd`、`rt.pwd`、`rt.path`
- 事件/图片附件：`rt.events`、`rt.look_at`
- 线程历史检查：`rt.threads`

从 core helper context 移除：`rt.goal`、`rt.mcp`、`rt.workflow`、`rt.scheduler` 等功能域 helper。

### 6.2 插件 namespace helpers

插件 runtime helper 第一阶段支持 **一级 namespace + 函数**：

```python
rt.goal.state()
rt.goal.add_task(...)
rt.mcp.connect(...)
rt.workflow.start(...)
rt.scheduler.create(...)
```

不支持多级嵌套：

```python
rt.goal.tasks.add(...)  # 本轮不做
```

规则：

- 插件通过 `ctx.runtime.register_namespace("goal", helpers=[...])` 注册 namespace。
- namespace 全局唯一；core/builtin reserved namespace 优先，third-party 冲突则后加载者失败。
- helper 名在 namespace 内唯一。
- runtime 侧通过 host RPC/proxy 暴露，不允许插件把 Python module 直接注入 managed script import path。
- 插件 helper 文档由对应插件 epoch context 自包含说明。

## 7. Command registry

插件 command handler 返回结构化 actions，不直接操作 TUI App 或 Engine 内部对象。

```python
CommandResult(actions=[
    TranscriptAction(kind="event", text="goal mode enabled"),
    SetPluginStateAction(...),
    SubmitTurnAction(text="...", conflict="queue"),
    OpenPickerAction(source="builtin.skills"),
    SetComposerAction(text="@skill:grilling "),
    OpenPanelAction(panel_id="builtin.todo.current"),
])
```

规则：

- 插件可注册顶层 slash command，如 `/goal`、`/skills`、`/mcp`、`/todo`。
- command id 使用命名空间，如 `builtin.goal.goal`；slash name 是 UI 暴露名。
- core reserved commands 不可覆盖。
- builtin 命令优先于 third-party。
- third-party 冲突时报错或禁用后加载者。
- command palette 从 registry 读取，不再硬编码 Goal/Skills/MCP。

Core reserved commands 保留：

- `/help`
- `/quit`
- `/clear` / `/new`
- `/cancel`
- `/status`
- `/threads`
- `/show`
- `/image`
- `/level` / `/model`
- `/title`

迁出/移除：

- `/goal` → `builtin.goal`
- `/skills` → `builtin.skills`
- `/mcp` → `builtin.mcp`
- `/agents`、`/bg` → 本轮隐藏或删除，后续由插件自行支持。

`/status` 保留为 core 命令，但内容聚合 core + plugin status contributions；底部 status line 也消费同一套结构化 status contributions。

## 8. UI contribution

第一阶段只开放 UI-neutral 结构化数据，不允许插件直接渲染 ANSI/Textual widget，也不允许直接修改 `Tui2State`。

支持：

```python
StatusItem(id="goal", text="Goal", detail="实现插件重构", priority=100, style="accent")

Panel(id="todo.current", title="Current TODO", kind="todo", data={...})

PickerSource(id="skills", trigger="@skill", query=...)

TranscriptEvent(kind="plugin.event", title="Goal updated", body={...}, severity="info")
```

Panel 第一阶段支持：

- `text`
- `list`
- `table`
- `todo`
- `progress`

不支持插件自定义按键/复杂交互流程；只支持通用结构化渲染和简单 actions。后续 TUI、Web UI 或其他 UI 插件都可以消费同一份 UI contribution。

## 9. Action registry

Action registry 是通用 core plugin capability，不是 scheduler 私有。

```python
ctx.actions.register(
    id="workflow.prompt",
    description="Start a workflow from a prompt.",
    schema={...},
    handler=start_workflow_from_prompt,
)
```

规则：

- action id 全局唯一，推荐命名空间：`workflow.prompt`、`helper.call`、`goal.remind`。
- handler 在 host 侧执行，返回 JSON-serializable result。
- schema 校验 payload。
- action 可声明是否允许 scheduler 调用、command 调用、是否需要 thread_id、是否需要 daemon。
- action registry 不自动进入模型上下文；插件可在自己的 context 中说明如何使用。

Scheduler 到期只调用 action registry，不硬编码 workflow/prompt/helper 类型。

## 10. Storage API

第一阶段提供 core-managed SQLite 存储，不让多数插件直接管理裸 SQLite。

### 10.1 API

```python
ctx.storage.global_kv()
ctx.storage.project_kv()
ctx.storage.thread_kv(thread_id)

ctx.storage.global_collection("todos")
ctx.storage.project_collection("todos")
ctx.storage.thread_collection(thread_id, "todos")
```

KV：

```python
kv.get("key")
kv.set("key", value)
kv.delete("key")
kv.list_prefix("prefix")
kv.update_json("key", updater)
```

Collection：

```python
collection.put(id, document)
collection.get(id)
collection.delete(id)
collection.list(limit=100, cursor=None)
collection.query_index("status", "open")
```

### 10.2 底层原则

- 默认 JSON/document 存储，简单易用。
- collection 支持 manifest 声明简单 indexes，避免高频列表/查询扫全量 JSON。
- 插件 namespace 强隔离。
- builtin plugins 也必须使用该 API，避免私有文件散落。
- 第一阶段不开放插件自定义 SQL migration；后续如确有需要再扩展。
- global scope 使用用户级 SQLite；project/thread scope 复用项目 state SQLite 或由 core 统一管理。
- core 建统一表（命名可在实现时细化）：`plugin_kv`、`plugin_documents`、`plugin_document_indexes`。
- 常用访问路径要有组合索引，例如 plugin/scope/key、plugin/scope/collection/document、plugin/scope/collection/index_name/index_value。

## 11. Builtin plugins 迁移设计

### 11.1 builtin.goal

Goal mode 改为 storage + helper 驱动，不再以 checklist/notes 文件作为主要机制。

启用 Goal mode 时：

- 插件写入 thread-scoped plugin storage。
- 插件发布 epoch context，指导模型使用 `rt.goal.*` helper 管理目标。
- 插件可 enqueue replayable turn context，在 compaction 后重发一次当前 Goal 状态。
- 插件通过 UI contribution 渲染当前阶段/TODO list/status。

`rt.goal.*` 最小 API：

```python
rt.goal.state() -> dict
rt.goal.set_objective(text: str) -> dict

rt.goal.list_tasks(status: str | None = None) -> list[dict]
rt.goal.add_task(text: str, *, status: str = "todo", priority: int | None = None) -> dict
rt.goal.update_task(task_id: str, **changes) -> dict
rt.goal.delete_task(task_id: str) -> dict

rt.goal.get_phase() -> dict
rt.goal.set_phase(name: str, *, summary: str = "") -> dict

rt.goal.get_notes() -> str
rt.goal.set_notes(text: str) -> dict
rt.goal.append_note(text: str) -> dict
```

建议 storage：

- thread KV：`enabled`、`objective`、`phase`、`notes`
- thread collection `tasks`：`id`、`text`、`status`、`priority`、`created_at`、`updated_at`

Goal epoch context 自包含说明：

- 当前 Goal mode 已启用。
- 使用 `rt.goal.*` 管理目标、任务、阶段、笔记。
- 关键进展后更新 task/phase/notes。
- 不要直接操作内部 plugin storage。

### 11.2 builtin.skills

Skills 插件保持简单：

- 发布 epoch context：列出已有 skills 的 name/scope/path/description。
- 提供 picker/mention UI。
- 不提供 `rt.skills.*` helper。
- 不再由 core 硬编码 `discover_skills` context。

`agent_available_skills` 必须自包含使用说明，例如：

```xml
<agent_available_skills>
<instructions>遇到适合任务的 skill，先用 run_python 读取对应 path 的 SKILL.md。用户文本中的 @skill:name 只是纯文本提示，不会自动加载。</instructions>
<skill>
<name>grilling</name>
<scope>user</scope>
<path>C:\Users\...\SKILL.md</path>
<description>Interview the user relentlessly...</description>
</skill>
</agent_available_skills>
```

### 11.3 builtin.mcp

MCP 插件提供：

- 自包含 epoch context：server name/scope/config path/description/instructions preview。
- `rt.mcp.*` runtime namespace。
- picker/mention UI。

`agent_available_mcp_servers` 内说明：

- `@mcp:name` 只是文本提示。
- 需要在 `run_python` 中使用 `rt.mcp.connect(name)` 或相关 helper。
- 连接后先 `initialize()`，检查 instructions，再 `list_tools()` / `call_tool()`。

### 11.4 builtin.workflow

Workflow 插件提供：

- `rt.workflow.*` runtime namespace。
- 自包含 workflow epoch context。
- workflow status/UI contribution。
- workflow 相关命令（如需要）。
- 注册 action `workflow.prompt`，供 scheduler 或其他插件触发 prompt workflow。

Workflow 配置迁移到 plugin config，例如：

```json
{
  "plugins": {
    "builtin.workflow": {
      "config": {
        "default_level": "fast",
        "node_timeout_s": 3600,
        "default_concurrency": 4
      }
    }
  }
}
```

### 11.5 builtin.scheduler

Scheduler 插件提供：

- `rt.scheduler.*` runtime namespace。
- schedule 管理 command/status/UI。
- daemon 定时触发、misfire/overlap、run history retention。
- 到期后调用 action registry，不直接依赖 workflow。

Scheduler config 迁移到 plugin config：

```json
{
  "plugins": {
    "builtin.scheduler": {
      "config": {
        "max_concurrent_jobs": 8,
        "run_history_retention_days": 7,
        "default_misfire_policy": "skip",
        "default_overlap_policy": "skip"
      }
    }
  }
}
```

如果 `builtin.workflow` 禁用，scheduler 仍可运行其他 action；但 `workflow.prompt` action 不可创建/运行。

## 12. Prompt 与动态 context 瘦身

Stable system prompt 只保留：

- 总体行为规则。
- core 能力说明。
- run_python 边界和脚本编排规则。
- core helpers 使用方式。
- core mentions，例如 `@file`、`@thread:id`。
- agent 生成 XML 的通用说明：顶层 `agent_` 前缀代表 agent/system/context 信息；用户 XML 原文不自动视为 agent context。

迁出到插件 context：

- skills 使用说明和 `@skill` 规则。
- MCP 使用说明和 `@mcp` 规则。
- workflow 使用说明。
- scheduler 使用说明。
- goal mode 使用说明。
- 插件 runtime helper 文档。

`RUNTIME_HELPERS_CONTEXT` 只描述 core helpers；插件 helper context 由各插件自包含 epoch context 发送。

## 13. TurnManager 多 user item 语义

移除 `_TAKEOVER_SEPARATOR = "\n---\n"` 拼接。

新的请求形态：

```python
UserInput(
    text="queued a",
    image_paths=[...],
    request_id="req_1",
)
```

合并 queued/guide/interrupt takeover 时：

- 仍可合并为一个实际 turn 执行。
- 但保留多个 `UserInput`。
- 在同一实际 turn 内依次追加多个 `role=user` items。
- ThreadStore 持久化多个 `item.user`，每个保留自己的 text/images/request_id。
- TUI/history/compaction/title generation 支持多个 user items。
- title generation 可继续使用第一个外部 user item。
- 当前用户消息前的内部 context items 只插入一次，位于该组外部 user items 前。

## 14. 迁移步骤建议

1. **重建插件平台骨架**
   - `src/uv_agent/builtin/` 与 builtin registry。
   - manifest/setup API。
   - plugin config parser/schema/merge/status。
   - command/action/UI/storage/runtime helper namespace registries。

2. **重建 context pipeline**
   - 结构化 XML renderer，顶层自动 `agent_` 前缀。
   - epoch publish/update/remove 主动机制。
   - turn enqueue/replay 主动机制。
   - internal context items 分组与 compaction 后 full epoch 恢复。
   - 移除 fingerprint/diff/provider 依赖。

3. **迁移 Goal 作为样板插件**
   - `/goal` command。
   - storage-backed goal state/tasks/phase/notes。
   - `rt.goal.*` namespace。
   - epoch context + replayable turn context。
   - status/UI contribution。

4. **迁移 Skills/MCP**
   - Skills：context + picker，无 helper。
   - MCP：context + picker + `rt.mcp.*` namespace。
   - 从 stable prompt/core engine 中移除 skills/MCP 硬编码。

5. **迁移 Workflow/Scheduler**
   - `rt.workflow.*`、`rt.scheduler.*` 从 core helper context 移到 builtin plugin。
   - workflow 注册 `workflow.prompt` action。
   - scheduler 改为 action registry consumer。
   - 配置迁移到 plugin config。

6. **TUI command/status/picker 改造**
   - command palette 消费 command registry。
   - status line 和 `/status` 聚合 plugin status contributions。
   - picker source 消费 UI contribution。
   - 隐藏或删除 Agent View `/agents`、`/bg`。

7. **TurnManager 多 user input 改造**
   - 去掉 `---` 合并。
   - TurnHandle 保存 `user_items`。
   - Engine 同 turn 内追加多个 external user items。
   - 更新 ThreadStore、history reconstruction、compaction、title、TUI queue 和测试。

8. **清理 core prompt/context/helper**
   - stable prompt 瘦身。
   - runtime helper context 只保留 core helpers。
   - 删除旧 goal/workflow/mcp/skills/worktree 硬编码路径。

## 15. 重点改动区域

- `src/uv_agent/plugins/*`：新插件 API、manager、registries、storage/config/context/action/UI。
- `src/uv_agent/builtin/*`：builtin goal/skills/mcp/workflow/scheduler 等插件。
- `src/uv_agent/agent/engine.py`：移除硬编码内置功能，接入主动 context pipeline 和 plugin manager。
- `src/uv_agent/agent/context_builder.py` / `src/uv_agent/prompts.py`：XML renderer、stable prompt 瘦身、core runtime helper context。
- `src/uv_agent/config.py`：插件配置结构、schema、level_ref 校验、迁移 workflow/scheduler 配置。
- `src/uv_agent/state_db.py`：plugin storage tables/indexes。
- `src/uv_agent/turn_manager.py`：多 external user item 队列语义。
- `src/uv_agent/tui2/app.py` / `components.py` / `events.py`：command/status/picker/panel 消费 registries 和 UI contributions；隐藏/删除 Agent View。
- `src/uv_agent_runtime/*`：动态 namespace helper proxy；移除功能域 helper 的 core 静态导出，改由 builtin plugin 注册。
- `tests/`：插件、context、config、turn manager、runtime helper namespace、TUI command/status/picker、workflow/scheduler 迁移测试。

## 16. 验证重点

- stable system prompt 和 `PYTHON_TOOL` 不因插件加载变化而改变。
- 模型可见 XML 顶层统一 `agent_` 前缀；内部字段不加前缀。
- 插件不能手写 XML；结构化 renderer 正确转义、排序 attrs、拒绝非法 tag。
- 同 epoch 内已落盘的 context 前缀序列不被改写。
- compaction 后 core 自动重发当前 full epoch documents。
- epoch update 批量 envelope 正确合并多个插件更新/移除。
- turn context 一次性消费；`replay_after_compaction` 只在新 epoch 重发一次。
- 外部 queued/guide/interrupt 用户输入不再 `---` 拼接，持久化为多个 `item.user`。
- Goal 插件完整替代旧 Goal 文件机制，并通过 `rt.goal.*` 工作。
- Skills/MCP/workflow/scheduler 不再由 core 硬编码 context/helper。
- command palette/status line 不再依赖硬编码 Goal/Skills/MCP。
- plugin storage 在高频读写/索引查询/并发下不丢数据。
- scheduler prompt 动作通过 action registry 和 workflow 插件解耦。
- `/agents`、`/bg` 已隐藏或移除，不残留 palette/help/status 入口。

## 17. 暂留问题

- Plugin storage 表结构、索引字段和 cursor 格式的最终细节。
- UI `Panel` 的 `todo/table/progress` 数据 schema 细节。
- `CommandResult actions` 的最小 action 类型集合和错误处理格式。
- Runtime namespace helper 的模型上下文展示格式。
- 旧 thread 中 goal 文件如何处理：本轮可不兼容，但需要避免启动时报错或误导模型。

## 18. 执行进度

> 本节由实现过程维护；设计约束以上文为准，除非发现阻塞性自相矛盾再询问用户。

- [x] 设计细节已与用户确认并写入本文件。
- [x] 阶段 1：插件平台骨架（manifest/setup、registries、config/schema、storage、namespace helpers）。
- [x] 阶段 2：主动式 context pipeline 与 XML renderer。
- [x] 阶段 3：builtin.goal 样板迁移。
- [x] 阶段 4：builtin.skills / builtin.mcp 迁移。
- [x] 阶段 5：builtin.workflow / builtin.scheduler 与 action registry 迁移。
- [x] 阶段 6：TUI command/status/picker/UI contribution 改造，隐藏 Agent View。
- [x] 阶段 7：TurnManager 多 external user item 语义。
- [x] 阶段 8：prompt/runtime helper context 瘦身、旧硬编码清理、测试更新与全量验证。
- [x] 验证：focused plugin/runtime/TUI suites 与最终 `uv run pytest -q --maxfail=30` 均已通过。
