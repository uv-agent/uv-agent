# Core + Builtin Plugins 改造草案

> 目标：把非核心能力从 Engine/TUI 中拆出为 builtin plugins，让第三方插件使用同一套一等公民接口。本文件是后续细化用的简略修改文档，当前不要求兼容旧插件/旧内部 API。

## 本轮约束

- **暂不支持 Agent View 插件化**：不为 Agent View 设计 command/UI/storage 扩展面；`/agents`、`/bg` 可暂时保留 legacy 实现，也可在冲突时移除/禁用。
- **允许破坏性修改**：旧插件 API、旧 Goal API、旧 TUI command 分发、旧配置字段都可以重构，不优先做兼容层。
- **必须保持 core 前缀稳定**：stable system prompt 和 `run_python` 工具定义不因插件变化改变；插件内容进入固定顺序的动态 context slots。
- **外部动作边界不变**：模型仍只有 `run_python`；插件只能通过 runtime helpers / host APIs 间接暴露能力。

## 推荐 Core 边界

Core 只保留：

- 线程、turn、历史重建、compaction、模型调用与 `run_python` orchestration。
- 稳定 context pipeline：规则加载、runtime environment、helper 声明、插件 context slot 排序与去重。
- 插件宿主能力：发现/加载、生命周期、事件、命令、UI contribution、配置、存储、runtime helper registry。
- 基础 TUI 框架：transcript、composer、command palette、status line、通用 panel/list/table renderer。

从 core 迁出为 builtin plugins：

- Goal mode。
- Workflow / scheduler 的模型上下文、命令与状态展示（不含 Agent View）。
- Skills / MCP discovery 的上下文与 picker/mention/命令。
- Worktree notice。
- 其他非核心状态栏 badge、提示块、辅助面板。

## 插件系统需要新增的能力

### 1. Manifest 与加载顺序

为每个插件声明：

- `id`、`version`、`builtin`、`default_enabled`。
- `capabilities`: `context`, `commands`, `ui`, `storage`, `runtime_helpers`, `events`。
- `dependencies` / `optional_dependencies`。
- `priority` 或固定 slot，用于稳定 context/UI 排序。

内置插件也走同一套 manifest，避免 Engine/TUI 对内置功能做特殊判断。

### 2. 结构化 Context Contribution

替换当前 `TurnContextBlock(text=...)` 为结构化贡献，例如：

```python
ContextContribution(
    slot="before_user",
    tag="goal_mode",
    attrs={"status": "enabled"},
    body={"objective": "...", "files": {...}, "rules": [...]},
    dedupe_key="goal-mode-state",
    visibility="model",
)
```

Core 负责：

- dict/list/text → XML 渲染与转义。
- 截断、去重、fingerprint、历史持久化。
- 插件排序稳定化。
- 统一包裹/标记来源，不让插件手写 XML。

### 3. Command Registry

新增 host command API：

- `register_command(name, description, handler, subcommands=..., args_schema=...)`。
- handler 返回结构化结果：更新 composer、打开 picker、写 transcript event、提交 turn、更新插件状态等。
- command palette 从 registry 读取，不再在 TUI 中硬编码 `/goal`、`/skills`、`/mcp`。
- 命令冲突直接报错或按 builtin 优先；不做复杂兼容。

Goal mode 迁移后应通过插件注册：

- `/goal enable [objective]`
- `/goal disable`
- `/goal reset [objective]`
- `/goal status`

### 4. UI Contribution Model

TUI 不接受插件渲染字符串，而消费 UI-neutral 数据结构：

- `StatusItem(id, label, text, priority, style_hint, scope=thread/global)`。
- `Panel(id, title, kind=list/table/todo/progress/text, data, actions=...)`。
- `PickerSource(id, trigger, query_handler)`。
- `TranscriptEvent(kind, title, body, severity, metadata)`。

本轮只覆盖 transcript TUI 需要的能力：status line、command palette、picker/mention、list/todo/table panel。Agent View 不进入该协议。

### 5. Storage API

基于现有 SQLite 底座提供插件隔离存储，不直接让多数插件管理裸 SQLite。

建议 API：

- `context.storage.global_kv(plugin_id)`：用户级全局。
- `context.storage.project_kv(plugin_id)`：当前项目级。
- `context.storage.thread_kv(plugin_id, thread_id)`：线程级 convenience。
- `get/set/delete/list_prefix/update_json`。
- 高阶需求可提供 `collection(name).put/get/query/delete`，并允许 manifest 声明简单 indexes。

存储原则：

- 默认 JSON/document 存储，满足易用。
- 对高频列表/查询提供索引，避免每次扫描 JSON。
- 插件 namespace 强隔离。
- builtin plugins 也必须使用该 API，避免私有文件散落。

### 6. Config API

保留现有 user/project merge，但插件配置改为 schema 化：

```json
{
  "plugins": {
    "builtin.goal": {"enabled": true, "config": {...}},
    "builtin.workflow": {"enabled": true, "config": {...}}
  },
  "model_roles": {
    "agent.default": "medium",
    "workflow.node": "small",
    "subagent.default": "small",
    "plugins.goal.default": "medium"
  }
}
```

建议新增 `model_roles`，替代把 workflow/subagent 默认模型继续塞到 `runtime` 下。解析优先级：plugin/thread override → project → user → default role → `runtime.default_level`。

插件可声明：默认配置、redaction keys、可编辑字段、模型 role 需求。

### 7. Events 与生命周期

将 host event surface 稳定化：

- `thread.*`, `turn.*`, `model.*`, `tool.*`, `context.*`, `config.*`, `ui.*`。
- 插件可订阅公开事件；不再只依赖 `plugin.*`。
- 插件可发布插件 namespace 事件，TUI 通过 UI contribution/render hints 显示。

## 其他功能修改点

### 排队/引导消息不再用 `---` 合并

现状：TurnManager 会把 takeover/guide 的多个用户请求拼成一个字符串。建议改为：

- `TurnHandle` 保存 `user_items: list[UserInput]`，每个输入有 text/image_paths/request_id。
- Engine 在同一 turn 内追加多个 `role=user` message item。
- ThreadStore 持久化多个 `item.user`，而不是一个合并后的文本。
- compaction/title/history/TUI queue 按多个 user item 展示。

模型层已经能处理多个 user message；主要改动在 TurnManager、ThreadStore 事件语义、history reconstruction、测试。

### XML 渲染集中到 Core

新增通用 renderer：

- `xml_text(value)`：转义文本。
- `xml_attrs(dict)`：属性转义与排序。
- `dict_to_xml(tag, data, attrs=None)`：简单 dict/list 渲染。
- `render_context_contribution(contribution)`：统一渲染插件 context。

迁移后 Goal/workflow/MCP/skills/worktree 都不应手写 XML 模板，除非是 core 内部固定模板。

## 迁移步骤建议

1. **先改插件宿主接口**：manifest、context contribution、command registry、UI contribution、storage/config skeleton。
2. **迁移 Goal mode**：作为样板插件，覆盖 command + per-thread state + status + pre-user context + runtime helper。
3. **替换 TUI command/status 硬编码**：保留通用命令和 status renderer；Goal/skills/MCP/workflow 从插件贡献读取。
4. **迁移 skills/MCP/worktree/workflow context**：从 Engine `_pre_user_context_items` 中移出。
5. **修改 TurnManager 多 user message 语义**：移除 `---` 合并，并更新历史/compaction/TUI 测试。
6. **重整配置**：引入 `model_roles` 与插件 schema config；删除或降级旧 workflow-specific runtime 字段。
7. **清理 core**：Engine 不再 import Goal/worktree/workflow/skills/MCP 具体实现，只调用插件 manager 收集贡献。

## 需要重点改的区域

- `src/uv_agent/plugins/*`：核心插件 API 扩展。
- `src/uv_agent/agent/engine.py`：移除硬编码内置功能，改为收集插件 contributions。
- `src/uv_agent/tui2/app.py` / `components.py` / `events.py`：命令、状态栏、picker、panel 改为消费 registry/contributions；不处理 Agent View 插件化。
- `src/uv_agent/config.py`：插件配置 schema、model roles、继承解析。
- `src/uv_agent/state_db.py`：插件存储表或 collection/index 表。
- `src/uv_agent/turn_manager.py`：多 user item 队列语义。
- `src/uv_agent/prompts.py` / `agent/context_builder.py`：XML renderer 与 stable dynamic context slots。
- `src/uv_agent_runtime/*`：插件 runtime helpers 的统一暴露方式。

## 验证重点

- stable system prompt 和 `PYTHON_TOOL` 字节序不因插件加载变化而改变。
- 动态 context 顺序稳定，插件 enable/disable 只影响固定 slot 内容。
- Goal 插件完整替代旧 Goal 功能。
- command palette/status line 不再依赖硬编码 Goal/skills/MCP。
- 多 user queued/guide 输入能被历史重建、compaction、title generation 正确处理。
- 插件 global/project/thread storage 在并发下不丢数据。

## 暂留问题

- builtin plugins 是放在 `src/uv_agent/builtin_plugins/`，还是独立 package entry points。
- 插件 UI panel 的最小 schema 是否先只做 list/table/todo/text。
- 插件 storage 是否允许自定义 SQL migration，还是本轮只提供 KV/document/index。
- 旧配置字段是否直接删除，还是只在 parser 中做一次性迁移提示。
