# TUI

Textual shell for the uv-agent conversation experience.

## 边界

- 负责：展示 transcript、composer、状态行、临时面板、skills/MCP 摘要、轻量 thread 命令和 Python runner 事件。
- 负责：把 `AgentEngine.run_turn` 事件转为 UI cell，不直接执行模型请求或 runner 操作。
- 不负责：配置读取、模型协议、Python runner、JSONL 记录和自动压缩策略。

## 约束

- 默认界面保持 Codex-style：单一主时间线 + 底部 composer/status，不恢复固定三栏 dashboard。
- 默认状态行保持低信息密度；模型、API、thread、state paths 等细节放入 `/status` 或临时面板。
- Composer 是多行输入：Enter 换行，Ctrl+Enter/Ctrl+J 发送；斜杠命令需要即时提示，Tab 可补全首个匹配。
- 退出必须避免误触：Ctrl+Q、Ctrl+C 或 `/quit` 均需二次确认；Windows 下要拦截控制台 Ctrl+C 事件。
- Python 工具调用必须作为 transcript 内联事件显示；长运行任务先显示 running，再更新为 exit/stdout/stderr 摘要。
- 配置、threads、run 详情使用临时面板显示，默认不占据主界面。
- 窄终端下优先保留 transcript 和 composer，状态行只显示短摘要。

## Units

### `UvAgentApp`

- 职责：Textual app 入口，管理输入提交、轻量命令、排队输入、assistant 流式 cell 和 runner 事件 cell。
- 对外约定：由 CLI 传入 `project_root` 后可直接 `.run()`；核心能力来自 `create_engine(project_root)`。
- 注意：不要在该类中新增模型 API 或 runner 业务规则，新增展示格式优先放入 `formatting.py`。

### `TranscriptCell`

- 职责： transcript 中的最小渲染块。
- 对外约定：通过 Textual/Rich markup 或 Rich renderable 更新内容。

### `formatting.py`

- 职责：保存与 TUI 展示有关的纯格式化 helper。
- 对外约定：函数必须无副作用，不能读取配置、线程或 runner 状态。
