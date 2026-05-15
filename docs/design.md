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

该包提供给 agent 使用的快捷 API，例如文件读写封装、结构化输出、JSONL 事件、路径处理、后续子 agent 快捷启动入口等。runner 必须确保最终执行脚本的 PEP 723 inline metadata 中包含 `uv-agent-runtime` 的版本约束。

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

## 受管理脚本与重跑

临时脚本是受管理 artifact，不是运行完即删除的系统临时文件。

建议结构：

```text
.uv-agent/
  scripts/
    <script_id>/
      script.py
      metadata.json
  runs/
    <run_id>.jsonl
  threads/
    <thread_id>.jsonl
```

每次新脚本执行：

- 分配 `script_id`
- 保存原始脚本和最终脚本
- 分配 `run_id`
- 写入 run JSONL

runner 必须支持重跑：

- 按 `script_id` 重跑某个已保存脚本
- 按 `run_id` 尽量复现某次运行

重跑默认复用原脚本内容。是否复用原 `uv_args`、`script_args`、cwd 和 timeout，必须由 API 明确表达。若需要严格依赖复现，后续可引入 `uv lock --script` 生成脚本锁文件；未引入锁文件前，不承诺跨时间的依赖版本完全一致。

## 会话模型

参考 Codex 的 thread / turn / item 分层，但实现保持轻量：

- `thread`：一段会话。
- `turn`：一次用户输入到 agent 完成响应的过程。
- `item`：消息、推理摘要、runner 调用、runner 输出、压缩摘要等可持久化事件。

初版只实现单 agent 会话，不实现子 agent。未来子 agent 可以作为 runtime 快捷能力或 agent 编排能力加入，但不改变“唯一外部工具是 Python runner”的边界。

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

## 自动压缩

自动压缩是 thread 级能力：

- 当上下文接近模型限制时触发
- 用户可以手动触发
- 压缩摘要必须作为 item 写入 thread JSONL

压缩摘要至少保留：

- 当前目标
- 已确认的项目约束
- 未完成任务
- 最近关键 runner 结果
- 可重跑脚本引用
- 用户明确偏好或决策

## 配置读取

配置读取参考 Codex 的分层思想，但保持 Python 项目简洁。建议优先级：

1. 内置默认值
2. 用户配置
3. 项目配置
4. 会话覆盖

具体文件名、schema、合并规则和敏感信息处理在实现配置 feature 时写入对应 feature `AGENTS.md`。OpenAI 凭据不得写入仓库、JSONL 或脚本 artifact。

本地测试配置可以放在 `.uv-agent/config.json`，该路径被 git 忽略。配置需要支持 provider、model、level、runtime、runner 等块，并在日志和错误展示中对 `api_key` 这类敏感字段脱敏。

配置是本项目自己的 schema，不照搬外部工具配置。provider 只描述共享的 base URL、认证、headers 和 endpoint，model 决定使用 `responses` 还是 `chat_completions` API。

模型客户端必须支持：

- Responses API
- Chat Completions API
- SSE 流式输出
- 非流式调用，用于压缩等后台任务

## TUI 目标

TUI 使用 Textual。

- 默认界面参考 Codex：单一主时间线 + 底部 composer，不把屏幕切成固定三栏。
- 主时间线显示用户输入、assistant 流式输出、Python runner 状态和 stdout/stderr 摘要。
- 底部 composer 固定显示当前运行状态、模型/API、上下文估算和当前 thread 标识。
- 临时面板只用于聚焦查看脚本内容、完整 runner 日志、thread 列表、配置状态等；它们不应抢占默认对话体验。
- 窄终端下优先保留 transcript 和 composer，附属信息降级为短状态文本。

TUI 只消费 core/session/runner 提供的状态和事件，不把业务规则写死在 UI 层。
