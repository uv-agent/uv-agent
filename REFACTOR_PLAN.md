# Refactor Plan: simplify `run_python`

目标只有一个：让 `run_python` 不再因为一次性脚本路径触发 `uv` 环境缓存膨胀。状态目录整理可以一起做，但它是配套清理，不是主线。

我的判断：这次应该直接放弃“保存脚本、按 `script_id` 重跑、PEP 723 自动依赖注入”这套产品形态。它实现重、收益低，而且正好制造了当前缓存问题。新的 `run_python` 应该像一个项目级 Python 工作台：同一个项目共享一个脚本 venv，每次调用只是写入并执行一份临时代码。

## 决策

### 1. `run_python` 语义

新的工具签名只保留：

```python
run_python(
    code: str,
    script_args: list[str] = [],
    timeout_s: float | None = None,
)
```

删除 `script_id`、`run_id`、`rerun_mode`、`uv_args`、`cwd`。旧的 rerun 能力不保留。

`cwd` 不作为工具参数暴露给模型。脚本工作目录继续由线程的 active cwd 决定；模型需要切换工作目录时，应使用现有的 `enter_dir` 机制，而不是在 `run_python` 调用里临时指定。

### 2. 项目共享 venv

每个项目只有一个脚本环境：

```text
~/.uv-agent/projects/<project-id>/runner/scriptenv/.venv/
```

`PythonRunner` 第一次运行时懒初始化：

1. `uv venv runner/scriptenv/.venv`
2. 把 `uv-agent` runtime 包安装进这个 venv
3. 后续所有脚本都用这个 venv 的 Python 直接执行

这里的 runtime 安装是 host 内部行为，不再暴露成 `RunnerConfig.runtime_dependency`。开发环境可以由 host 自动识别当前源码包并安装本地包；正常安装环境则安装当前分发包。关键点是：脚本仍然通过 venv 里的 package import `uv_agent_runtime`，不能靠仓库路径或当前进程 `.venv`。

### 3. 依赖管理交给模型

不再解析 PEP 723，也不自动装用户脚本依赖。模型需要包时，在脚本里通过 runtime 包装工具执行安装。安装依赖时不要 `enter_dir`，也不要给 `run_process_text` 传 `cwd`；目标环境由 `--python sys.executable` 决定，当前目录不参与依赖安装语义。

```python
import sys

from uv_agent_runtime import run_process_text

run_process_text(
    ["uv", "pip", "install", "--python", sys.executable, "-q", "requests"],
    check=True,
)
```

安装结果会留在项目共享 venv 里，之后的 `run_python` 可以复用。

### 4. run log 只保留最近记录

脚本和日志按 run 存：

```text
runner/runs/<run_id>.py
runner/runs/<run_id>.jsonl
```

新增 `RunnerConfig.max_run_logs`，默认 200。超过后按时间淘汰，`.py` 和 `.jsonl` 成对删除。

### 5. runtime 环境变量

保留现有 runtime helper 需要的变量，尤其是：

- `UV_AGENT_RUNTIME_PROJECT_ROOT`
- `UV_AGENT_RUNTIME_STATE_DIR`
- `UV_AGENT_RUNTIME_THREAD_ID`
- `UV_AGENT_RUNTIME_THREAD_KIND`
- `UV_AGENT_RUNTIME_TURN_ID`
- `UV_AGENT_RUNTIME_RUN_ID`

删除 `UV_AGENT_RUNTIME_SCRIPT_ID`。暂时不强求新增 `UV_BIN`；runner 内部能找到 `uv` 即可，提示里也可以继续写 `"uv"`。

### 6. 状态目录顺手整理

新项目使用更简单的布局，不做旧布局迁移：

```text
~/.uv-agent/projects/<project-id>/
├── threads/
│   ├── <thread_id>.jsonl
│   ├── <thread_id>.json
│   └── <thread_id>.lock
├── subthreads/
├── attachments/<thread_id>/
├── runner/
│   ├── runs/
│   └── scriptenv/.venv/
└── tui/clipboard/
```

线程元数据和 JSONL 放同目录同名文件。`parent_script_id` 一并删除，只保留 `parent_run_id` 等仍然有意义的关系。

## 主要改动

- 删除 `runner/metadata.py`、`runner/store.py`、`RerunRequest`、`PythonRunner.rerun/stream_rerun`。
- 删除 `PythonRunRequest.uv_args`，删除 `PythonRunResult.script_id/script_path/final_script_path`。
- 删除 runtime helper `saved_scripts` 及提示中的 rerun/saved scripts 内容。
- 新增 `runner/scriptenv.py`：负责 venv 创建和 runtime 包安装。
- 新增 `runner/run_log.py`：负责 `<run_id>.py/.jsonl` 写入和 LRU 清理。
- 重写 `PythonRunner.stream_run`：ensure venv → 写 run 脚本 → venv Python 执行 → 复用现有 stdout/stderr/timeout/cancel/structured event 管道。
- 更新 `PYTHON_TOOL` schema 和系统提示，彻底移除 PEP 723/rerun/script id 描述。
- 更新 config、docs、tests、TUI 文案和 formatting。

## 落地顺序

1. 先改 runner config、tool schema、models，切掉旧参数。
2. 实现 `scriptenv.py`，确保共享 venv 能创建并安装 `uv_agent_runtime`。
3. 实现 `run_log.py`，重写 `PythonRunner.stream_run`。
4. 删除 rerun、PEP 723、saved scripts 相关代码和测试。
5. 调整状态目录布局和 thread/attachment/clipboard 注入路径。
6. 更新文档和 prompt。
7. 跑 `uv run pytest`，再做手动 smoke。

## 验证重点

- 第一次 `run_python` 会创建 venv，且 `from uv_agent_runtime import ...` 可用。
- 多次运行不会让 `~/.cache/uv/environments-v2/` 每次新增环境。
- 脚本里通过 `run_process_text(["uv", "pip", "install", "--python", sys.executable, ...])` 安装依赖后，下一次可直接 `import requests`。
- timeout/cancel 仍能杀掉子进程树。
- structured events、attachments、subagent、thread helpers 仍能工作。

## 不做

- 不迁移旧状态目录。
- 不保留 rerun。
- 不保留 saved scripts。
- 不解析 PEP 723。
- 不做依赖锁定、冲突检测或自动清理 venv。
- 不做 `/reset-scriptenv`，以后需要再加。
