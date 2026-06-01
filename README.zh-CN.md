# uv-agent

<img align="right" src="docs/t2.png" alt="uv-agent tui2 截图" width="300">

[English](README.md)

`uv-agent` 是一个 Windows-first 的 coding agent，默认提供 ANSI-first 终端 TUI。它围绕
单一
对外动作面 `run_python` 设计：模型编写受管理的 Python 脚本，uv-agent 通过
`uv run` 执行，再由 `uv_agent_runtime` 提供文件编辑、命令执行、代码搜索、MCP、
subagent、图片附件等 helper。

这个边界让编码任务更容易检查、重放、中断和压缩。这一设计让它可以轻松移植到任何
Python 和 uv 环境。项目仍处于实验阶段，公开 API、配置字段和 runtime 行为可能继续变化。

## 为什么用 uv-agent？

- **Windows-first 编码界面。** 终端原生 transcript、多行输入框、命令面板、模型/tool
  时间线、文件和 thread mention、图片附件，以及中英文界面。
- **单一动作边界。** 模型没有直接 shell、文件系统、浏览器或 MCP 工具；外部工作都通过
  受管理的 Python run 和同一条持久事件流完成。
- **适合长任务。** checkpoint 压缩、workspace rules、skills、MCP declarations、
  Goal 状态和 Worktree 状态会按需作为结构化上下文重放。
- **实用编码工作流。** `/goal` 为较长任务提供轻量的 per-thread checklist/notes；
  Worktree mode 为任务创建隔离的 Git 分支 worktree。
- **可扩展 runtime。** 插件可以注册 `uv_agent_runtime` helper、订阅事件，也可以从外部
  系统提交 turn，同时不增加额外的模型可见工具。

## 快速开始

先安装：

- **uv** — https://docs.astral.sh/uv/getting-started/installation/
- **ripgrep** — https://github.com/BurntSushi/ripgrep#installation
- **Git** — 常规编码工作流和 Worktree mode 需要。

运行最新发布版本：

```powershell
uvx uv-agent@latest
```

在本地源码中运行：

```powershell
uv run uv-agent
```

不打开 TUI，直接问一个单轮问题：

```powershell
uvx uv-agent@latest ask "Reply with exactly: ok"
```

继续已有 thread：

```powershell
uvx uv-agent@latest ask --thread thr_xxx "Continue from here"
```

## 模型配置

uv-agent 不内置真实 provider 配置。发起模型请求前，需要至少配置一个 provider、一个
model 和一个 level。

配置从 `~/.uv-agent/config.json` 读取，项目可以用 `.uv-agent/config.json` 覆盖。
项目本地 `.uv-agent/` 目录已被 git 忽略。API key 建议放在环境变量或被忽略的本地
配置里。

支持的模型 API 格式：

| `api` 取值 | 格式 |
| --- | --- |
| `"responses"` | OpenAI Responses API |
| `"chat_completions"` | OpenAI Chat Completions API |
| `"anthropic_messages"` | Anthropic Messages API |

<details>
<summary>完整配置示例</summary>

```json
{
  "providers": {
    "deepseek": {
      "base_url": "https://api.deepseek.com",
      "api_key": "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
      "timeout_s": 7200,
      "chat_completions": {
        "path": "/chat/completions"
      },
      "message_passthrough": {
        "assistant": [
          "reasoning_content"
        ]
      },
      "reasoning_display": {
        "assistant_message_fields": [
          "reasoning_content"
        ],
        "stream_delta_fields": [
          "reasoning_content"
        ]
      }
    },
    "minimax": {
      "base_url": "https://api.minimaxi.com",
      "api_key": "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
      "timeout_s": 7200,
      "chat_completions": {
        "path": "/v1/chat/completions"
      },
      "anthropic_messages": {
        "path": "/anthropic/v1/messages"
      }
    }
  },
  "models": {
    "deepseek-v4-flash": {
      "provider": "deepseek",
      "model": "deepseek-v4-flash",
      "api": "chat_completions",
      "supports_images": false,
      "context_window_tokens": 1000000,
      "params": {
        "reasoning_effort": "high"
      }
    },
    "deepseek-v4-pro": {
      "provider": "deepseek",
      "model": "deepseek-v4-pro",
      "api": "chat_completions",
      "supports_images": false,
      "context_window_tokens": 1000000,
      "params": {
        "reasoning_effort": "max"
      }
    },
    "MiniMax-M2.7": {
      "provider": "minimax",
      "model": "MiniMax-M2.7-highspeed",
      "api": "anthropic_messages",
      "supports_images": false,
      "context_window_tokens": 204800
    }
  },
  "levels": {
    "deepseek-flash": {
      "model": "deepseek-v4-flash"
    },
    "deepseek-pro": {
      "model": "deepseek-v4-pro"
    },
    "MiniMax-M2.7": {
      "model": "MiniMax-M2.7"
    }
  },
  "runtime": {
    "default_level": "deepseek-flash",
    "ask_default_level": "deepseek-flash",
    "store_provider_response": false,
    "max_agent_rounds": 1000,
    "compression": {
      "enabled": true,
      "model_level": "deepseek-flash",
      "trigger_ratio": 0.9
    },
    "title_generation": {
      "enabled": true,
      "model_level": "deepseek-flash"
    },
    "branch_name_generation": {
      "enabled": true,
      "model_level": "deepseek-flash",
      "timeout_s": 15.0
    }
  },
  "runner": {
    "default_timeout_s": 7200,
    "max_output_bytes": 1000000
  },
  "pricing": {
    "currency": "RMB",
    "unit": "1M_tokens",
    "models": {
      "deepseek-v4-flash": {
        "input": 1,
        "output": 2,
        "cached_input": 0.02
      },
      "deepseek-v4-pro": {
        "input": 3,
        "output": 6,
        "cached_input": 0.025
      }
    }
  },
  "ui": {
    "completion_notification": {
      "enabled": true
    }
  },
  "plugins": {
    "disabled": [],
    "config": {}
  }
}

```

</details>

在 TUI 中可以用 `/config` 切换默认 level、界面语言、完成通知和自动压缩等用户侧设置。
完整字段见 [configuration](docs/configuration.md)，单独示例文件见
[config.example.json](docs/config.example.json)。

## 日常工作流

- 正常输入后按 `Enter` 发送；需要在输入框里换行时按 `Ctrl+Enter` 或 `Ctrl+J`。
- 空输入框按 `/` 打开 tui2 命令面板，继续输入即可过滤命令。
- `@` 插入文件 mention，`@@` 插入 thread mention，`/threads` 恢复历史任务。
- `/level <name>`（或 `/model <name>`）切换 model level；选择会按 thread 记住。
- `/goal enable [objective]` 开启持久 checklist/notes；可以在第一条消息前启用，
  thread 开始时再初始化。
- 需要把工作分发到后台 worktree 会话时使用 Agent View；它在工作流里的定位见
  [Agent View](#agent-view)。
- `/status`、`/mcp`、`/skills` 可查看 runtime 状态和可用能力。
- 如果需要 `/config`、`/models`、Worktree 管理或剪贴板图片快捷键等原 Textual 专属面板，
  使用 `uv-agent tui` 启动旧界面。

完整命令和快捷键见 [TUI and slash commands](docs/tui.md)。

## Agent View

Agent View 是 uv-agent 用来管理多条后台 agent 会话的仪表盘。它适合把不会立即阻塞当前
对话的工作拆出去并行处理，例如定位 bug、尝试实现方案、修测试、做代码审阅，或运行较长的
验证任务。

从 Agent View 派发的任务会作为后台会话运行在独立 Git worktree 和自动生成的分支里。
这样每个任务的改动都和当前 checkout 隔离，之后可以更容易地查看、比较、合并或
丢弃。面板会把这些会话集中到一处，方便跟踪状态、浏览最新输出、继续某个会话，或在需要时
把它接回主对话深入处理。

Agent View 更像一个有边界的后台工作队列，而不是全局历史线程列表。普通线程只有明确加入后
才会显示；从 Agent View 创建的 worktree 任务会自动纳入。这样面板会聚焦于被委派出去的工作，
不会混入项目里的每一次对话。

## TUI 界面

uv-agent 提供两个交互界面：

- **tui2** — 默认界面（`uv-agent` 或 `uv-agent tui2`）。轻量 ANSI TUI，直接在终端中
  渲染，包含紧凑状态行、命令和 mention 面板、模型/tool 流式事件、Goal mode、
  Worktree mode 和图片附件。
- **Textual TUI** — 原始 widget-based 界面（`uv-agent tui`）。保留更丰富的 Textual
  布局，偏好该界面时仍可手动进入。截图只链接在这里：[docs/t1.png](docs/t1.png)。

## 插件

插件是通过 `uv_agent.plugins` entry point 发现的普通 Python 包。插件运行在 uv-agent
host 进程中，可以注册 runtime helper、订阅事件，或从外部系统提交 turn。

一次性带插件运行时，把插件包加到 `uvx` 启动的同一个环境即可：

```powershell
uvx --with your-uv-agent-plugin uv-agent@latest
```

已经安装但不想启用的插件，可以在配置里的 `plugins.disabled` 中禁用。插件 API、事件总线、
helper 注册和示例见 [Plugin system](docs/plugins.md)。

## Runtime 与上下文

模型每一轮看到的输入，都由稳定的 system prompt 和可重放的 pre-user 上下文组成。
system prompt 保持精简，通常不变；项目和运行时信息则会在用户消息之前，以结构化消息按需注入。

- **受管理的运行时。** `run_python` 是唯一对外动作面。受管理脚本运行在项目共享 uv 环境中：
  `~/.uv-agent/projects/<project-id>/runner/scriptenv/`，并通过 `uv_agent_runtime`
  使用文件编辑、搜索、子进程、依赖安装、subagent、图片上下文、MCP 客户端、插件 helper 等能力。
  脚本使用的 uv 环境与当前工作目录是两个独立概念；工作目录可以随 `enter_dir` 或 Worktree mode 变化。
- **增量 runtime 上下文。** runtime environment、model levels、helper 列表、脚本环境的直接依赖、
  skills、MCP servers 和插件 helper 会拆成带指纹的 context part。uv-agent 只把变化的部分放进
  `<context_update ...>` 发送；skill 或 MCP server 消失时，也会显式标记移除，避免模型继续依赖旧能力。
- **工作区与 thread 上下文。** workspace rules 会渐进披露：模型先收到规则索引；只有进入相关目录时，
  才会收到对应 AGENTS.md 的完整内容。active cwd 变化、图片附件、Worktree 通知、tool 结果、run 日志和
  thread metadata 都进入同一条持久事件流，并在重建 turn 时重放。
- **Goal mode 的持久任务记忆。** `/goal` 会在
  `~/.uv-agent/projects/<project-id>/goals/<thread-id>/` 下为当前 thread 建立一层持久记忆，包含
  `goal.json`、`checklist.md` 和 `notes.md`。Goal mode 开启后，uv-agent 会重放 `<goal_mode>` 通知，告诉模型
  这些文件的位置和维护规则。模型用 `checklist.md` 记录验收标准、进度、阻塞和下一步，用 `notes.md`
  记录决策、调查结果、约束和交接上下文。受管理脚本可通过 `goal_paths()` 找到这些文件，不需要硬编码路径。
- **压缩与恢复。** checkpoint compaction 会总结对话，但会把可重新加载的 runtime context、
  workspace rules，以及 Goal/Worktree 通知排除在 retained history 之外。新的 epoch 会先重放
  当前结构化上下文，再接入保留历史；对于长任务进度，压缩或恢复后应优先参考 Goal 文件，
  而不是只依赖聊天记录。

thread 状态、run 日志、共享脚本依赖、附件、Goal 文件和其他项目 runtime 数据，统一保存在
`~/.uv-agent/projects/<project-id>/`。

## 文档

- [Configuration](docs/configuration.md)
- [Full config example](docs/config.example.json)
- [TUI and slash commands](docs/tui.md)
- [Runtime and managed scripts](docs/runtime.md)
- [Plugin system](docs/plugins.md)

## 开发

`uv-agent` 采用“自举”开发：项目日常使用 uv-agent 自身完成阅读、修改、测试和迭代。

```powershell
uv run pytest
```

本地调试状态、截图、配置、脚本、运行记录和 thread 数据应放在 `.uv-agent/`，不要提交。

## 许可证

MIT。见 [LICENSE](LICENSE)。
