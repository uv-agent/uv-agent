# uv-agent

[English](README.md)

`uv-agent` 是一个 Windows-first 的 coding agent，提供 Textual TUI。它首先面向
Windows 体验设计，尽量避免许多 coding agent 在 PowerShell 引号、shell 语义、
Unix-first 假设上经常“水土不服”的问题。它对外只有一个动作面：`run_python`：
模型把 Python 脚本提交给受管理的 `uv run` runner，再由脚本完成实际工作，而不是
依赖脆弱的 shell 片段。已安装插件可以在不增加额外模型工具的前提下，为这个 runtime
扩展新的 helper、事件订阅和外部 turn 入口。围绕这个 `run_python` 边界，uv-agent 的上下文层采用
Harness Engineering 思路：通过 checkpoint 压缩、稳定的增量更新、协议安全的中断处理
和 epoch 重放，让模型视角在长程任务中保持一致。见 [上下文管理](#上下文管理)。

公开 API、配置字段和 runtime 行为可能随项目演进而继续调整。

## 前置要求

请先安装以下工具：

- **uv** — https://docs.astral.sh/uv/getting-started/installation/
  Python 包与项目管理器，用于运行 agent。
- **ripgrep** — https://github.com/BurntSushi/ripgrep#installation
  用于在工作区内快速搜索文件内容。

## 安装与运行

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

## 配置

用户级配置默认位于 `~/.uv-agent/config.json`。项目可以用 `.uv-agent/config.json`
覆盖；这个项目本地目录已被 git 忽略。API key 应放在环境变量或被忽略的本地配置里。

> **API 兼容**  
> 本项目支持三种 API 格式——在模型配置中设置 `api` 字段即可：
> 
> | `api` 取值 | 格式 | 状态 |
> |---|---|---|
> | `"chat_completions"` | OpenAI Chat Completions API | ✅ 支持 |
> | `"responses"` | OpenAI Responses API | ✅ 支持 |
> | `"anthropic_messages"` | Anthropic Messages API | ✅ 支持 |
> 
> 欢迎提交 Issue 和 PR！示例配置

```json
{
  "providers": {
    "deepseek": {
      "base_url": "https://api.deepseek.com",
      "api_key": "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
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

在 TUI 中可以用 `/config` 切换默认 level、界面语言和自动压缩。把 `ui.language`
设为 `zh-CN` 可使用中文界面。完成通知可通过 `ui.completion_notification`
配置；非 Windows 平台使用终端 bell 作为完成提示音。


完整字段见 [configuration](docs/configuration.md)，详细示例见
[config.example.json](docs/config.example.json)。

## 文档

- [Configuration](docs/configuration.md)
- [Full config example](docs/config.example.json)
- [TUI and slash commands](docs/tui.md)
- [Runtime and managed scripts](docs/runtime.md)
- [Plugin system](docs/plugins.md)

## 核心思路

- agent 对外只有一个动作面：`run_python`。
- 受管理脚本运行在项目共享 uv 环境中；第三方依赖通过
  `add_dependency` 添加到这个环境。
- 发布包同时包含 `uv_agent` 和 `uv_agent_runtime`；受管理脚本从
  `uv_agent_runtime` 导入快捷 helper。
- 已安装插件可以注册额外的 `uv_agent_runtime` helper、订阅 agent 事件，并从外部系统
  提交 turn，同时保持单一 `run_python` 动作边界。
- workspace rules、skills 和 MCP declarations 作为上下文渐进披露。MCP 调用通过
  Python runtime helper 完成，不直接暴露成模型工具。
- thread 状态、run 日志、共享脚本环境和附件位于
  `~/.uv-agent/projects/<project-id>/`。

## 上下文管理

上下文管理是 uv-agent Harness Engineering 思路的一部分：把 Agent 的输入、行动、状态和异常处理都纳入一套明确的工程协议中，让它在长时间运行时仍然可追踪、可恢复、可维护。两个机制构成核心锚点：**checkpoint 压缩**提供持久续接点，单一 **`run_python` 执行面**让所有外部动作进入同一条事件流。

- **基于指纹的增量更新。** runtime environment、model levels、helpers、skills、MCP declarations 等上下文会被拆成独立部分。只有发生变化的动态部分会通过 `<context_update ...>` 重新发送；未变化内容在当前 epoch 内继续有效，被移除的 skill 或 MCP server 会显式标记为不可再依赖。
- **稳定前缀与稳定顺序。** system prompt 保持稳定，动态上下文以 pre-user message 的形式追加，并使用固定的更新前缀和固定的章节顺序，减少长对话中因上下文变化带来的漂移。
- **协议安全的序列补全。** 因为 `run_python` 是唯一对外动作面，tool call、runner result、工作目录变化、rule 加载、附件和依赖状态都会进入同一条持久事件流。回合被中断时，未完成的工具调用会收到显式的合成输出和桥接消息；部分流式输出、provider 错误或工具错误会被记录下来，而不会被伪装成成功完成。
- **压缩后的 epoch 重放。** 压缩 checkpoint 保存 continuation summary 和近期保留对话，同时排除可重新加载的 runtime/rules 上下文。新的 epoch 会先重新发送当前 runtime context 和 workspace rules，再接入 retained history；工具调用后的中途压缩也使用同样顺序，让 assistant 在压缩摘要之后继续执行。

这些机制让 uv-agent 在工作区变化、运行时变化、中断、错误和超长会话中，仍能保持模型视角一致、任务连续，并稳定地长程运行。

## 开发

`uv-agent` 采用“自举”开发：项目日常使用 uv-agent 自身完成阅读、修改、测试和迭代。

```powershell
uv run pytest
```

本地调试状态、截图、配置、脚本、运行记录和 thread 数据应放在 `.uv-agent/`，不要提交。

## 许可证

MIT。见 [LICENSE](LICENSE)。
