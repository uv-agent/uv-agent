# uv-agent

[English](README.md)

`uv-agent` 是一个可移植的 coding agent，提供 Textual TUI。它对外只有一个动作面：
`run_python`：模型把 Python 脚本提交给受管理的 `uv run` runner，再由脚本完成实际工作。
这种单一工具面的设计让 agent 在不同平台上的行为保持一致——任何安装了 Python 和 uv
的操作系统都能获得相同的体验。

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

## 核心思路

- agent 对外只有一个动作面：`run_python`。
- 受管理脚本运行在项目共享 uv 环境中；第三方依赖通过
  `add_dependency` 添加到这个环境。
- 发布包同时包含 `uv_agent` 和 `uv_agent_runtime`；受管理脚本从
  `uv_agent_runtime` 导入快捷 helper。
- workspace rules、skills 和 MCP declarations 作为上下文渐进披露。MCP 调用通过
  Python runtime helper 完成，不直接暴露成模型工具。
- thread 状态、run 日志、共享脚本环境和附件位于
  `~/.uv-agent/projects/<project-id>/`。

## 开发

```powershell
uv run pytest
```

本地调试状态、截图、配置、脚本、运行记录和 thread 数据应放在 `.uv-agent/`，不要提交。

## 许可证

MIT。见 [LICENSE](LICENSE)。
