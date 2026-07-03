# uv-agent

<img align="right" src="docs/t2.png" alt="uv-agent tui screenshot" width="300">

[English](README.md)

**Python-native coding agent——单一 `run_python` 边界，所有外部动作可审查、可重放、可中断。**

`uv-agent` 把模型的能力收敛到一个边界清晰的出口：模型只能通过 `run_python` 触碰外部世界。
每次调用都是一段完整的 Python 脚本，在 `uv run` 管理的隔离环境中执行，通过
`uv_agent_runtime` 提供的 helper 完成文件编辑、命令执行、代码搜索、图片，以及 MCP client、
workflow 图等插件提供的能力。因为只有这一个出口，你可以回放任何一次 run，看清每一步做了什么、为什么这么做。

项目仍在实验阶段，公开 API、配置字段和 runtime 行为可能继续变化。

## 特点

- **单一工具边界** — 没有 shell、文件系统、浏览器、MCP 等模型直接工具。模型只写 Python，
  由托管运行时执行。每一次外部动作都是可审计的脚本。
- **缓存感知 NetGain 压缩** — 长对话不再盲目压缩。回合前轻量级 judge 轮次让模型估算
  剩余调用次数和历史依赖程度，通过经济学公式计算压缩的净收益，仅在"压缩省下的缓存费用 >
  丢失上下文的信息损失"时才触发。压缩时保留最近 K token 原文，避免摘要丢失关键信息。
- **Python 托管运行时** — 脚本运行在项目共享的 `uv` 环境中，`uv_agent_runtime` 提供
  read/write/edit、FFF 原生搜索、子进程、依赖安装、图片和插件注册的 runtime namespace。
  脚本即文档——不像 bash 命令那样晦涩难读。
- **插件体系** — 可信 Python 包通过 `uv_agent.plugins` entry point 发现，可增加
  runtime namespace（`rt.mcp`、`rt.workflow`、自定义 helper）、action、斜杠命令、
  UI provider、模型上下文、事件订阅、持久存储，以及从外部系统提交 turn，同时保持模型只有
  `run_python` 一个动作边界。
- **无界面服务模式** — `uv-agent daemon` 不打开 TUI，只启动 host 和插件，适合
  scheduler、聊天桥接、webhook、后台事件转发等需要长期运行的项目级集成。
- **自举开发** — uv-agent 用 uv-agent 开发自身。项目日常的阅读、修改、测试、迭代都由
  uv-agent 完成。
- **渐进式上下文披露** — skills、MCP server、workspace rules 不是一次性塞进 prompt。
  模型先收到索引，需要时才展开完整内容；能力消失时显式标记移除，避免模型依赖过期上下文。
- **Goal mode 持久记忆** — `/goal` 为长任务建立独立于聊天记录的 checklist/notes 层。
  压缩或恢复后，模型优先参考 goal 插件状态而非只能依赖摘要后的聊天历史。
- **Prompt 缓存友好设计** — 系统 prompt 前缀在 epoch 内保证字节级不变；压缩请求结构与
  正常调用保持一致，最大化 provider 端缓存命中。缓存读几乎免费。

## 缓存感知压缩

v0.16.0 引入的 cache-aware compaction 是 uv-agent 在长对话场景下的核心优化。与传统的
"上下文满 N% 就压缩"不同，uv-agent 在每回合开始前做一次经济学判断：

1. 模型估算当前任务还需要多少轮对话（`remaining_calls_bucket`）以及对历史上下文的依赖
   程度（`history_dependency`）。
2. 枚举 K 个候选保留量，用 NetGain 公式计算每种选择的净收益：后续节省的缓存费用
   减去压缩调用成本、缓存失效损失、信息失真惩罚，加上上下文质量改善收益。
3. 仅当最佳净收益超过阈值时才触发压缩；否则跳过，避免为短期任务浪费压缩开销。

压缩请求与正常模型调用共享完全相同的前缀结构（system prompt → tools → messages），
确保 provider 端 prompt 前缀缓存持续命中。一次典型压缩调用的输入 token 中 90%+ 走
缓存价格（约为正常输入价格的 1%~10%）。

这一设计参考了 [bash-agent](https://github.com/lloydzhou/bash-agent) 的 DP 压缩算法，
在此致谢。

## 快速开始

需要先安装：

- **uv** — https://docs.astral.sh/uv/getting-started/installation/
- **Git** — 常规编码工作流和 Worktree mode 需要。

```powershell
# 运行最新发布版本
uvx uv-agent@latest

# 从本地源码运行
uv run uv-agent

# 单轮问答（不打开 TUI）
uvx uv-agent@latest ask "帮我看看这个项目的结构"

# 继续已有会话
uvx uv-agent@latest ask --thread thr_xxx "继续刚才的任务"

# 启动无界面服务 host，供插件和 scheduler 长期运行
uvx uv-agent@latest daemon --replace
```

## 模型配置

uv-agent 不内置 provider 配置。在 `~/.uv-agent/config.json`（或项目级
`.uv-agent/config.json`）中配置至少一个 provider、model 和 level。API key 建议
放在环境变量或被 git 忽略的本地配置文件中。

支持的 API 格式：

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
    "max_output_bytes": 1000000,
    "max_run_logs": 200,
    "scriptenv_index_url": null
  },
  "logging": {
    "level": "INFO",
    "file_enabled": true,
    "console_enabled": false,
    "max_bytes": 5000000,
    "backup_count": 3
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
    "my-plugin": {
      "enabled": false
    },
    "another-plugin": {
      "enabled": true,
      "config": {
        "option": "value"
      }
    }
  }
}

```

</details>

TUI 内用 `/config` 可切换默认 level、界面语言、压缩策略等。完整配置项见
[configuration](docs/configuration.md)，示例见 [config.example.json](docs/config.example.json)。

## 日志

uv-agent 会把运行日志写到 project state 的日志目录，通常是
`~/.uv-agent/projects/<project-id>/log/uv-agent.log`。每个插件的独立日志位于
`~/.uv-agent/plugins/<plugin-id>/logs/plugin.log`。主日志和插件日志都使用顶层
`logging.max_bytes` 与 `logging.backup_count` 轮转配置；默认每个当前日志约 5 MB，
保留 3 个备份。`--log-level` 会覆盖当前进程的 `logging.level`。

## 日常工作流

- 输入后 `Enter` 发送；`Ctrl+Enter` / `Ctrl+J` 在输入框中换行。
- 空输入框按 `/` 打开命令面板，输入关键词过滤。`@` 引用文件，`@@` 引用历史会话。
- `/level <name>` 切换模型；`/status` 查看运行时状态（含缓存压缩判定详情）。
- `/goal enable [目标]` 开启持久任务清单，适合跨多轮的长任务。

完整命令见 [TUI and slash commands](docs/tui.md)。

## TUI 界面

- **tui**（默认，`uv-agent` 或 `uv-agent tui`）— 轻量 ANSI TUI，直接渲染在终端中。
  紧凑状态行、流式事件、Goal/Worktree mode、图片附件。

## 服务模式

`uv-agent daemon` 运行与 TUI 相同的 host/plugin 栈，但不打开终端界面。适合需要长期
驻留的插件能力：定时 action、外部聊天或 webhook 桥接、从外部系统提交 turn、后台事件转发。

daemon 会在 project state 中获取 lease 并定期 heartbeat，确保同一工作区只有一个活跃
host 负责这些集成。需要替换旧实例或清理 stale lease 时使用 `--replace`。

```powershell
uv-agent daemon --replace
```

## 插件

插件是通过 `uv_agent.plugins` entry point 发现并加载到 uv-agent host 进程中的可信
Python 包。它们扩展 host，但不改变模型边界：模型仍然只能通过 `run_python` 行动，插件能力
则体现为脚本 helper、命令、action、UI 扩展和结构化模型上下文。

插件可以：

- 暴露 `rt.mcp`、`rt.workflow` 或项目自定义的 `rt.<name>` runtime helper namespace；
- 注册供 scheduler 和自动化流程调用的 action；
- 添加斜杠命令、picker/UI provider 和本地化文本；
- 订阅 host 事件、维护私有存储，并从外部系统提交 turn。

内置的 Goal、Worktree、Skills、MCP、Workflow、Scheduler 能力也使用同一套插件接口。
请只安装你信任的插件。

```powershell
uvx --with your-uv-agent-plugin uv-agent@latest
```

详见 [Plugin system](docs/plugins.md)。

## Runtime 与上下文

模型每轮看到的输入 = 稳定 system prompt + 按需注入的结构化上下文。

- `run_python` 是唯一对外动作面。脚本在项目共享 uv 环境中执行，导入 `uv_agent_runtime`
  helper。uv 环境与工作目录独立；cwd 可随 `enter_dir` 或 Worktree mode 变化。
- runtime 上下文（helper 列表、skills、MCP server 等）由插件发布 epoch context；
  插件在刷新或压缩后重新发布完整上下文，并可在自身状态变化时显式发送 XML update。
- workspace rules 渐进披露：先索引、进入目录后才展开完整 AGENTS.md。
- Goal mode 提供独立于聊天记录的 checklist/notes 持久层，压缩后仍可恢复任务进度。
- checkpoint 压缩总结对话，排除可重载的 runtime 上下文，新 epoch 先重放结构化上下文
  再接入保留历史。

## 文档

- [Configuration](docs/configuration.md)
- [Full config example](docs/config.example.json)
- [TUI and slash commands](docs/tui.md)
- [Runtime and managed scripts](docs/runtime.md)
- [Plugin system](docs/plugins.md)

## 开发

uv-agent 以自举方式开发——用 uv-agent 自身完成阅读、修改、测试和迭代。

```powershell
uv run pytest
```

本地调试状态、截图、配置、运行记录放在 `.uv-agent/`，不提交。

## 致谢

缓存感知压缩的设计参考了 [bash-agent](https://github.com/lloydzhou/bash-agent)
的动态规划压缩算法及其缓存对齐思想，特此感谢。

## 许可证

MIT。见 [LICENSE](LICENSE)。
