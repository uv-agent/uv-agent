# uv-agent

[English](README.md)

`uv-agent` 是一个可移植的 coding agent，提供 Textual TUI。它对外只有一个动作面：
`run_python`：模型把 Python 脚本提交给受管理的 `uv run` runner，再由脚本完成实际工作。
这种单一工具面的设计让 agent 在不同平台上的行为保持一致——任何安装了 Python 和 uv
的操作系统都能获得相同的体验。

公开 API、配置字段和 runtime 行为可能随项目演进而继续调整。

## 安装与运行

> **前置要求**  
> 系统上需要安装
> [**uv**](https://docs.astral.sh/uv/getting-started/installation/) 和
> [**ripgrep**](https://github.com/BurntSushi/ripgrep#installation)。
> uv 是用于运行 agent 的 Python 包与项目管理器；ripgrep 用于在工作区内快速搜索文件内容。

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

最小配置结构：

```json
{
  "providers": {
    "main": {
      "base_url": "https://api.example.com/v1",
      "api_key_env": "UV_AGENT_API_KEY",
      "responses": { "path": "/responses" }
    }
  },
  "models": {
    "main": {
      "provider": "main",
      "model": "your-model-name",
      "api": "responses",
      "context_window_tokens": 128000
    }
  },
  "levels": {
    "medium": { "model": "main" }
  },
  "runtime": {
    "default_level": "medium"
  },
  "ui": {
    "language": "auto",
    "completion_notification": {
      "enabled": true,
      "terminal": true,
      "bell": true
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
- 受管理脚本用 PEP 723 inline metadata 声明第三方依赖。
- 发布包同时包含 `uv_agent` 和 `uv_agent_runtime`；受管理脚本从
  `uv_agent_runtime` 导入快捷 helper。
- workspace rules、skills 和 MCP declarations 作为上下文渐进披露。MCP 调用通过
  Python runtime helper 完成，不直接暴露成模型工具。
- thread 状态、run 日志、保存的脚本和附件位于
  `~/.uv-agent/projects/<project-id>/`。

## 开发

```powershell
uv run pytest
```

本地调试状态、截图、配置、脚本、运行记录和 thread 数据应放在 `.uv-agent/`，不要提交。

## 许可证

MIT。见 [LICENSE](LICENSE)。
