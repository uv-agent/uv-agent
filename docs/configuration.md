# Configuration

`uv-agent` reads JSON configuration from a small set of layered files. The
package does not ship a real provider configuration, so model calls require at
least one provider, one model, and one level.

## File Locations

Config is loaded in this order:

1. Built-in defaults.
2. User config: `~/.uv-agent/config.json`.
3. Project config: `.uv-agent/config.json` under the current project.
4. Extra config from `UV_AGENT_CONFIG`, when the environment variable is set.

Later layers override earlier layers. The project `.uv-agent/` directory is
intended for local state and ignored by git.

Runtime state is stored separately under:

```text
~/.uv-agent/projects/<project-id>/
```

That directory contains thread JSONL, run JSONL, saved managed scripts, and
attachments.

## Provider Options

Providers describe HTTP endpoints and authentication.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `base_url` | string | required | Base URL used with endpoint `path`. |
| `api_key` | string | `null` | Direct bearer token. Prefer `api_key_env` for secrets. |
| `api_key_env` | string | `null` | Environment variable that contains the bearer token. |
| `headers` | object | `{}` | Static HTTP headers added to every request. |
| `params` | object | `{}` | JSON payload fields shared by all model requests for this provider. |
| `responses` | endpoint | `{ "path": "/responses" }` | Endpoint config for the Responses-style API. |
| `chat_completions` | endpoint | `{ "path": "/chat/completions" }` | Endpoint config for the Chat Completions-style API. |
| `anthropic_messages` | endpoint | `{ "path": "/v1/messages" }` | Endpoint config for the Anthropic Messages-style API. |

Endpoint config shape:

```json
{
  "path": "/responses",
  "params": {}
}
```

`path` is appended to `base_url` with no extra URL rewriting. For example,
`base_url: "https://api.example.com/v1"` and `path: "/responses"` becomes
`https://api.example.com/v1/responses`.

Secrets are redacted in config display paths, but committed config should still
avoid direct `api_key` values.

## Model Options

Models bind a provider to a concrete remote model name and API format.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `provider` | string | required | Provider name from `providers`. |
| `model` | string | required | Remote model identifier sent to the provider. |
| `api` | string | `"responses"` | One of `responses`, `chat_completions`, or `anthropic_messages`. |
| `context_window_tokens` | integer | `128000` | Context size used by the local context meter. |
| `supports_images` | boolean or null | `null` | Set `false` to block image attachments for this model. |
| `params` | object | `{}` | JSON payload fields for this model. |

Payload params are merged from provider, endpoint, model, and level settings.
Later layers win when the same key appears.

## Level Options

Levels are named runtime choices. The TUI and `uv-agent ask --level <name>` use
level names rather than concrete model names.

```json
{
  "levels": {
    "small": { "model": "fast" },
    "medium": { "model": "main" },
    "large": {
      "model": "main",
      "params": {
        "reasoning": { "effort": "high" }
      }
    }
  }
}
```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `model` | string | required | Model name from `models`. |
| `params` | object | `{}` | Level-specific payload params merged into the model params. |

## Runtime Options

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `default_level` | string | `"medium"` | Level used when no explicit level is selected. |
| `auto_compress` | boolean | `true` | Automatically compress thread context near the trigger threshold. |
| `store_provider_response` | boolean | `false` | Store raw provider responses in thread state. |
| `max_agent_rounds` | integer | `100` | Maximum model/tool loop rounds for one turn. |
| `compression` | object | see below | Context compression settings. |
| `title_generation` | object | see below | Thread title generation settings. |

Compression options:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `model_level` | string or null | `null` | Optional level used for compression. `null` uses the active/default level. |
| `prompt` | string | built in | Prompt used to summarize context. |
| `trigger_ratio` | number | `0.7` | Compress when estimated context usage reaches this ratio. |
| `target_ratio` | number | `0.3` | Target ratio after compression. |
| `min_tokens` | integer | `5000` | Do not compress below this estimated token count. |

Title generation options:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | boolean | `true` | Generate a short title for new threads. |
| `model_level` | string or null | `null` | Optional level used for title generation. |
| `prompt` | string | built in | Prompt used for title generation. |

## UI Options

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `language` | string | `"auto"` | TUI language. Use `auto`, `en`, or `zh-CN`. |

The `/config` panel can edit `runtime.default_level`, `runtime.auto_compress`,
and `ui.language`. Model, provider, and level definitions are edited in JSON.

## Runner Options

Runner settings control managed Python script execution.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `runtime_dependency` | string | installed `uv-agent` package | Dependency spec injected into managed scripts so they can import `uv_agent_runtime`. |
| `runtime_package_name` | string | `"uv-agent"` | Package name used when deriving default runner behavior. |
| `default_uv_args` | array | `[]` or local reinstall args | Extra args passed to `uv run` for every managed script. |
| `default_timeout_s` | number | `60` | Default script timeout in seconds. |
| `max_output_bytes` | integer | `1000000` | Maximum captured output bytes per run. |
| `max_saved_scripts` | integer | `32` | Number of recently used managed scripts kept per project. |

For local source development, point `runtime_dependency` at the checkout:

```json
{
  "runner": {
    "runtime_dependency": "uv-agent @ file:///C:/path/to/uv-agent"
  }
}
```

When a local file dependency is used, `uv-agent` automatically adds
`--reinstall-package uv-agent` unless `default_uv_args` is explicitly set.

## Full Example

See [config.example.json](config.example.json).
