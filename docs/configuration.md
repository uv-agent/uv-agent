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

That directory contains the project SQLite state database (`uv-agent.sqlite3`),
exported managed scripts, the shared script environment, and attachments.

## Provider Options

Providers describe HTTP endpoints and authentication.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `base_url` | string | required | Base URL used with endpoint `path`. |
| `api_key` | string | `null` | Direct bearer token. Prefer `api_key_env` for secrets. |
| `api_key_env` | string | `null` | Environment variable that contains the bearer token. |
| `headers` | object | `{}` | Extra SDK default headers passed on a best-effort basis. |
| `timeout_s` | number or null | `7200` | Provider SDK request timeout in seconds. Set `null` to keep the SDK default. |
| `params` | object | `{}` | JSON payload fields shared by all model requests for this provider. |
| `message_passthrough` | object | `{}` | Chat message fields to persist and replay for provider-specific APIs. Models inherit this unless they override fields. |
| `reasoning_display` | object | `{}` | Provider-specific fields that should be shown as reasoning in the TUI. Models inherit this unless they override fields. |
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

Model requests use the official OpenAI and Anthropic SDKs. Endpoint `path`
exists so existing configs can describe the target API shape, but SDK-backed
requests strip the SDK-owned suffix from `base_url`: `/responses`,
`/chat/completions`, and `/v1/messages` are owned by the corresponding SDK
method. For example, `base_url: "https://api.example.com/v1"` and `path:
"/responses"` creates an OpenAI SDK client with base URL
`https://api.example.com/v1`.

Secrets are redacted in config display paths, but committed config should still
avoid direct `api_key` values.

SDK credential behavior is preserved. `api_key` and `api_key_env` are passed to
the provider SDK when configured; otherwise the SDK may use its own environment
variables or raise its normal missing-credentials error. `headers` can add
provider-specific headers, but it is not a replacement for SDK credential
configuration unless that provider SDK accepts the header itself.

Provider `message_passthrough` and `reasoning_display` are defaults for every
model that uses the provider. A model can override individual fields while
inheriting the rest.

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
| `message_passthrough` | object | provider default | Chat message fields to persist and replay for this model. |
| `reasoning_display` | object | provider default | Provider-specific fields that should be shown as reasoning for this model. |

Request params are merged from provider, endpoint, model, and level settings.
Later layers win when the same key appears. Params accepted by the SDK method
are passed as normal keyword arguments. Unknown params are merged into SDK
`extra_body`; an explicit `extra_body` object in params is merged there too.
`message_passthrough` and `reasoning_display` are inherited from provider to
model; model config overrides only the fields it names.

`message_passthrough` shape:

```json
{
  "assistant": ["reasoning_content"],
  "user": [],
  "system": [],
  "tool": []
}
```

Configured fields are copied from provider responses into stored assistant
message items and replayed into later Chat Completions requests. This is useful
for OpenAI-compatible providers that require vendor fields such as
`reasoning_content` to be sent back with previous assistant messages.

`reasoning_display` shape:

```json
{
  "assistant_message_fields": ["reasoning_content"],
  "stream_delta_fields": ["reasoning_content"],
  "unknown_text_delta_as_reasoning": false
}
```

These settings only affect what uv-agent displays and stores as reasoning. They
do not cause fields to be replayed unless the same fields are also listed under
`message_passthrough`. `unknown_text_delta_as_reasoning` is a fallback for
third-party Chat Completions streams: when enabled, string delta fields that are
not normal content, tool calls, or known control fields are accumulated as
reasoning.

Example for a Mimo-style OpenAI-compatible provider:

```json
{
  "providers": {
    "mimo": {
      "base_url": "https://api.xiaomimimo.com/v1",
      "headers": {
        "api-key": "set-this-in-your-untracked-user-config"
      },
      "message_passthrough": {
        "assistant": ["reasoning_content"]
      },
      "reasoning_display": {
        "assistant_message_fields": ["reasoning_content"],
        "stream_delta_fields": ["reasoning_content"]
      }
    }
  },
  "models": {
    "mimo-main": {
      "provider": "mimo",
      "model": "mimo",
      "api": "chat_completions"
    }
  }
}
```

`headers` values are static JSON strings. Keep custom provider keys in your
untracked user or project config file.

## Pricing Options

Pricing is optional. When configured, uv-agent computes an incremental charge
from provider-reported usage after every model call and stores only the running
thread total. The footer shows the current thread total as a right-side compact
amount with 4 decimal places; `/status` shows the same total with 6 decimals.
If pricing is absent, the UI hides billing.

Prices are configured per model and default to the vendor-standard unit of one
million tokens:

```json
{
  "pricing": {
    "currency": "USD",
    "unit": "1M_tokens",
    "models": {
      "main": {
        "input": 2.0,
        "output": 8.0,
        "cached_input": 0.5,
        "unit": null
      }
    }
  }
}
```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `input` | number | `0.0` | Price per unit for input tokens. |
| `output` | number | `0.0` | Price per unit for output tokens. |
| `cached_input` | number | `0.0` | Price per unit for cached input tokens. |
| `unit` | string or null | `null` | Optional per-model unit override. Falls back to the top-level `unit`. |

The model key normally matches the local model name under `models`. The remote
provider model id and level name are also accepted as fallbacks. Supported
currency symbols are `USD`/`$` and `CNY`/`RMB`/`¥`; unknown currency codes are
displayed before the amount (for example `EUR 0.1234`).

Billable input is calculated as non-cached input plus cached input plus output.
For OpenAI-compatible usage with `*_tokens_details.cached_tokens`, cached tokens
are subtracted from input and charged at `cached_input`. Anthropic-style
`cache_read_input_tokens` are charged as cached input; `cache_creation_input_tokens`
are charged as ordinary input.

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
| `store_provider_response` | boolean | `false` | Store raw provider responses in thread state. |
| `max_agent_rounds` | integer | `100` | Maximum model/tool loop rounds for one turn. |
| `compression` | object | see below | Context compression settings. |
| `title_generation` | object | see below | Thread title generation settings. |
| `stream_retry` | object | see below | Streaming retry configuration. |

Compression options:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | boolean | `true` | Automatically compress thread context near the trigger threshold. |
| `model_level` | string or null | `null` | Optional level used to **perform** the compression summary. `null` uses the active/default level. The trigger threshold is always computed against the **active turn's** model context window, not this level's. |
| `trigger_ratio` | number | `0.7` | Compress when estimated context usage reaches this ratio of the active turn's model context window. Also serves as the mid-turn safety-net threshold (Path B). |
| `min_tokens` | integer | `5000` | Do not compress below this estimated token count. |
| `cache_aware` | boolean | `false` | Enable cache-aware NetGain pre-turn judge compaction (Path A). When off, only the threshold-based trigger (Path B) runs. |
| `margin` | number | `1.5` | Safety margin multiplier for the NetGain formula: compression only fires when `NetGain > max(MinGain, CompactCost * Margin)`. |
| `min_gain` | number | `0.0001` | Minimum net gain (in `pricing.currency`) required to trigger cache-aware compaction. |
| `judge_model_level` | string or null | `null` | Optional level for the pre-turn judge call. `null` reuses `model_level` (or the active level if that is also null). |
| `judge_min_context_ratio` | number | `0.20` | Skip the judge round when estimated context tokens are below this ratio of the active model's context window. |

Compression follows two independent paths:

**Path A – Cache-aware pre-turn judge** (`cache_aware: true`).  Before each
turn a lightweight judge round asks the model for two semantic parameters:
`remaining_calls_bucket` (projected future rounds) and `history_dependency`
(low / medium / high / exact).  A local NetGain formula then computes
whether the estimated future cache-read savings outweigh the cost of
generating a summary, accounting for the cache-rebuild penalty on the
retained recent context.  The judge round appears briefly in the TUI status
bar ("Judging / 判断中") but does not add a transcript cell.

**Path B – Threshold-triggered safety net.**  When `cache_aware` is off, or
during mid-turn tool loops regardless of the judge outcome, compaction
fires when the token count reaches `trigger_ratio` of the context window.
Mid-turn compaction retains the most recent 25% of the context window
verbatim alongside the summary.

Title generation options:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | boolean | `true` | Generate a short title for new threads. |
| `model_level` | string or null | `null` | Optional level used for title generation. |

Stream retry options control how the client retries failed streaming
requests with exponential backoff:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `max_retries` | integer | `5` | Maximum number of retries for a failed stream request. |
| `base` | number | `1.0` | Base delay in seconds before the first retry. |
| `factor` | number | `2.0` | Exponential backoff multiplier. |
| `max` | number | `30.0` | Maximum delay in seconds between retries. |
| `jitter` | number | `0.2` | Random jitter factor applied to each retry delay. |

## UI Options

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `language` | string | `"auto"` | TUI language. Use `auto`, `en`, or `zh-CN`. |
| `completion_notification` | object or boolean | see below | Notify when a TUI turn finishes. |

Completion notification options:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | boolean | `true` | Enable completion notifications. |
| `terminal` | boolean | `true` | Add a short transcript event when a background thread finishes. |
| `bell` | boolean | `true` | Play a completion sound when a turn finishes. The terminal UI uses a short buzzer-like cue when available; other platforms write a terminal BEL. |

The `/config` panel can edit `runtime.default_level`,
`runtime.compression.enabled`, `ui.language`, and
`ui.completion_notification.enabled`. Model, provider, level, and plugin settings
are edited in JSON.

## Plugin Options

Plugins are installed Python packages discovered through the `uv_agent.plugins`
entry point group. They are enabled by default unless explicitly disabled.
Enabled plugins may still be skipped when their manifest activation policy does
not match the current host lifecycle.

```json
{
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

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | boolean | `true` | Whether the plugin is loaded. |
| `config` | object | `{}` | Plugin-owned config object passed to `PluginContext.config`. |

See [Plugin system](plugins.md) for the plugin API and examples.

## Runner Options

Runner settings control managed Python script execution.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `default_timeout_s` | number | `7200` | Default script timeout in seconds. |
| `max_output_bytes` | integer | `1000000` | Maximum captured output bytes per run. |
| `max_run_logs` | integer | `200` | Number of recent run records kept per project. Matching exported debug scripts are pruned with old rows. |
| `scriptenv_index_url` | string or null | `null` | Optional uv default package index URL written to the managed `runner/scriptenv/pyproject.toml`. |

## Logging Options

Operational logs use Python's standard logging system under the `uv_agent`
namespace. The main project log is written to:

```text
~/.uv-agent/projects/<project-id>/log/uv-agent.log
```

Plugin logs are written separately under:

```text
~/.uv-agent/plugins/<plugin-id>/logs/plugin.log
```

The main log and per-plugin `plugin.log` files use the same rotation settings.
With the defaults, each active file is capped at about 5 MB and up to three
rotated backups are retained.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `level` | string or integer | `"INFO"` | Logging level for `uv_agent` and plugin loggers. CLI `--log-level` overrides this for the current process. |
| `file_enabled` | boolean | `true` | Write the main project log file. |
| `console_enabled` | boolean | `false` | Also write logs to stderr. The TUI keeps this off by default to avoid corrupting terminal rendering. |
| `max_bytes` | integer | `5000000` | Maximum bytes per active log file before rotation. Set `0` to disable rotation. |
| `backup_count` | integer | `3` | Number of rotated backup files retained per log. |

## Full Example

See [config.example.json](config.example.json).
