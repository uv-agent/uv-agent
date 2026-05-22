# ty check 真实问题排查与修复清单

## 背景

`docs/pending-ty-check-errors.md` 记录了一次 `uvx ty check` 的过滤结果。由于 `ty` 仍处于实验阶段，部分诊断来自类型推断局限、第三方库 stub 不精确，或测试 fake/框架回调的动态用法，并不一定对应运行时问题。

本分支的目标不是“把所有 ty 输出清零”，而是逐条识别其中更可能导致运行时崩溃、协议误解或真实维护风险的问题，并用小步提交修复。明显属于 Textual/ty 推断局限、测试类型标注噪声、或已经由运行时守卫保证安全的项目，暂不改动，避免引入无意义的类型体操。

## 处理原则

1. 优先修复会造成运行时 crash 或行为错误的问题。
2. 对大量连锁报错，先修正根协议/根类型，而不是批量修改派生调用点。
3. 对“仅 ty 看不懂”的动态框架用法，除非可以通过更清晰的类型收窄改善可读性，否则暂不改。
4. 每组相关修复独立提交，便于回滚和审查。
5. 修改后运行聚焦测试；最终运行 `uv run pytest`。

## 排查清单

### 需要修复

- [x] `src/uv_agent/model/types.py` / `src/uv_agent/agent/engine.py`：`ModelClient.stream_response` 协议应表达“调用后直接得到 async iterator”，否则 async generator 实现会被误判为 coroutine。
- [x] `src/uv_agent_runtime/patch.py`：`_read_pending` 用裸 `object` sentinel 导致 `text` 无法被可靠收窄；补充显式 sentinel 类型/读写 helpers，避免 `_apply_hunks` 收到非 `str`。
- [x] `src/uv_agent_runtime/textops.py`：`Popen` helper 泛型、换行 Literal 收窄需要整理，移除不必要的 ignore。
- [x] `src/uv_agent/runner/scriptenv.py`：显式导入 `TOMLKitError`，避免依赖 `tomlkit.exceptions` 作为动态属性。
- [x] `src/uv_agent/config.py`：对原始配置层做字典收窄，避免非对象配置项导致 `dict(value)`/`**raw` 在坏配置下崩溃；同时让 passthrough/reasoning merge 更明确。
- [x] `src/uv_agent/model/content.py`：`pending_assistant["tool_calls"]` 应在追加前确保是 list，避免用户 passthrough 字段碰撞时 `.append` 崩溃。
- [x] `src/uv_agent/model/sdk.py`：`object_dump` 对可迭代对象直接 `dict(value)` 可能抛异常；应保持 best-effort。
- [x] `src/uv_agent/notifications.py`：终端 bell stream 用 Protocol 表达实际需求，替代对 `object` 的动态调用。
- [x] `src/uv_agent/mcp_probe.py`：MCP transport 做显式 Literal 收窄，移除 ignore。
- [x] `src/uv_agent/tui/app.py` / `widgets.py` / `panels.py`：少量可读的真实收窄（`before` 类型、severity Literal、delay/item guard、watcher 参数名、队列返回类型）可以修复；大量 Textual action handler 推断噪声暂不处理。

### 暂不处理 / 观察项

- [x] `anyio.to_thread.run_sync`：当前依赖版本中存在该 API，经验证不是运行时问题；暂不改成其他 API。
- [x] `Path.read_text(newline="")`：当前项目运行 Python 3.14，`pathlib.Path.read_text/write_text` 支持 `newline`；项目声明 `>=3.12` 时需要关注旧版本兼容，若未来确认 3.12 不支持再改为 `open(..., newline="")`。
- [x] `tui/config_panels.py` 大量 `Self@...`：Textual action handler / mixin 推断局限，暂不做大面积改动。
- [x] 测试文件中的 `pytest` unresolved-import：这是检查环境未安装 dev 依赖导致，不是源码问题。
- [x] 测试 fake client/override 签名噪声：多数由协议根问题或测试动态 fake 引起，先不批量改。


## 本分支实际修复摘要

- 修正 `ModelClient.stream_response` 协议为同步返回 `AsyncIterator`，匹配现有 async generator 实现和引擎调用方式。
- 为 runtime patch sentinel、subprocess kill helpers、换行风格 helpers、scriptenv TOML 异常导入增加明确类型收窄。
- 加固配置解析：仅把 JSON object 当作配置对象； endpoint 支持字符串 shorthand、拒绝未知 endpoint 字段，并避免坏嵌套值触发无意义崩溃。
- 加固模型数据转换：assistant `tool_calls` passthrough 碰撞时恢复为合法列表；`object_dump` 对非 mapping iterable 保持 best-effort。
- 用小协议表达通知 stream 能力；MCP probe transport 显式规范化为 Literal。
- 对 TUI 的 mount `before`、重放 item、通知 severity、selection 写入、pending queue、scroll watcher 做局部收窄；Textual action handler 推断噪声仍按计划不做大改。

## 验证记录

- `uv run pytest tests/test_agent.py tests/test_model_client.py -q`：105 passed
- `uv run pytest tests/test_runtime.py tests/test_runner.py -q`：71 passed
- `uv run pytest tests/test_config.py tests/test_model_client.py tests/test_notifications.py tests/test_mcp_probe.py -q`：59 passed（后续配置补丁后相关子集 54 passed）
- `uv run pytest tests/test_tui.py -q`：109 passed
- `uv run pytest`：343 passed
