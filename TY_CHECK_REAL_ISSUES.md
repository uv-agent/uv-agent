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

- [ ] `src/uv_agent/model/types.py` / `src/uv_agent/agent/engine.py`：`ModelClient.stream_response` 协议应表达“调用后直接得到 async iterator”，否则 async generator 实现会被误判为 coroutine。
- [ ] `src/uv_agent_runtime/patch.py`：`_read_pending` 用裸 `object` sentinel 导致 `text` 无法被可靠收窄；补充显式 sentinel 类型/读写 helpers，避免 `_apply_hunks` 收到非 `str`。
- [ ] `src/uv_agent_runtime/textops.py`：`Popen` helper 泛型、换行 Literal 收窄需要整理，移除不必要的 ignore。
- [ ] `src/uv_agent/runner/scriptenv.py`：显式导入 `TOMLKitError`，避免依赖 `tomlkit.exceptions` 作为动态属性。
- [ ] `src/uv_agent/config.py`：对原始配置层做字典收窄，避免非对象配置项导致 `dict(value)`/`**raw` 在坏配置下崩溃；同时让 passthrough/reasoning merge 更明确。
- [ ] `src/uv_agent/model/content.py`：`pending_assistant["tool_calls"]` 应在追加前确保是 list，避免用户 passthrough 字段碰撞时 `.append` 崩溃。
- [ ] `src/uv_agent/model/sdk.py`：`object_dump` 对可迭代对象直接 `dict(value)` 可能抛异常；应保持 best-effort。
- [ ] `src/uv_agent/notifications.py`：终端 bell stream 用 Protocol 表达实际需求，替代对 `object` 的动态调用。
- [ ] `src/uv_agent/mcp_probe.py`：MCP transport 做显式 Literal 收窄，移除 ignore。
- [ ] `src/uv_agent/tui/app.py` / `widgets.py` / `panels.py`：少量可读的真实收窄（`before` 类型、severity Literal、delay/item guard、watcher 参数名、队列返回类型）可以修复；大量 Textual action handler 推断噪声暂不处理。

### 暂不处理 / 观察项

- [ ] `anyio.to_thread.run_sync`：当前依赖版本中存在该 API，经验证不是运行时问题；暂不改成其他 API。
- [ ] `Path.read_text(newline="")`：当前项目运行 Python 3.14，`pathlib.Path.read_text/write_text` 支持 `newline`；项目声明 `>=3.12` 时需要关注旧版本兼容，若未来确认 3.12 不支持再改为 `open(..., newline="")`。
- [ ] `tui/config_panels.py` 大量 `Self@...`：Textual action handler / mixin 推断局限，暂不做大面积改动。
- [ ] 测试文件中的 `pytest` unresolved-import：这是检查环境未安装 dev 依赖导致，不是源码问题。
- [ ] 测试 fake client/override 签名噪声：多数由协议根问题或测试动态 fake 引起，先不批量改。
