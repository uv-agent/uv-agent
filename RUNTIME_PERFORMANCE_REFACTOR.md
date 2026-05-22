# Runtime 性能重构清单

## 背景

`src/uv_agent_runtime/` 是每次 `run_python` 脚本都会接触的辅助包。这里的性能问题会直接影响工具冷启动、代码搜索、补丁 dry-run、符号查询等高频操作。前一次通用性能审计侧重主应用流程；本清单仅跟踪本次 runtime helper 的无感重构计划。

本次重构要求：

- 对用户无感：保持公开 API、返回结构和错误语义兼容。
- 对模型无感：不改变 prompt 中已公布的 helper 名称和调用方式。
- 奥卡姆剃刀：只处理已经确认的主要性能瓶颈，不做无关功能扩展。
- 渐进式提交：每个提交聚焦一个小目标，便于回滚和审查。

## 目标

1. 降低 `from uv_agent_runtime import ...` 的冷启动导入成本。
2. 让 `search_text(max_total=...)` 和 `find_files(max_total=...)` 在达到上限时真正停止 ripgrep。
3. 让 `apply_patch_any(dry_run=True)` 不再 snapshot/restore 整个仓库。
4. 修正 `codequery` 在不同语言、glob、单文件查询之间互相清缓存的问题，并避免全量查询时的无意义 cache churn。
5. 保持现有测试通过，并为关键兼容行为补充小测试。

## 计划

### 1. runtime 顶层导入改为 lazy export

现状：`uv_agent_runtime.__init__` eager import 了 MCP SDK、tree-sitter、unidiff 等重模块。即使只导入 `enter_dir`，也会加载全部子模块。

方案：

- 保留 `__all__` 中的所有公开名称。
- 使用 `__getattr__` 按名称延迟导入对应子模块。
- 对 `from uv_agent_runtime import codequery` 这类子模块导入保持兼容。
- 不改变任何 helper 的调用方式。

验收：

- 现有 runtime/runner 测试通过。
- `from uv_agent_runtime import enter_dir` 不触发 MCP SDK 导入。

### 2. codesearch 上限改为流式提前终止

现状：`search_text` / `find_files` 先 `subprocess.run(..., capture_output=True)` 全量收集，再在 Python 中按 `max_total` 截断。

方案：

- 保留无 `max_total` 时的简单 capture 路径。
- 有 `max_total` 时使用 `Popen` 流式读取 stdout。
- 达到上限后主动 terminate ripgrep。
- 主动终止导致的非零返回码不视为错误；真实 ripgrep 错误仍然抛出。
- `rg --max-count` 仅作为 per-file 限制继续保留，不误用作全局限制。

验收：

- `max_total` 返回数量不变。
- 大量匹配时不等待 ripgrep 全量输出。

### 3. patch dry-run 改为 in-memory 验证

现状：`apply_patch_any(dry_run=True)` snapshot 整个 root，真实写盘后再 restore，仓库文件数越大越慢。

方案：

- 在 patch 层增加 dry-run 入口，复用解析和 hunk 验证逻辑。
- dry-run 只读取 patch 涉及的文件，不写入文件系统。
- `apply_patch_any` 的 unified diff 路径继续先转换为 patch envelope。
- 保留 `PatchResult` 和错误行为。

验收：

- dry-run 后新增/修改/删除文件均不落盘。
- patch 失败时仍按 `check` 参数返回或抛错。

### 4. codequery 缓存刷新去除错误 prune

现状：`_refresh_cache` 会删除当前 candidate set 之外的 root 缓存。不同语言、glob、单文件 root 查询可能互相删除缓存。

方案：

- `_refresh_cache` 只更新当前候选文件的缓存，不删除候选外文件。
- 对当前候选中已消失的文件继续删除对应缓存。
- 保留 `clear_cache` 作为显式清理入口。
- 暂不引入复杂 SQL symbol 索引，避免扩大技术债。

验收：

- 单文件查询不会删除同 root 下其他文件缓存。
- 删除候选文件后的当前查询结果正确。

## 非目标

- 不重写 runtime helper API。
- 不更改 prompt helper 文档。
- 不引入新的第三方依赖。
- 不处理主应用 engine/session 的性能问题。
- 不做大规模格式化或无关清理。

## 进度

- [x] 1. runtime 顶层导入改为 lazy export
- [x] 2. codesearch 上限改为流式提前终止
- [ ] 3. patch dry-run 改为 in-memory 验证
- [ ] 4. codequery 缓存刷新去除错误 prune
- [ ] 5. 运行相关测试并提交最终状态
