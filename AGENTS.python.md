# Python 扩展规范

## 适用条件

当任务涉及本项目的 Python 源码、`uv` 项目配置、agent 核心库、Python runner、JSONL 会话记录或 Textual TUI 时，必须遵守本规范。

本规范描述开发本项目时的工程约定。产品目标和运行时设计写在 `docs/design.md`；如果两者都涉及同一主题，本规范约束代码组织与协作方式，设计文档描述产品语义。

## 目录约定

- 源码目录：`src/uv_agent/`
- TUI 目录：`src/uv_agent/tui/`
- Agent 核心目录：`src/uv_agent/core/`
- Python runner 目录：`src/uv_agent/runner/`
- 会话与记录目录：`src/uv_agent/session/` 或按实现阶段拆分到更明确的 feature 目录
- 测试目录：`tests/`
- 配置文件：`pyproject.toml`、`uv.lock`
- 运行时数据：默认放在 `.uv-agent/`，包括受管理脚本、run JSONL 和会话 JSONL；该目录不作为源码契约

## 命名规范

- 文件/目录：小写 snake_case
- 类：PascalCase
- 函数/变量：snake_case
- 常量：UPPER_SNAKE_CASE
- Textual 组件：类名使用 PascalCase，文件名使用 snake_case

## 代码风格

- 开发本项目时，临时辅助 Python 脚本使用 `python3` 运行；项目脚本和项目命令优先使用 `uv`
- 上一条只约束开发协作方式，不描述产品内 agent runner 的执行语义；runner 设计以 `docs/design.md` 为准
- 格式化工具：未定
- Lint 工具及配置：未定
- 导入排序规则：未定
- 新增公共 API 时优先使用类型标注和 dataclass / pydantic 等结构化模型；具体依赖在实现时再确认
- 只在不直观的设计选择上写 `NOTE:` 注释，不用注释复述普通控制流

## Unit 约定

- 单文件建议行数上限：约 300 行；明显承担多个职责时提前拆分
- 单函数建议行数上限：约 60 行；超过时优先抽出带类型约定的小函数
- 单文件导出成员上限：约 8 个；公共导出增加时同步考虑 feature `AGENTS.md`
- 拆分信号：TUI 逻辑依赖 runner 细节、runner 依赖 Textual、session 记录与模型调用互相纠缠、或同一文件同时处理配置/执行/渲染

## 依赖管理

- 包管理器：`uv`
- 锁文件：`uv.lock`
- 新增依赖前确认：新增长期依赖前先判断是否属于核心库、TUI、runner 或测试工具；避免为了单个临时脚本把依赖加入项目依赖
- 产品内 agent 临时脚本的依赖、runtime 注入和执行隔离语义写在 `docs/design.md`

## 构建与运行

- 开发：`uv run main.py`
- 构建：未定
- 测试：`uv run pytest`
- Lint：未定

## TUI 实现约定

- TUI 使用 Textual
- 布局和交互目标以 `docs/design.md` 为准
- TUI 只能消费 core/session/runner 提供的状态和事件，不把业务规则写死在 UI 层
- 除 TUI 外的核心能力必须可以在无界面环境中 import 并调用

## 测试约定

- 测试框架：`pytest` + `pytest-asyncio`
- 测试文件位置：`tests/`
- 命名规则：`test_*.py`
- 覆盖率要求：未定
- runner 测试必须覆盖 argv 构造、inline metadata 规范化、uv_args 记录、JSONL 事件、历史脚本重跑语义

## 部署与环境

- 环境变量管理方式：未定
- 部署方式：未定
- OpenAI 凭据读取方式在实现配置层时确定；不要把凭据写入仓库、JSONL 或脚本 artifact
