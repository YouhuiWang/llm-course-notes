# 大模型发展与相关技术详解

> 从统计语言模型到智能体：原理、推导与工程实践的系统讲义。
> **25 讲 · 8 个可复现实验 · 网页与 PDF 双交付**
>
> 🌐 在线阅读：<https://youhuiwang.github.io/llm-course-notes/>
> 📦 仓库：<https://github.com/YouhuiWang/llm-course-notes>

本项目是一份可直接用于研究生课程的教学讲义。与常见综述不同，它坚持三件事：

1. **每个技术选择都还原成它要解决的问题** —— 有推导，不只有结论；
2. **每个原理都有可运行的最小实现** —— 8 个实验全部 CPU 可跑；
3. **每个 2024 年之后的事实都有来源** —— 附抓取日期与原文摘句。

## 快速开始

**最简单：双击 `启动讲义.command`**（macOS），会自动构建并在浏览器打开；
看完双击 `关闭讲义.command` 停掉后台服务。

手动方式：


```bash
# 1. 文档站（Python 3.12）
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
make serve                 # http://127.0.0.1:8000

# 2. 图表（全部由脚本生成，不手工维护）
uv pip install --python .venv/bin/python -r requirements-fig.txt
make figures

# 3. PDF（Chrome headless 打印"全书打印版"）
make pdf                   # 输出 dist/大模型发展与相关技术详解.pdf

# 4. 一致性检查（死链、术语、结构审计）
make check

# 5. 单讲交付流水线：内容 → 出图 → 审计 → 构建 → 死链 → PDF → 提交
make pipeline L=l16          # 任一步失败即中止
make pipeline ARGS=--all     # 对全部已完成讲次跑一遍
```

## 目录结构

```
docs/            讲义正文（24 讲 + 8 实验 + 附录）
figures/         插图目录（符号链接到 docs/assets/img，SVG 供网页，PDF 供印刷）
scripts/figs/    每张图一个脚本，可单独重跑
labs/            实验代码，每个实验一个目录
refs/            papers.bib + sources.yaml（事实来源台账）
site/            构建产物（不入库）
dist/            PDF 产物
```

## 阅读路径

| 路径 | 读什么 | 用时 |
|---|---|---|
| 🟢 入门 | 直觉 + 图解 + 工程要点，跳过 🔴 折叠推导 | ~8 h |
| 🟡 标准 | 全部正文 + 前两档习题（一学期课程） | ~40 学时 |
| 🔴 深入 | 推导 + 论文 + 实验复现 | 视方向而定 |

从 [第 1 讲](docs/00-导论/l01-what-is-llm.md) 开始，或直接看 [课程地图](docs/index.md)。

九部分 25 讲：导论 → 前史 → Transformer → 对齐 → 架构与系统 → 应用层 → 推理与前沿 → 治理与实践 → **模型解剖**。

## 进度

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M0 | 骨架与构建链 | ✅ 已完成 |
| M1 | 样章 L4（注意力）、L17（Agent） | ✅ 已完成 |
| M2 | L1–L9 成文 | ✅ 已完成 |
| M3 | L10–L16 + Lab1–3 | ✅ 已完成 |
| M4 | L17–L19 + Lab7 | ✅ 已完成 |
| M5 | L20–L22 + Lab8 | ✅ 已完成 |
| M6 | L23–L25 + 附录 | ✅ 已完成 |
| M7 | PDF 精排、校对、发布上线 | ✅ 已完成 |

**全部完成**：25 讲正文、8 个实验、106 张插图、7.7 万字，PDF 482 页。
在线版：<https://youhuiwang.github.io/llm-course-notes/>

详见 [PLAN.md](PLAN.md)。

## 写作规范

每讲按 `docs/_template.md` 的十段式展开，硬性要求见 [PLAN.md](PLAN.md) §5：

- **≥2 个算例**，其中至少 1 个能手算（数字真实、步骤可复核）
- **≥4 张图**，全部由 `scripts/figs/` 脚本生成，图注写"读图要点"
- 每 800–1200 字配一张图

新建一讲：复制 `docs/_template.md` → 改名 `l{NN}-{slug}.md` → 填写。

## 打开方式

**双击 `启动讲义.command`** —— 自动构建、后台起服务、打开浏览器；
**双击 `关闭讲义.command`** —— 停止服务。

也可以手动：

```bash
./scripts/serve.sh start    # 启动（默认端口 8123）
./scripts/serve.sh stop     # 停止
./scripts/serve.sh status   # 查看状态
PORT=9000 ./scripts/serve.sh start   # 换端口
```

在线版本：<https://youhuiwang.github.io/llm-course-notes/>

## 部署

```bash
make build     # 静态站 → site/（--strict，死链会构建失败）
make pdf       # PDF → dist/
```

GitHub Pages 的 workflow 模板见 [环境手册 §7](docs/appendix/env.md#7)。

## 约定

- 公式编号 `(L4.3)` = 第 4 讲第 3 式，可跨章引用。
- 术语首次出现中英对照，之后用中文；完整表见 `docs/appendix/glossary.md`。
- 2024 年后的事实标注 `📅 事实核查至 YYYY-MM`。

## 许可

文字 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh)，代码 MIT。
