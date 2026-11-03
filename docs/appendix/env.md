# 环境手册

> 目标：从零到能跑通全书代码，不超过 10 分钟。

## 1 前置

| 项 | 要求 | 检查命令 |
|---|---|---|
| 操作系统 | macOS 12+ / Linux / WSL2 | `uname -a` |
| Python | 3.12（**不要用 3.14**） | `python3 --version` |
| uv | ≥ 0.5 | `uv --version` |
| 磁盘 | ≥ 5 GB（含 torch 与模型权重） | `df -h .` |

!!! warning "为什么固定 3.12"
    PyTorch 对新版本 Python 的轮子发布有滞后。3.14 上 `pip install torch` 可能退化为源码编译，耗时且易失败。uv 可以独立下载 3.12，不影响系统 Python。

```bash
# 安装 uv（若未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2 三条独立的构建链

本项目有三类任务，各自依赖不同，建议分环境：

| 链 | 用途 | 依赖文件 | 命令 |
|---|---|---|---|
| 文档站 | 写讲义、预览、出 PDF | `requirements.txt` | `make serve` / `make pdf` |
| 图表 | 生成 SVG/PDF 插图 | `requirements-fig.txt` | `make figures` |
| 实验 | 跑 8 个 Lab | `requirements-lab.txt` | `python labs/lab1_attention/main.py` |

**最省事的做法**：一个 venv 装全部（约 2 GB）。

```bash
cd 大模型发展与相关技术详解
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt -r requirements-fig.txt -r requirements-lab.txt
```

## 3 验证

```bash
# 文档站
make serve          # 打开 http://127.0.0.1:8000

# 图表（会打印每张图的输出路径）
make figures

# PDF
make pdf            # 输出 dist/大模型发展与相关技术详解.pdf
```

三件事都成功，环境就绪。

## 4 常见问题

### 4.1 `mkdocs: command not found`

没有激活 venv，或用了系统 Python。用 `make serve`（内部调用 `.venv/bin/mkdocs`）可绕过。

### 4.2 中文在图中显示为方块

matplotlib 找不到中文字体。检查：

```bash
.venv/bin/python -c "from matplotlib import font_manager as fm; print([f.name for f in fm.fontManager.ttflist if 'Hei' in f.name or 'PingFang' in f.name])"
```

若为空，安装思源黑体（Noto Sans CJK SC），或在 `scripts/figs/style.py` 的 `CJK_CANDIDATES` 里加入本机字体名。

### 4.3 PDF 生成失败

- 检查 Chrome 路径：`ls "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"`
- 自定义路径：`CHROME=/path/to/chrome make pdf`
- 页面没生成：先 `make build`，确认 `site/print_page/index.html` 存在
- PDF 里公式是源码：KaTeX 未加载，检查 `docs/assets/vendor/katex/` 是否完整

### 4.4 `--strict` 构建报死链

MkDocs 严格模式会把断链当错误。先运行 `make check` 定位，再修 Markdown 链接（常见原因：相对路径层级写错）。

### 4.5 torch 安装慢

```bash
# 使用国内镜像（可选）
uv pip install --python .venv/bin/python torch --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

CPU 版足够跑全部实验；GPU 只在 Lab 5 的大模型量化对比中才有明显收益。

### 4.6 Apple Silicon 上的多进程报错

Lab 7 的沙箱执行用到 `multiprocessing`，macOS 默认 spawn 模式。若报 `RuntimeError`，在入口加：

```python
if __name__ == "__main__":
    ...
```

## 5 目录速查

```
.venv/                 虚拟环境（不入库）
docs/                  讲义源文件
docs/assets/vendor/    KaTeX 与 Mermaid（本地化，离线可用）
docs/assets/img/       插图（脚本生成，勿手改）
scripts/figs/          每张图一个脚本
labs/                  实验代码
refs/                  papers.bib + sources.yaml
site/ dist/            构建产物（不入库）
```

## 6 版本锁定策略

- 文档链：`requirements.txt` 只锁大版本（`>=`），保证插件兼容；
- 实验链：`requirements-lab.txt` 锁小版本（`==`），保证数值可复现；
- 每次升级后跑 `make figures && make build && make pdf`，三件事都过才算升级成功。

---

## 7 部署

### 7.1 本地预览

```bash
make serve          # http://127.0.0.1:8000，改文件自动刷新
```

### 7.2 构建静态站

```bash
make build          # 输出到 site/，可直接用任意静态服务器托管
python -m http.server -d site 8000
```

`make build` 带 `--strict`：任何死链或未知文件都会让构建失败。

### 7.3 部署到 GitHub Pages

仓库里加一个 workflow（`.github/workflows/pages.yml`）：

```yaml
name: pages
on:
  push:
    branches: [main]
permissions:
  contents: read
  pages: write
  id-token: write
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt
      - run: mkdocs build --strict
      - uses: actions/upload-pages-artifact@v3
        with:
          path: site
  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
    steps:
      - uses: actions/deploy-pages@v4
```

然后在仓库 Settings → Pages 里把 Source 设为 "GitHub Actions"。

### 7.4 分发 PDF

```bash
make pdf            # dist/大模型发展与相关技术详解.pdf
```

PDF 由 `scripts/build_pdf.sh` 用 Chrome headless 打印"全书打印版"页面生成，
与网页同源——改 Markdown 就同时改了两者。

### 7.5 版本管理建议

- 每讲完成打一个 tag（如 `v0.2-l09`）；
- PDF 产物不入库（`dist/` 已在 `.gitignore`）；
- 术语表与时间线是"单一数据源"，改数据后跑 `make figures` 或 `gen_timeline.py` 重新生成。

---

## 8 已知问题与排查

### 8.1 PDF 目录页为空

**现象**：`make pdf` 生成的 PDF 第 2 页只有"目录"两个字，下面空白。

**原因**：`mkdocs-print-site-plugin` 的 `print-site.js` 只从右栏
（`.md-sidebar--secondary`）复制目录，而本站启用了 `toc.integrate`，
右栏不存在，脚本报错。

**修复**：`docs/assets/js/print-toc.js` 直接扫描打印页里的 `h1` 重建目录，
不依赖主题结构。已在 `mkdocs.yml` 的 `extra_javascript` 中注册。

### 8.2 某一讲的标题在打印页里变成 `<h00-导论-...>`

**现象**：PDF 目录缺某一讲，HTML 里该页的标题标签名被污染
（`<h00-导论-l01-what-is-llm-1>`）。

**原因**：当 Markdown 的 H1 生成的锚点是一个**纯数字**（如 `# 第 1 讲` 生成 `id="1"`），
插件在做字符串替换时会误伤标签名。

**修复**：给该标题加显式锚点，避开纯数字 id：

```markdown
# 第 1 讲　大模型是什么 {: #l01 }
```

新建讲次时，若标题以数字开头，建议一律加显式锚点。
