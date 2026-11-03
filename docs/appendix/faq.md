# 常见问题

## 环境与构建

### Q1　必须用 Python 3.12 吗？

是。PyTorch 对 3.14 的轮子尚不完整，`pip install torch` 可能退化为源码编译。
uv 可以独立下载 3.12，不影响系统 Python：

```bash
uv venv --python 3.12 .venv
```

### Q2　`make serve` 报 `mkdocs: command not found`

没有激活 venv。Makefile 内部调用 `.venv/bin/mkdocs`，所以先确认 `.venv` 存在：

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

### Q3　中文在生成的图里显示成方块

matplotlib 找不到中文字体。检查：

```bash
.venv/bin/python -c "from matplotlib import font_manager as fm; print([f.name for f in fm.fontManager.ttflist if 'Hei' in f.name or 'PingFang' in f.name])"
```

为空就装思源黑体，或把本机字体名加进 `scripts/figs/style.py` 的 `CJK_CANDIDATES`。

### Q4　PDF 目录页是空的

已修复（`docs/assets/js/print-toc.js`）。若自己改过主题，注意：print-site 插件只从**右栏**复制目录，而本站启用了 `toc.integrate`，右栏不存在，插件脚本会报错。

### Q5　某一讲在 PDF 目录里消失了

该讲标题的锚点是纯数字（如 `id="1"`），触发了插件替换 bug。给标题加显式锚点即可：

```markdown
# 第 1 讲　大模型是什么 {: #l01 }
```

详见[环境手册 §8](env.md)。

## 阅读

### Q6　需要什么数学基础？

微积分、线性代数、概率论的基本概念即可。全书出现的推导都给了中间步骤，
最"硬"的是第 4 讲的方差论证与第 9 讲的拉格朗日乘子，都在正文里逐步展开。

### Q7　三种阅读路径怎么选？

- **🟢 入门**：只读「直觉 + 图解 + 工程要点」，跳过 🔴 折叠的推导，约 8 小时；
- **🟡 标准**：全部正文 + 前两档习题，适合一学期课程，约 40 学时；
- **🔴 深入**：推导 + 论文 + 第三档习题 + 复现实验。

### Q8　公式太多看不下去怎么办？

先看图。每张图的图注都写了"读图要点"，用一句话说明这张图要你看什么。
看懂图再回头看公式，通常会发现公式只是在描述图里已经说清的事。

### Q9　讲次之间有依赖吗？

有。主线是 L1 → L2 → L3 → L4 → L5，之后可以按兴趣跳：
对齐（L8–L10）、系统（L11–L14）、应用（L15–L19）、前沿（L20–L22）。
每讲开头都有"本讲位置"，写明前置知识。

## 实验

### Q10　没有 GPU 能跑吗？

能。8 个实验全部设计为 CPU 可跑，最慢的 Lab2（迷你 GPT）约 20 秒。

### Q11　实验需要下载模型吗？

不需要。Lab6 用 TF-IDF + SVD 代替预训练嵌入，Lab7 用规则策略代替 LLM，
Lab2/Lab4 是从零搭的小模型。全部离线可跑。

### Q12　实验结果和 `expected.md` 对不上

先看 `expected.md` 里的"常见失败与排查"表。若趋势对但数值有偏差，属于正常
（随机种子、CPU 型号、库版本都会影响）；若趋势不对，多半是实现问题。

### Q13　能用自己的数据替换实验里的语料吗？

可以。每个实验的语料都写在 `main.py` 顶部的常量里，替换后重新跑即可。
注意 Lab2 的语料太短会导致过拟合，建议至少几千字符。

## 使用与贡献

### Q14　能用于课程教学吗？

可以。文字部分采用 CC BY-NC-SA 4.0（非商业使用、署名、相同方式共享），
代码 MIT。课程大纲可参考 `PLAN.md` 的 24 讲安排，实验有评分表。

### Q15　发现错误怎么反馈？

优先改源文件而不是 PDF：所有内容都在 `docs/` 下的 Markdown 里，
图在 `scripts/figs/` 的脚本里，改完跑 `make pipeline L=<讲次>` 会自动校验。

### Q16　怎么新增一讲？

1. 复制 `docs/_template.md`，改名 `l{NN}-{slug}.md`；
2. 写内容（十段式，≥4 图、≥2 算例、≥1 段代码）；
3. 写 `scripts/figs/l{NN}_*.py` 生成插图；
4. 加进 `mkdocs.yml` 的 nav；
5. 跑 `make pipeline L=l{NN}` 直到全绿。

### Q17　为什么每讲都要有算例和图？

因为"看懂"和"会用"之间隔着一次手算。算例让抽象公式落到具体数字上，
图让趋势变成可看的东西——这两样是讲义区别于综述的地方。

### Q18　时间线和论文清单怎么更新？

时间线改 `refs/timeline.yaml`，然后跑 `python scripts/gen_timeline.py`；
论文清单直接改 `docs/appendix/papers.md`。两者都是单一数据源，
不要在生成的页面里手工改。
