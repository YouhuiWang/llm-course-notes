# 数学基础速查

> 本页只收录正文反复用到的工具，不做教科书式展开。每条给出"什么时候用 + 关键结论"。

## 1 Softmax

### 定义

$$\softmax(\boldsymbol{z})_i=\frac{e^{z_i}}{\sum_{j=1}^{K}e^{z_j}}$$

### 数值稳定形式

$$\softmax(\boldsymbol{z})_i=\frac{e^{z_i-m}}{\sum_j e^{z_j-m}},\qquad m=\max_j z_j$$

减常数不改变结果（分子分母同乘 $e^{-m}$），但避免了上溢。

### 雅可比矩阵

$$\frac{\partial p_i}{\partial z_j}=p_i(\delta_{ij}-p_j)
\quad\Longleftrightarrow\quad
\boldsymbol{J}=\operatorname{diag}(\boldsymbol{p})-\boldsymbol{p}\boldsymbol{p}^\top$$

**用途**：解释 softmax 饱和导致梯度消失（第 4 讲）。当 $\boldsymbol{p}\to$ one-hot 时，$\boldsymbol{J}\to\boldsymbol{0}$。

### 温度

$$\softmax(\boldsymbol{z}/\tau)$$

$\tau<1$ 更尖锐，$\tau>1$ 更平坦。注意它与 $\sqrt{d_k}$ 缩放的数学形式相同、目的不同。

---

## 2 交叉熵与 KL 散度

### 交叉熵

$$H(p,q)=-\sum_i p_i\log q_i$$

语言模型的损失就是它：$p$ 是 one-hot 的真实 token，$q$ 是模型分布，于是

$$\mathcal{L}=-\log q_{y}$$

### KL 散度

$$D_{\mathrm{KL}}(p\,\|\,q)=\sum_i p_i\log\frac{p_i}{q_i}=H(p,q)-H(p)$$

**性质**：非负（Jensen 不等式），不对称。在 RLHF 中作为策略约束项：

$$\mathcal{L}=\E_{x\sim\pi_\theta}[r(x)]-\beta\,D_{\mathrm{KL}}(\pi_\theta\,\|\,\pi_{\text{ref}})$$

### 为什么 RLHF 用 KL 而不是 L2

KL 惩罚的是**分布形状的偏离**，而不是参数距离。它允许策略在等价输出上自由移动，只惩罚"跑偏到另一个分布"。

---

## 3 信息论三件套

| 量 | 定义 | 在 LLM 里的含义 |
|---|---|---|
| 熵 $H(p)$ | $-\sum p\log p$ | 输出的不确定性；采样多样性的理论上限 |
| 交叉熵 $H(p,q)$ | $-\sum p\log q$ | 训练损失 |
| 困惑度 | $\exp(H)$ | 平均每 token 的"候选数" |

**关系**：$H(p,q)=H(p)+D_{\mathrm{KL}}(p\|q)$。当 $p$ 是 one-hot 时 $H(p)=0$，交叉熵 = KL。

---

## 4 矩阵求导

设 $\boldsymbol{x}\in\R^n$，$\boldsymbol{A}\in\R^{n\times n}$：

$$\frac{\partial(\boldsymbol{a}^\top\boldsymbol{x})}{\partial\boldsymbol{x}}=\boldsymbol{a},
\qquad
\frac{\partial(\boldsymbol{x}^\top\boldsymbol{A}\boldsymbol{x})}{\partial\boldsymbol{x}}=(\boldsymbol{A}+\boldsymbol{A}^\top)\boldsymbol{x}
\overset{\boldsymbol{A}\text{ 对称}}{=}2\boldsymbol{A}\boldsymbol{x}$$

链式法则在反向传播中的形式：

$$\frac{\partial\mathcal{L}}{\partial\boldsymbol{x}}=\left(\frac{\partial\boldsymbol{y}}{\partial\boldsymbol{x}}\right)^\top\frac{\partial\mathcal{L}}{\partial\boldsymbol{y}}$$

**用途**：手推 attention 的反向（第 4 讲习题）、理解 residual 的梯度直通（第 5 讲）。

---

## 5 概率工具

### 贝叶斯

$$p(\theta\mid\mathcal{D})=\frac{p(\mathcal{D}\mid\theta)p(\theta)}{p(\mathcal{D})}$$

**用途**：解释 in-context learning 的"隐式贝叶斯"假说（第 7 讲）。

### 期望与方差

$$\Var(X)=\E[X^2]-\E[X]^2$$

独立变量：$\Var(\sum X_i)=\sum\Var(X_i)$。

**用途**：$\sqrt{d_k}$ 的方差推导（第 4 讲）——这是全书出现频率最高的一次计算。

### 重参数化技巧

$$\epsilon\sim\mathcal{N}(0,1),\quad z=\mu+\sigma\epsilon$$

**用途**：VAE、扩散模型的采样（第 19 讲）。

---

## 6 常用不等式

| 不等式 | 形式 | 用途 |
|---|---|---|
| Jensen | $f(\E[X])\le\E[f(X)]$（$f$ 凸） | 证明 KL ≥ 0 |
| Cauchy–Schwarz | $\lvert\langle x,y\rangle\rvert\le\|x\|\|y\|$ | 余弦相似度的界 |
| 三角不等式 | $\|x+y\|\le\|x\|+\|y\|$ | 分析误差累积 |

---

## 7 复杂度记号

| 记号 | 含义 | 例子 |
|---|---|---|
| $O(n^2d)$ | 注意力打分的时间复杂度 | 第 4 讲 |
| $O(nd^2)$ | 线性投影 | 第 4 讲 |
| $O(n)$ | KV cache 显存随长度增长 | 第 13 讲 |

**注意**：这里的 $O$ 隐藏了常数。实际选型时，$n$ 与 $d$ 的相对大小决定哪一项主导——$n\approx d$ 是分界点。
