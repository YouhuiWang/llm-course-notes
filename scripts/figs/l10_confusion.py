"""L10-2：拒答的混淆矩阵。

把"该拒"与"该答"和"实际拒了/答了"交叉，四种结果的代价完全不同：
误拒（FP）伤体验，漏拒（FN）出事故。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from style import INK, MUTED, PALETTE, save, setup

setup()

# 行：真实（该拒 / 该答）；列：系统（拒了 / 答了）
mat = np.array([[190, 10], [30, 770]])

fig, ax = plt.subplots(figsize=(5.6, 3.4))
im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=900)

labels = [["TP\n正确拦截", "FN\n漏放（事故）"], ["FP\n误拒（体验）", "TN\n正常通过"]]
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{labels[i][j]}\n{mat[i, j]}", ha="center", va="center",
                fontsize=8.2, color="white" if mat[i, j] > 500 else INK)

ax.set_xticks([0, 1], ["系统：拒答", "系统：回答"])
ax.set_yticks([0, 1], ["真实：有害", "真实：无害"])
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)

ax.set_title("同一套策略，两种错误代价不同", loc="left", fontsize=9.5)

fig.text(
    0.5, -0.06,
    "召回 95%（190/200）看似不错，但误拒 30/800 = 3.75% 已经足以让用户投诉",
    ha="center", fontsize=7.2, color=INK,
)

save(fig, "l10-confusion")
