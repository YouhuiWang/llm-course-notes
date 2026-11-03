/* KaTeX 渲染：支持 arithmatex 的 \( \) 与 \[ \]，以及 $$ 行间公式 */
(function () {
  function renderMath() {
    if (typeof renderMathInElement === "undefined") return;
    renderMathInElement(document.body, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false },
        { left: "$", right: "$", display: false },
      ],
      ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code", "option"],
      throwOnError: false,
      errorColor: "#cc0000",
      macros: {
        "\\R": "\\mathbb{R}",
        "\\E": "\\mathbb{E}",
        "\\Var": "\\operatorname{Var}",
        "\\softmax": "\\operatorname{softmax}",
        "\\attn": "\\operatorname{Attn}",
        "\\dmodel": "d_{\\text{model}}",
        "\\dff": "d_{\\text{ff}}",
      },
    });
  }

  if (typeof document$ !== "undefined") {
    document$.subscribe(renderMath);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", renderMath);
  } else {
    renderMath();
  }
})();
