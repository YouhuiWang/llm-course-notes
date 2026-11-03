/* 为「全书打印版」生成目录。
 *
 * 为什么需要自己写：print-site 插件自带的 print-site.js 只从
 * .md-sidebar--secondary（右栏）复制目录，而本站启用了 toc.integrate，
 * 右栏不存在，于是插件生成的目录页是空的。
 *
 * 这里直接扫描打印页里的 h1/h2 重建目录，不依赖主题结构。
 */
(function () {
  function buildToc() {
    var box = document.querySelector("#print-page-toc nav");
    if (!box || box.querySelector("ul")) return;      // 没有目录页或已生成

    // 只收章级标题：h2 有四百多个，全列出来会占五六页
    var heads = document.querySelectorAll("section.print-page h1");
    if (!heads.length) return;

    var ul = document.createElement("ul");
    ul.className = "print-toc";

    heads.forEach(function (h) {
      var text = (h.textContent || "").replace("¶", "").trim();
      if (!text || !h.id) return;
      var li = document.createElement("li");
      li.className = h.tagName === "H1" ? "print-toc-h1" : "print-toc-h2";
      var a = document.createElement("a");
      a.href = "#" + h.id;
      a.textContent = text;
      li.appendChild(a);
      ul.appendChild(li);
    });

    if (ul.children.length) box.appendChild(ul);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", buildToc);
  } else {
    buildToc();
  }
})();
