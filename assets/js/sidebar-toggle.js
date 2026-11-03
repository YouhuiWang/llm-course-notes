/* 左侧栏收起/展开：状态存在 localStorage，刷新后保持 */
(function () {
  var KEY = "llm-course-sidebar";
  var ICON_MENU =
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 5h18v2H3V5zm0 6h18v2H3v-2zm0 6h18v2H3v-2z"/></svg>';
  var ICON_CLOSE =
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 5h12v2H3V5zm0 6h12v2H3v-2zm0 6h12v2H3v-2zM18 7l1.4 1.4L16.8 12l2.6 3.6L18 17l-4-5 4-5z"/></svg>';

  function collapsed() {
    return document.body.classList.contains("sidebar-collapsed");
  }

  function sync(btn) {
    btn.innerHTML = collapsed() ? ICON_CLOSE : ICON_MENU;
    btn.title = collapsed() ? "展开侧栏" : "收起侧栏";
    btn.setAttribute("aria-label", btn.title);
  }

  function init() {
    if (document.querySelector(".sidebar-toggle")) return;
    var btn = document.createElement("button");
    btn.className = "sidebar-toggle";
    btn.type = "button";

    var anchor = document.querySelector(".md-header__option") || document.querySelector(".md-header__inner");
    if (!anchor) return;
    anchor.parentNode.insertBefore(btn, anchor);

    sync(btn);
    btn.addEventListener("click", function () {
      document.body.classList.toggle("sidebar-collapsed");
      localStorage.setItem(KEY, collapsed() ? "1" : "0");
      sync(btn);
    });
  }

  // 允许用 ?sidebar=off 直接以收起状态打开（便于分享与截图）
  if (/[?&]sidebar=off/.test(location.search)) {
    document.body.classList.add("sidebar-collapsed");
  } else if (localStorage.getItem(KEY) === "1") {
    document.body.classList.add("sidebar-collapsed");
  }

  if (typeof document$ !== "undefined") {
    document$.subscribe(init);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
