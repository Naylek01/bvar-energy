(function () {
  "use strict";

  // GRAPH_PNG_CORNER_CONTRACT_V1
  // One explicit PNG control per Plotly graph. It lives in a dedicated strip
  // above the graph, so it never covers the Plotly title or plotted data.
  const BUTTON_CLASS = "dashboard-png-corner-button";

  function clean(value) {
    return String(value || "")
      .replace(/<[^>]*>/g, "")
      .replace(/[^\w.-]+/g, "_")
      .replace(/^_+|_+$/g, "")
      .slice(0, 90);
  }

  function vintage() {
    const node = document.getElementById("xlsx-export-vintage");
    return node ? String(node.textContent || "").trim() : "";
  }

  function filename(plot, root) {
    const title = (
      plot
      && plot.layout
      && plot.layout.title
      && (plot.layout.title.text || plot.layout.title)
    );
    const base = clean(title) || clean(root.id) || "dashboard_chart";
    const v = clean(vintage());
    return v ? base + "_vintage_" + v : base;
  }

  function removeLegacyPngButtons(root) {
    root.querySelectorAll("button").forEach((button) => {
      if (button.classList.contains(BUTTON_CLASS)) return;
      const text = String(button.textContent || "").trim().toUpperCase();
      if (text === "PNG" || text === "↓ PNG" || text === "DOWNLOAD PNG") {
        button.remove();
      }
    });
  }

  function hideNativePlotlyDownload(root) {
    root.querySelectorAll(
      '.modebar-btn[data-title*="Download plot"], '
      + '.modebar-btn[data-title*="png" i]'
    ).forEach((node) => {
      node.style.display = "none";
    });
  }

  function attach(plot) {
    if (!plot) return;
    const root = plot.closest(".dash-graph") || plot.parentElement;
    if (!root) return;

    removeLegacyPngButtons(root);
    hideNativePlotlyDownload(root);

    if (root.querySelector(":scope > ." + BUTTON_CLASS)) return;

    if (!root.style.position || root.style.position === "static") {
      root.style.position = "relative";
    }
    const currentPadding = parseFloat(
      window.getComputedStyle(root).paddingTop || "0"
    );
    if (!Number.isFinite(currentPadding) || currentPadding < 30) {
      root.style.paddingTop = "30px";
    }
    root.style.boxSizing = "border-box";

    const button = document.createElement("button");
    button.type = "button";
    button.className = BUTTON_CLASS;
    button.textContent = "↓ PNG";
    button.title = "Download chart as PNG";
    Object.assign(button.style, {
      position: "absolute",
      top: "4px",
      right: "8px",
      zIndex: "40",
      border: "1px solid #D2D0D1",
      borderRadius: "7px",
      background: "rgba(255,255,255,.98)",
      color: "#3A3D40",
      fontFamily: "Inter, Segoe UI, sans-serif",
      fontSize: "10px",
      fontWeight: "700",
      padding: "4px 9px",
      cursor: "pointer",
      boxShadow: "0 1px 3px rgba(0,0,0,.06)",
      whiteSpace: "nowrap",
    });

    button.addEventListener("click", async function (event) {
      event.preventDefault();
      event.stopPropagation();
      try {
        if (!window.Plotly || !window.Plotly.downloadImage) {
          throw new Error("Plotly downloadImage is unavailable.");
        }
        await window.Plotly.downloadImage(plot, {
          format: "png",
          scale: 2,
          filename: filename(plot, root),
        });
      } catch (error) {
        console.error("PNG export failed", error);
        window.alert("PNG export failed: " + error.message);
      }
    });

    root.insertBefore(button, root.firstChild);
  }

  function scan() {
    document.querySelectorAll(".js-plotly-plot").forEach(attach);
  }

  const observer = new MutationObserver(scan);
  observer.observe(
    document.documentElement,
    {childList: true, subtree: true}
  );
  window.addEventListener("load", scan);
  document.addEventListener("DOMContentLoaded", scan);
  window.setInterval(scan, 1500);
})();
