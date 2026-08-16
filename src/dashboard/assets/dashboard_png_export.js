/*
Inflation Dashboard — universal clean PNG export v2
===================================================

Presentation-only asset. No callback, model, registry, cache or economic object
is touched.

Graphs
------
For every rendered Dash/Plotly graph, add a small "↓ PNG" control. At click time
the exporter reads the *live* Plotly DOM object, so browser-only zoom/pan,
legend visibility and dragged shapes are preserved. It then applies the same
report-style principles as the Economic Conditions Heatmap export:
- white background;
- no chart title;
- 1050 px report width;
- scale 2;
- boxed axes with inward ticks;
- current visible ranges preserved.

The existing Economic Heatmap export is intentionally left alone because it
already implements this exact contract with its dedicated Kaleido callback.

Tables
------
For every native HTML table and Dash DataTable, add a "↓ PNG" control. The
currently rendered table is extracted from the DOM and rebuilt as a clean
Plotly table before PNG export. This avoids raw browser screenshots and removes
Dash controls/scrollbars/chrome from the downloaded image.

The asset is loaded automatically by Dash from src/dashboard/assets/.
*/

(function () {
    "use strict";

    if (window.__inflationUniversalPngExportV2) {
        return;
    }
    window.__inflationUniversalPngExportV2 = true;

    const EXPORT_WIDTH = 1050;
    const EXPORT_SCALE = 2;
    const GRAPH_MIN_HEIGHT = 340;
    const GRAPH_MAX_HEIGHT = 1600;
    const TABLE_MIN_HEIGHT = 220;
    const TABLE_MAX_HEIGHT = 12000;

    function nowStamp() {
        const d = new Date();
        const pad = (x) => String(x).padStart(2, "0");
        return (
            d.getFullYear()
            + pad(d.getMonth() + 1)
            + pad(d.getDate())
            + "_"
            + pad(d.getHours())
            + pad(d.getMinutes())
            + pad(d.getSeconds())
        );
    }

    function cleanText(value) {
        const div = document.createElement("div");
        div.innerHTML = String(value == null ? "" : value);
        return (div.textContent || div.innerText || "")
            .replace(/\s+/g, " ")
            .trim();
    }

    function slugify(value, fallback) {
        const raw = cleanText(value || fallback || "dashboard_export")
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, "_")
            .replace(/^_+|_+$/g, "");
        return raw || "dashboard_export";
    }

    function visible(el) {
        if (!el) return false;
        const style = window.getComputedStyle(el);
        if (style.display === "none" || style.visibility === "hidden") {
            return false;
        }
        const rect = el.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0;
    }

    function nearestHeading(node) {
        if (!node) return "";

        // Prefer a heading immediately preceding the target.
        let cursor = node.previousElementSibling;
        for (let i = 0; cursor && i < 5; i += 1, cursor = cursor.previousElementSibling) {
            const heading = cursor.matches && cursor.matches("h1,h2,h3,h4,h5,h6,.panel-title,.section-title,.eyebrow")
                ? cursor
                : cursor.querySelector
                    ? cursor.querySelector("h1,h2,h3,h4,h5,h6,.panel-title,.section-title,.eyebrow")
                    : null;
            const text = heading ? cleanText(heading.textContent) : "";
            if (text && text.length <= 180) return text;
        }

        const panel = node.closest
            ? node.closest(".panel,.chart-panel,.card,.section,[class*='panel']")
            : null;
        if (panel) {
            const heading = panel.querySelector(
                ".panel-title,.section-title,h1,h2,h3,h4,h5,h6,.eyebrow"
            );
            const text = heading ? cleanText(heading.textContent) : "";
            if (text && text.length <= 180) return text;
        }
        return "";
    }

    function triggerDownload(dataUrl, filename) {
        const a = document.createElement("a");
        a.href = dataUrl;
        a.download = filename;
        a.style.display = "none";
        document.body.appendChild(a);
        a.click();
        a.remove();
    }

    function copyObject(value) {
        return JSON.parse(JSON.stringify(value));
    }

    function clamp(value, low, high) {
        return Math.max(low, Math.min(high, value));
    }

    function reportAxis(axis, isX, fullAxis) {
        const out = Object.assign({}, axis || {});
        const full = fullAxis || {};

        // Preserve the browser's actual zoom/pan window even when the user
        // changed it client-side and the minimal Dash layout never received it.
        if (Array.isArray(full.range) && full.range.length === 2) {
            out.range = copyObject(full.range);
            out.autorange = false;
        }
        if (full.type && !out.type) out.type = full.type;
        if (full.categoryorder && !out.categoryorder) {
            out.categoryorder = full.categoryorder;
        }
        if (Array.isArray(full.categoryarray) && !out.categoryarray) {
            out.categoryarray = copyObject(full.categoryarray);
        }

        // Report PNGs always get a complete chart frame for visible axes.
        if (out.visible !== false) {
            out.showline = true;
            out.linecolor = "#111827";
            out.linewidth = 1;
            out.mirror = true;
            out.ticks = "inside";
            out.ticklen = 4;
            out.tickcolor = "#111827";
            out.tickwidth = 1;
        }
        if (isX && out.title && typeof out.title === "object") {
            out.title = Object.assign({}, out.title, {text: ""});
        }
        return out;
    }

    function styleGraphFigure(plot) {
        const data = copyObject(plot.data || []);
        const layout = copyObject(plot.layout || {});
        const fullLayout = plot._fullLayout || {};

        // Match the current Heatmap report contract.
        layout.title = {text: ""};
        layout.width = EXPORT_WIDTH;
        layout.paper_bgcolor = "white";
        layout.plot_bgcolor = "white";
        layout.font = Object.assign({}, layout.font || {}, {
            size: 13,
            family: "Arial, sans-serif",
            color: "#111827"
        });

        const axisKeys = new Set(["xaxis", "yaxis"]);
        Object.keys(layout).forEach(function (key) {
            if (/^[xy]axis\d*$/.test(key)) axisKeys.add(key);
        });
        Object.keys(fullLayout).forEach(function (key) {
            if (/^[xy]axis\d*$/.test(key)) axisKeys.add(key);
        });
        axisKeys.forEach(function (key) {
            const isX = key.indexOf("xaxis") === 0;
            const fullAxis = fullLayout[key];
            if (!fullAxis && !layout[key]) return;
            layout[key] = reportAxis(layout[key], isX, fullAxis);
        });

        // Keep a real legend in report exports. The previous generic exporter
        // reused dashboard legend coordinates above the plotting area but then
        // reduced the top margin, which clipped volatility legends entirely.
        const legendTraces = data.filter(function (trace) {
            return trace
                && trace.showlegend !== false
                && cleanText(trace.name || "") !== "";
        });
        layout.showlegend = legendTraces.length > 0;
        if (layout.showlegend) {
            layout.legend = Object.assign({}, layout.legend || {}, {
                orientation: "h",
                x: 0,
                xanchor: "left",
                y: 1.02,
                yanchor: "bottom",
                bgcolor: "rgba(255,255,255,0.92)",
                borderwidth: 0,
                font: Object.assign({}, (layout.legend || {}).font || {}, {
                    size: 11,
                    color: "#111827"
                })
            });
        }
        layout.margin = {
            l: 78,
            r: 42,
            t: layout.showlegend ? 100 : 42,
            b: 58
        };

        delete layout.dragmode;
        delete layout.uirevision;
        delete layout.hovermode;
        delete layout.hoverlabel;

        const liveHeight =
            Number(layout.height)
            || Math.round(plot.getBoundingClientRect().height)
            || 500;
        const height = clamp(liveHeight, GRAPH_MIN_HEIGHT, GRAPH_MAX_HEIGHT);
        layout.height = height;

        return {data: data, layout: layout, height: height};
    }

    async function exportGraph(graphRoot, button) {
        if (!window.Plotly) {
            throw new Error("Plotly is not available in the browser.");
        }
        const plot = graphRoot.querySelector(".js-plotly-plot");
        if (!plot || !plot.data || !plot.layout) {
            throw new Error("Rendered Plotly graph is unavailable.");
        }

        const styled = styleGraphFigure(plot);
        const layoutTitle =
            plot.layout && plot.layout.title
                ? cleanText(
                    typeof plot.layout.title === "string"
                        ? plot.layout.title
                        : plot.layout.title.text
                )
                : "";
        const title =
            layoutTitle
            || nearestHeading(graphRoot)
            || graphRoot.id
            || "dashboard_chart";

        button.classList.add("inflation-png-busy");
        const oldText = button.textContent;
        button.textContent = "…";
        try {
            const dataUrl = await window.Plotly.toImage(
                {data: styled.data, layout: styled.layout},
                {
                    format: "png",
                    width: EXPORT_WIDTH,
                    height: styled.height,
                    scale: EXPORT_SCALE
                }
            );
            triggerDownload(
                dataUrl,
                slugify(title, "dashboard_chart") + "_" + nowStamp() + ".png"
            );
        } finally {
            button.classList.remove("inflation-png-busy");
            button.textContent = oldText;
        }
    }

    function makeGraphButton(graphRoot) {
        // Economic Heatmap already has its dedicated "Download report PNG".
        if (
            graphRoot.id === "economic-heatmap"
            || graphRoot.closest("#economic-heatmap")
        ) {
            return;
        }

        let button = graphRoot.querySelector(
            ":scope > .inflation-png-graph-button"
        );
        if (button) return;

        const computed = window.getComputedStyle(graphRoot);
        if (computed.position === "static") {
            graphRoot.style.position = "relative";
        }

        button = document.createElement("button");
        button.type = "button";
        button.className =
            "inflation-png-button inflation-png-graph-button";
        button.textContent = "↓ PNG";
        button.title =
            "Download the currently visible chart as a clean report PNG";
        button.setAttribute(
            "aria-label",
            "Download current chart as PNG"
        );
        button.addEventListener("click", function (event) {
            event.preventDefault();
            event.stopPropagation();
            exportGraph(graphRoot, button).catch(function (err) {
                console.error("PNG chart export failed:", err);
                const oldText = button.textContent;
                button.textContent = "Error";
                window.setTimeout(function () {
                    button.textContent = oldText;
                }, 1800);
            });
        });
        graphRoot.appendChild(button);
    }

    function cellText(cell) {
        if (!cell) return "";
        return cleanText(cell.innerText || cell.textContent || "");
    }

    function rowCells(row) {
        return Array.from(row.children || []).filter(function (cell) {
            return cell.matches && cell.matches("th,td");
        });
    }

    function expandRow(row) {
        const cells = rowCells(row);
        const values = [];
        cells.forEach(function (cell) {
            const span = Math.max(
                1,
                parseInt(cell.getAttribute("colspan") || "1", 10)
            );
            values.push(cellText(cell));
            for (let j = 1; j < span; j += 1) {
                values.push("");
            }
        });
        return values;
    }

    function largestVisibleTable(root) {
        if (!root) return null;
        if (root.tagName && root.tagName.toLowerCase() === "table") {
            return root;
        }
        const candidates = Array.from(root.querySelectorAll("table"))
            .filter(visible);
        if (!candidates.length) return null;
        candidates.sort(function (a, b) {
            return b.querySelectorAll("th,td").length
                - a.querySelectorAll("th,td").length;
        });
        return candidates[0];
    }

    function extractTable(root) {
        const table = largestVisibleTable(root);
        if (!table) {
            throw new Error("Rendered table is unavailable.");
        }

        let rows = Array.from(table.querySelectorAll("tr")).filter(function (row) {
            return visible(row) && rowCells(row).length > 0;
        });
        if (!rows.length) {
            throw new Error("Rendered table has no visible rows.");
        }

        const matrix = rows.map(expandRow);
        const maxCols = Math.max.apply(
            null,
            matrix.map(function (row) { return row.length; })
        );
        matrix.forEach(function (row) {
            while (row.length < maxCols) row.push("");
        });

        let headerIndex = 0;
        const explicitHeader = rows.findIndex(function (row) {
            const cells = rowCells(row);
            return cells.length && cells.every(function (cell) {
                return cell.tagName.toLowerCase() === "th";
            });
        });
        if (explicitHeader >= 0) headerIndex = explicitHeader;

        const header = matrix[headerIndex].slice();
        const body = matrix.filter(function (_, idx) {
            return idx !== headerIndex;
        });

        // Remove rows that are entirely empty.
        const filteredBody = body.filter(function (row) {
            return row.some(function (value) {
                return cleanText(value) !== "";
            });
        });

        // Ensure unique nonempty report headers.
        for (let c = 0; c < header.length; c += 1) {
            if (!cleanText(header[c])) {
                header[c] = c === 0 ? "Item" : "";
            }
        }

        return {
            table: table,
            header: header,
            rows: filteredBody,
            ncols: maxCols
        };
    }

    function columnWeights(extracted) {
        const allRows = [extracted.header].concat(extracted.rows);
        const weights = [];
        for (let c = 0; c < extracted.ncols; c += 1) {
            let maxLen = 4;
            allRows.forEach(function (row) {
                maxLen = Math.max(
                    maxLen,
                    cleanText(row[c] || "").length
                );
            });
            weights.push(
                Math.max(
                    c === 0 ? 1.8 : 1.0,
                    Math.min(c === 0 ? 4.0 : 2.5, maxLen / 12)
                )
            );
        }
        return weights;
    }

    async function exportTable(targetRoot, button) {
        if (!window.Plotly) {
            throw new Error("Plotly is not available in the browser.");
        }

        const extracted = extractTable(targetRoot);
        const title =
            nearestHeading(targetRoot)
            || targetRoot.id
            || "dashboard_table";

        const columns = [];
        for (let c = 0; c < extracted.ncols; c += 1) {
            columns.push(
                extracted.rows.map(function (row) {
                    return row[c] == null ? "" : row[c];
                })
            );
        }

        const align = Array.from(
            {length: extracted.ncols},
            function (_, idx) { return idx === 0 ? "left" : "right"; }
        );

        const rowHeight = 29;
        const titleHeight = title ? 52 : 22;
        const height = clamp(
            titleHeight
            + 46
            + Math.max(1, extracted.rows.length) * rowHeight
            + 34,
            TABLE_MIN_HEIGHT,
            TABLE_MAX_HEIGHT
        );

        const trace = {
            type: "table",
            columnwidth: columnWeights(extracted),
            header: {
                values: extracted.header,
                align: align,
                height: 32,
                fill: {color: "#F3F4F6"},
                line: {color: "#C9CDD3", width: 1},
                font: {
                    family: "Arial, sans-serif",
                    size: 12,
                    color: "#111827"
                }
            },
            cells: {
                values: columns,
                align: align,
                height: rowHeight,
                fill: {color: "white"},
                line: {color: "#DADDE2", width: 1},
                font: {
                    family: "Arial, sans-serif",
                    size: 12,
                    color: "#111827"
                }
            }
        };

        const layout = {
            width: EXPORT_WIDTH,
            height: height,
            paper_bgcolor: "white",
            plot_bgcolor: "white",
            margin: {
                l: 34,
                r: 34,
                t: title ? 58 : 28,
                b: 28
            },
            font: {
                family: "Arial, sans-serif",
                color: "#111827"
            },
            title: title
                ? {
                    text: cleanText(title),
                    x: 0.01,
                    xanchor: "left",
                    y: 0.985,
                    yanchor: "top",
                    font: {size: 15, color: "#111827"}
                }
                : {text: ""}
        };

        const hidden = document.createElement("div");
        hidden.className = "inflation-export-hidden-plot";
        hidden.setAttribute("aria-hidden", "true");
        document.body.appendChild(hidden);

        button.classList.add("inflation-png-busy");
        const oldText = button.textContent;
        button.textContent = "…";
        try {
            await window.Plotly.newPlot(
                hidden,
                [trace],
                layout,
                {
                    staticPlot: true,
                    displayModeBar: false,
                    responsive: false
                }
            );
            const dataUrl = await window.Plotly.toImage(
                hidden,
                {
                    format: "png",
                    width: EXPORT_WIDTH,
                    height: height,
                    scale: EXPORT_SCALE
                }
            );
            triggerDownload(
                dataUrl,
                slugify(title, "dashboard_table") + "_" + nowStamp() + ".png"
            );
        } finally {
            try {
                window.Plotly.purge(hidden);
            } catch (_) {}
            hidden.remove();
            button.classList.remove("inflation-png-busy");
            button.textContent = oldText;
        }
    }

    function tableToolbarFor(targetRoot) {
        const previous = targetRoot.previousElementSibling;
        if (
            previous
            && previous.classList
            && previous.classList.contains("inflation-png-table-toolbar")
        ) {
            return previous;
        }

        const toolbar = document.createElement("div");
        toolbar.className = "inflation-png-table-toolbar";

        const button = document.createElement("button");
        button.type = "button";
        button.className = "inflation-png-button";
        button.textContent = "↓ PNG";
        button.title =
            "Download the currently rendered table as a clean report PNG";
        button.setAttribute(
            "aria-label",
            "Download current table as PNG"
        );

        button.addEventListener("click", function (event) {
            event.preventDefault();
            event.stopPropagation();

            // Resolve the current React-rendered target at click time.
            let current = toolbar.nextElementSibling;
            if (!current) current = targetRoot;

            exportTable(current, button).catch(function (err) {
                console.error("PNG table export failed:", err);
                const oldText = button.textContent;
                button.textContent = "Error";
                window.setTimeout(function () {
                    button.textContent = oldText;
                }, 1800);
            });
        });

        toolbar.appendChild(button);
        if (targetRoot.parentNode) {
            targetRoot.parentNode.insertBefore(toolbar, targetRoot);
        }
        return toolbar;
    }

    function scanGraphs() {
        document.querySelectorAll(".js-plotly-plot").forEach(function (plot) {
            const root =
                plot.closest(".dash-graph")
                || plot.parentElement;
            if (!root) return;
            makeGraphButton(root);
        });
    }

    function scanTables() {
        // Dash DataTables first: bind one exporter to the complete component
        // rather than to each internal fixed-header/fixed-column table.
        document.querySelectorAll(".dash-spreadsheet-container").forEach(
            function (root) {
                if (root.closest(".inflation-export-hidden-plot")) return;
                if (!largestVisibleTable(root)) return;
                tableToolbarFor(root);
            }
        );

        // Native html.Table and other regular tables.
        document.querySelectorAll("table").forEach(function (table) {
            if (table.closest(".dash-spreadsheet-container")) return;
            if (table.closest(".inflation-export-hidden-plot")) return;
            if (table.closest("[data-png-export-ignore='1']")) return;
            if (!visible(table)) return;
            tableToolbarFor(table);
        });
    }

    let scanScheduled = false;
    function scheduleScan() {
        if (scanScheduled) return;
        scanScheduled = true;
        window.requestAnimationFrame(function () {
            scanScheduled = false;
            scanGraphs();
            scanTables();
        });
    }

    function start() {
        scheduleScan();
        const observer = new MutationObserver(function () {
            scheduleScan();
        });
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });

        // Layout-only changes can expose a page without creating new DOM.
        window.addEventListener("resize", scheduleScan);
        document.addEventListener("click", function () {
            window.setTimeout(scheduleScan, 0);
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", start, {once: true});
    } else {
        start();
    }
})();
