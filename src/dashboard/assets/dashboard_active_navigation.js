/* Inflation Dashboard — active navigation state v1
   Presentation-only. Marks the current global page, domain and sub-page. */
(function () {
    "use strict";
    if (window.__inflationActiveNavigationV1) return;
    window.__inflationActiveNavigationV1 = true;

    function normalise(path) {
        let p = String(path || "/overview").split("?")[0].split("#")[0];
        if (!p || p === "/") return "/overview";
        if (p.length > 1 && p.endsWith("/")) p = p.slice(0, -1);
        const aliases = {
            "/headline": "/headline/forecast",
            "/headline/overview": "/headline/forecast",
            "/headline/contributions": "/headline/forecast",
            "/headline/components": "/headline/forecast",
            "/headline/diagnostics": "/headline/estimation",
            "/core": "/core/forecast"
        };
        return aliases[p] || p;
    }

    function hrefPath(link) {
        try {
            return normalise(new URL(link.href, window.location.origin).pathname);
        } catch (_) {
            return normalise(link.getAttribute("href"));
        }
    }

    function domainFor(path) {
        if (path.startsWith("/headline")) return "headline";
        if (path.startsWith("/core")) return "core";
        if (["/forecast", "/aggregate", "/scenarios", "/structural", "/estimation"].includes(path)) {
            return "energy";
        }
        return null;
    }

    function update() {
        const path = normalise(window.location.pathname);
        const domain = domainFor(path);

        document.querySelectorAll("a.nav-link").forEach(function (link) {
            const active = hrefPath(link) === path;
            link.classList.toggle("is-active", active);
            if (active) link.setAttribute("aria-current", "page");
            else link.removeAttribute("aria-current");
        });

        document.querySelectorAll("a.domain-pill").forEach(function (link) {
            const label = (link.textContent || "").trim().toLowerCase();
            const linkDomain = label.indexOf("headline") >= 0
                ? "headline"
                : label.indexOf("core") >= 0
                    ? "core"
                    : "energy";
            const active = domain !== null && linkDomain === domain;
            link.classList.toggle("is-active", active);
            if (active) link.setAttribute("aria-current", "true");
            else link.removeAttribute("aria-current");
        });
    }

    function schedule() {
        window.requestAnimationFrame(update);
    }

    ["pushState", "replaceState"].forEach(function (name) {
        const original = window.history[name];
        if (typeof original !== "function") return;
        window.history[name] = function () {
            const result = original.apply(this, arguments);
            window.dispatchEvent(new Event("inflation-location-change"));
            return result;
        };
    });

    window.addEventListener("popstate", schedule);
    window.addEventListener("inflation-location-change", schedule);
    document.addEventListener("click", function (event) {
        if (event.target && event.target.closest && event.target.closest("a.nav-link,a.domain-pill")) {
            window.setTimeout(schedule, 0);
        }
    });

    const observer = new MutationObserver(schedule);
    function start() {
        update();
        observer.observe(document.body, {childList: true, subtree: true});
    }
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", start, {once: true});
    } else {
        start();
    }
})();
