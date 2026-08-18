/* Inflation Dashboard — active navigation state v1
   Presentation-only. Marks the current global page, domain and sub-page. */
(function () {
    "use strict";
    if (window.__inflationActiveNavigationV1) return;
    window.__inflationActiveNavigationV1 = true;

    function normalise(pathname) {
    var path = String(pathname || "/").split("?")[0].replace(/\/+$/, "") || "/";
    var aliases = {
        "/": "/overview",
        "/aggregate": "/forecast",
        "/forecast/aggregate": "/forecast",
        "/headline": "/forecast/headline",
        "/headline/overview": "/forecast/headline",
        "/headline/forecast": "/forecast/headline",
        "/headline/contributions": "/forecast/headline",
        "/headline/components": "/forecast/headline",
        "/headline/scenarios": "/scenarios/headline",
        "/headline/structural": "/structural/headline",
        "/headline/diagnostics": "/estimation/headline",
        "/headline/estimation": "/estimation/headline",
        "/core": "/forecast/core",
        "/core/forecast": "/forecast/core",
        "/core/scenarios": "/scenarios/core",
        "/core/structural": "/structural/core",
        "/core/estimation": "/estimation/core"
    };
    return aliases[path] || path;
}

    function hrefPath(link) {
        try {
            return normalise(new URL(link.href, window.location.origin).pathname);
        } catch (_) {
            return normalise(link.getAttribute("href"));
        }
    }

    function domainFor(pathname) {
    var path = String(pathname || "/").split("?")[0].replace(/\/+$/, "") || "/";
    var parts = path.toLowerCase().split("/").filter(Boolean);
    if (parts.length && (parts[0] === "headline" || parts[0] === "core")) {
        return parts[0];
    }
    if (parts.length >= 2 && (parts[1] === "headline" || parts[1] === "core")) {
        return parts[1];
    }
    return "energy";
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
