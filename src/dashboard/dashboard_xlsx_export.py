from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from xml.sax.saxutils import escape


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _excel_col(n: int) -> str:
    out = ""
    while n:
        n, rem = divmod(n - 1, 26)
        out = chr(65 + rem) + out
    return out


def _cell_xml(ref: str, value: Any, style: int = 0) -> str:
    style_attr = f' s="{style}"' if style else ""
    if isinstance(value, bool):
        return f'<c r="{ref}" t="b"{style_attr}><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            if value == value and abs(float(value)) != float("inf"):
                return f'<c r="{ref}"{style_attr}><v>{value}</v></c>'
        except Exception:
            pass
    text = _safe_text(value)
    return f'<c r="{ref}" t="inlineStr"{style_attr}><is><t xml:space="preserve">{escape(text)}</t></is></c>'


def _sheet_xml(rows: list[list[Any]], header_rows: int = 1) -> str:
    max_cols = max((len(r) for r in rows), default=1)
    widths = []
    for c in range(max_cols):
        mx = 8
        for row in rows[:5000]:
            if c < len(row):
                mx = max(mx, min(48, len(_safe_text(row[c])) + 2))
        widths.append(mx)
    cols = "".join(
        f'<col min="{i+1}" max="{i+1}" width="{w}" customWidth="1"/>'
        for i, w in enumerate(widths)
    )
    body=[]
    for r_idx,row in enumerate(rows, start=1):
        cells=[]
        for c_idx,value in enumerate(row, start=1):
            cells.append(_cell_xml(f"{_excel_col(c_idx)}{r_idx}", value, 1 if r_idx <= header_rows else 0))
        body.append(f'<row r="{r_idx}">{"".join(cells)}</row>')
    freeze = '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>' if header_rows else '<sheetViews><sheetView workbookViewId="0"/></sheetViews>'
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        + freeze + f'<cols>{cols}</cols><sheetData>{"".join(body)}</sheetData></worksheet>'
    )


def _styles_xml() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Calibri"/></font></fonts>
  <fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill></fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/></cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>'''


def _sanitize_sheet(name: str, used: set[str]) -> str:
    name = re.sub(r"[\\/*?:\[\]]", "_", str(name or "Sheet")).strip() or "Sheet"
    base = name[:31]
    candidate = base
    i=2
    while candidate.casefold() in used:
        suffix=f"_{i}"
        candidate=(base[:31-len(suffix)] + suffix)
        i+=1
    used.add(candidate.casefold())
    return candidate


def _sanitize_filename(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(text or "chart"))
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return text[:80] or "chart"


def _trace_rows(figure: dict[str, Any]) -> tuple[list[list[Any]], list[tuple[str, list[list[Any]]]]]:
    headers = ["trace_index", "trace_name", "trace_type", "point_index", "x", "y", "z", "text"]
    rows=[headers]
    tables=[]
    for ti,trace in enumerate(list(figure.get("data") or []), start=1):
        if not isinstance(trace, dict):
            continue
        ttype=str(trace.get("type") or "scatter")
        name=str(trace.get("name") or f"Trace {ti}")
        if ttype == "table":
            header=(trace.get("header") or {}).get("values") or []
            cells=(trace.get("cells") or {}).get("values") or []
            nrows=max((len(col) for col in cells if isinstance(col,list)), default=0)
            table_rows=[[_safe_text(x) for x in header]]
            for r in range(nrows):
                table_rows.append([col[r] if isinstance(col,list) and r < len(col) else "" for col in cells])
            tables.append((name, table_rows))
            continue
        x=trace.get("x") or []
        y=trace.get("y") or []
        z=trace.get("z") or []
        text=trace.get("text") or []
        if isinstance(z,list) and z and isinstance(z[0],list):
            xs=x if isinstance(x,list) else []
            ys=y if isinstance(y,list) else []
            for ri,zrow in enumerate(z):
                if not isinstance(zrow,list):
                    continue
                for ci,zv in enumerate(zrow):
                    rows.append([ti,name,ttype,f"{ri}:{ci}", xs[ci] if ci < len(xs) else ci, ys[ri] if ri < len(ys) else ri, zv, ""])
            continue
        n=max(len(x) if isinstance(x,list) else 0, len(y) if isinstance(y,list) else 0, len(text) if isinstance(text,list) else 0)
        for i in range(n):
            rows.append([ti,name,ttype,i, x[i] if isinstance(x,list) and i<len(x) else "", y[i] if isinstance(y,list) and i<len(y) else "", "", text[i] if isinstance(text,list) and i<len(text) else ""])
    return rows,tables


def build_chart_xlsx(payload: dict[str, Any]) -> tuple[bytes, str]:
    figure=dict(payload.get("figure") or {})
    vintage=str(payload.get("vintage") or "").strip()
    vintage_date=""
    if re.fullmatch(r"\d{8}", vintage):
        vintage_date=f"{vintage[:4]}-{vintage[4:6]}-{vintage[6:]}"
    layout=dict(figure.get("layout") or {})
    title=layout.get("title")
    if isinstance(title,dict): title=title.get("text")
    title=re.sub(r"<[^>]+>", "", str(title or payload.get("graph_id") or "Chart data"))
    exported=datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    metadata=[
        ["Field","Value"],
        ["Vintage", vintage],
        ["Vintage date", vintage_date],
        ["Chart id", str(payload.get("graph_id") or "")],
        ["Chart title", title],
        ["Exported UTC", exported],
    ]
    data_rows, table_sheets = _trace_rows(figure)
    sheets=[("Metadata",metadata,1),("Chart_Data",data_rows,1)]
    used={"metadata","chart_data"}
    for name, rows in table_sheets:
        sheets.append((_sanitize_sheet(name,used),rows,1))

    out=io.BytesIO()
    with zipfile.ZipFile(out,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>' + ''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1,len(sheets)+1)) + '</Types>')
        z.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        workbook_sheets=''.join(f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>' for i,(name,_,_) in enumerate(sheets,1))
        z.writestr("xl/workbook.xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'+workbook_sheets+'</sheets></workbook>')
        rels=''.join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1,len(sheets)+1)) + f'<Relationship Id="rId{len(sheets)+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        z.writestr("xl/_rels/workbook.xml.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+rels+'</Relationships>')
        z.writestr("xl/styles.xml", _styles_xml())
        for i,(_,rows,header_rows) in enumerate(sheets,1):
            z.writestr(f"xl/worksheets/sheet{i}.xml", _sheet_xml(rows, header_rows))
    filename=f"{_sanitize_filename(title)}_vintage_{vintage or 'unknown'}.xlsx"
    return out.getvalue(), filename


def build_export_xlsx(payload: dict[str, Any]) -> tuple[bytes, str]:
    if payload.get("table") is not None:
        vintage = str(payload.get("vintage") or "").strip()
        vintage_date = ""
        if re.fullmatch(r"\d{8}", vintage):
            vintage_date = f"{vintage[:4]}-{vintage[4:6]}-{vintage[6:]}"
        table = dict(payload.get("table") or {})
        headers = list(table.get("headers") or [])
        rows = [list(row) for row in (table.get("rows") or [])]
        title = str(payload.get("title") or payload.get("table_id") or "Table")
        exported = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        metadata = [
            ["Field", "Value"],
            ["Vintage", vintage],
            ["Vintage date", vintage_date],
            ["Table id", str(payload.get("table_id") or "")],
            ["Table title", title],
            ["Exported UTC", exported],
        ]
        data_rows = [headers] + rows if headers else rows
        sheets = [("Metadata", metadata, 1), ("Table_Data", data_rows, 1 if headers else 0)]
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>' + ''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1, len(sheets) + 1)) + '</Types>')
            z.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
            workbook_sheets = ''.join(f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>' for i, (name, _, _) in enumerate(sheets, 1))
            z.writestr("xl/workbook.xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>' + workbook_sheets + '</sheets></workbook>')
            rels = ''.join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1, len(sheets) + 1)) + f'<Relationship Id="rId{len(sheets)+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
            z.writestr("xl/_rels/workbook.xml.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + rels + '</Relationships>')
            z.writestr("xl/styles.xml", _styles_xml())
            for i, (_, sheet_rows, header_rows) in enumerate(sheets, 1):
                z.writestr(f"xl/worksheets/sheet{i}.xml", _sheet_xml(sheet_rows, header_rows))
        return out.getvalue(), f"{_sanitize_filename(title)}_vintage_{vintage or 'unknown'}.xlsx"
    return build_chart_xlsx(payload)


def register_xlsx_export(server) -> None:
    """Register one generic endpoint used by the dashboard XLSX client asset."""
    endpoint = "dashboard_xlsx_export_v1"
    if endpoint in getattr(server, "view_functions", {}):
        return

    from flask import Response, request

    @server.route("/_dashboard/export-xlsx", methods=["POST"], endpoint=endpoint)
    def _download_xlsx():
        raw = request.get_data(cache=False)
        if len(raw) > 25_000_000:
            return Response("Export payload is too large.", status=413, mimetype="text/plain")
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
            if not isinstance(payload, dict):
                raise ValueError("JSON payload must be an object.")
            content, filename = build_export_xlsx(payload)
        except Exception as exc:
            return Response(
                f"XLSX export failed: {type(exc).__name__}: {exc}",
                status=400,
                mimetype="text/plain",
            )
        return Response(
            content,
            status=200,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )


__all__ = ["build_chart_xlsx", "build_export_xlsx", "register_xlsx_export"]
