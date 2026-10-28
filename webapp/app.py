#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
제조 Copilot 웹앱 — mfg-rag 검색 엔진 재사용 (Flask)
- rag/server.py 의 검색 본체 함수를 그대로 호출 (동일 인덱스·동의어 로직)
- 디자인: public-sales-insight-coach 디자인 시스템 적용
- 실행: uv run --no-project --python 3.12 --with flask python app.py
"""
import os, sys, json, types, importlib.util
from pathlib import Path
from flask import Flask, render_template, request, abort, Response

ROOT = Path(__file__).resolve().parent
RAG_DIR = ROOT.parent / "rag"


def load_engine():
    """rag/server.py 를 mcp 패키지 설치 없이 로드하기 위한 스텁 주입"""
    mcp_pkg = types.ModuleType("mcp")
    srv = types.ModuleType("mcp.server")
    fm = types.ModuleType("mcp.server.fastmcp")

    class _StubMCP:
        def __init__(self, *a, **k):
            pass

        def tool(self, *a, **k):
            def deco(f):
                return f
            return deco

        def run(self):
            pass

    fm.FastMCP = _StubMCP
    for name, mod in (("mcp", mcp_pkg), ("mcp.server", srv), ("mcp.server.fastmcp", fm)):
        sys.modules[name] = mod
    spec = importlib.util.spec_from_file_location("mfg_engine", RAG_DIR / "server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


engine = load_engine()
app = Flask(__name__)


def J(s):
    return json.loads(s)


def raw_json(s):
    return Response(s, mimetype="application/json")


@app.template_filter("sizefmt")
def sizefmt(n):
    n = int(n or 0)
    if n >= 1024 * 1024:
        return f"{n / 1024 / 1024:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.0f} KB"
    return f"{n} B"


@app.template_filter("num")
def numfmt(n):
    return f"{int(n or 0):,}"


# ---------- 페이지 ----------
@app.route("/")
def home():
    stats = J(engine._get_stats())
    return render_template("index.html", stats=stats)


@app.route("/search")
def search_page():
    return render_template("search.html", q=request.args.get("q", ""))


@app.route("/datasets")
def datasets_page():
    data = J(engine._list_datasets())
    return render_template("datasets.html", datasets=data["datasets"], count=data["dataset_count"])


@app.route("/dataset/<sn>")
def dataset_page(sn):
    con = engine.db()
    docs = [dict(r) for r in con.execute(
        "SELECT id, title, dtype, nsec, nbytes, source_file FROM docs WHERE sn = ? ORDER BY dtype DESC", (sn,))]
    name = ""
    if docs:
        row = con.execute(
            "SELECT COALESCE(MAX(CASE WHEN dtype='설명서' THEN ds_name END), MAX(ds_name)) FROM docs WHERE sn = ?",
            (sn,)).fetchone()
        name = row[0] or ""
        labels = [r[0] for r in con.execute(
            "SELECT DISTINCT s.label FROM sections s JOIN docs d ON d.id = s.doc_id "
            "WHERE d.sn = ? AND s.label != '(서두)' ORDER BY s.sec_idx LIMIT 24", (sn,))]
    else:
        labels = []
    con.close()
    if not docs:
        abort(404)
    total_sec = sum(d["nsec"] for d in docs)
    total_bytes = sum(d["nbytes"] for d in docs)
    return render_template("dataset.html", sn=sn, name=name, docs=docs,
                           labels=labels, total_sec=total_sec, total_bytes=total_bytes)


@app.route("/document/<doc_id>")
def document_page(doc_id):
    doc = J(engine._get_document(doc_id, full=False))
    if "error" in doc:
        abort(404)
    return render_template("document.html", doc=doc)


@app.route("/demo")
def demo_page():
    return render_template("demo.html")


# ---------- API (검색 엔진 그대로) ----------
@app.route("/api/stats")
def api_stats():
    return raw_json(engine._get_stats())


@app.route("/api/search")
def api_search():
    return raw_json(engine._search_docs(
        request.args.get("q", ""), request.args.get("dtype", ""),
        request.args.get("sn", ""), request.args.get("limit", 10)))


@app.route("/api/sections")
def api_sections():
    return raw_json(engine._search_sections(
        request.args.get("q", ""), request.args.get("dtype", ""),
        request.args.get("sn", ""), request.args.get("limit", 10)))


@app.route("/api/recommend")
def api_recommend():
    return raw_json(engine._recommend_datasets(
        request.args.get("q", ""), request.args.get("limit", 5)))


@app.route("/api/compare")
def api_compare():
    return raw_json(engine._compare_datasets(
        request.args.get("a", ""), request.args.get("b", "")))


@app.route("/api/hybrid")
def api_hybrid():
    try:
        limit = int(request.args.get('limit', 10))
        return raw_json(engine._hybrid_search(request.args.get('q', ''),
                        request.args.get('dtype', ''), request.args.get('sn', ''), limit))
    except ValueError:
        return raw_json(json.dumps({'error': 'limit는 정수여야 합니다', 'results': []}, ensure_ascii=False)), 400


@app.route("/api/semantic")
def api_semantic():
    try:
        limit = int(request.args.get('limit', 10))
        return raw_json(engine._semantic_search(request.args.get('q', ''),
                        request.args.get('dtype', ''), request.args.get('sn', ''), limit))
    except ValueError:
        return raw_json(json.dumps({'error': 'limit는 정수여야 합니다', 'results': []}, ensure_ascii=False)), 400


@app.route("/api/document/<doc_id>")
def api_document(doc_id):
    full = request.args.get("full", "0") in ("1", "true", "yes")
    return raw_json(engine._get_document(doc_id, full=full))


@app.route("/api/datasets")
def api_datasets():
    return raw_json(engine._list_datasets())


@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", code=404, msg="문서를 찾을 수 없습니다."), 404


sys.path.insert(0, str(Path(__file__).resolve().parent))
from public_api import register_public_api
register_public_api(app, engine)
if os.environ.get('MFG_TRUST_PROXY') == '1':
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", "8142")), debug=False)
