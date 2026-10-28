"""Public read-only API safeguards and RAG evidence pages."""
import collections
import html
import json
import os
import re
import sqlite3
import threading
import time
from urllib.parse import quote
from flask import request, jsonify, Response
import markdown
import bleach


def register_public_api(app, engine):
    origins = set(os.getenv('MFG_ALLOWED_ORIGINS', os.getenv('ALLOWED_ORIGINS', 'https://pray2god4u.github.io')).split(','))
    origins = {x.strip().rstrip('/') for x in origins if x.strip()}
    lock = threading.Lock()
    calls = collections.defaultdict(collections.deque)
    public_base = os.getenv('API_PUBLIC_BASE_URL', '').rstrip('/')

    @app.before_request
    def validate_public_request():
        if not request.path.startswith('/api/'):
            return None
        origin = request.headers.get('Origin')
        if origin and origin.rstrip('/') not in origins:
            return jsonify(error='허용되지 않은 접근 출처입니다', status='forbidden'), 403
        if request.method == 'OPTIONS':
            return Response(status=204)
        if os.getenv('MFG_PUBLIC_MODE') == '1' and request.path.startswith('/api/document/'):
            doc_id = request.path.removeprefix('/api/document/')
            data = json.loads(engine._get_document(doc_id, full=False))
            return jsonify(data), 404 if data.get('error') else 200
        if request.path in ('/api/semantic', '/api/hybrid', '/api/search', '/api/sections'):
            query = request.args.get('q', '')
            if len(query) > 1024:
                return jsonify(error='검색 질문은 1,024자 이내로 입력해 주세요', status='invalid'), 400
            if request.args.get('dtype', '') not in ('', '가이드', '설명서'):
                return jsonify(error='문서 유형은 설명서 또는 가이드입니다', status='invalid'), 400
            try:
                limit = int(request.args.get('limit', 10))
                if not 1 <= limit <= 30:
                    raise ValueError()
            except ValueError:
                return jsonify(error='결과 수는 1~30 사이 정수여야 합니다', status='invalid'), 400
        if request.path in ('/api/semantic', '/api/hybrid'):
            if not request.args.get('q', '').strip():
                return jsonify(error='검색 질문을 입력해 주세요', status='invalid'), 400
            now = time.monotonic()
            with lock:
                if len(calls) > 10000:
                    for key in list(calls):
                        if not calls[key] or now - calls[key][-1] >= 60:
                            del calls[key]
                key = request.remote_addr or 'unknown'
                queue = calls[key]
                while queue and now - queue[0] >= 60:
                    queue.popleft()
                if len(queue) >= int(os.getenv('SEARCH_REQUESTS_PER_MINUTE', '20')):
                    response = jsonify(error='검색 요청이 많습니다. 잠시 후 다시 시도해 주세요', status='rate_limited')
                    response.headers['Retry-After'] = '60'
                    return response, 429
                queue.append(now)

    @app.after_request
    def public_headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        origin = request.headers.get('Origin')
        if request.path.startswith('/api/'):
            response.headers['Cache-Control'] = 'no-store'
            response.headers.add('Vary', 'Origin')
            if origin and origin.rstrip('/') in origins:
                response.headers['Access-Control-Allow-Origin'] = origin
                response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
                response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            if response.is_json and request.path in ('/api/semantic', '/api/hybrid'):
                payload = response.get_json(silent=True)
                if isinstance(payload, dict):
                    for row in payload.get('results', []):
                        doc = row.get('doc_id', '')
                        path = '/evidence/' + quote(doc, safe='')
                        chunk = row.get('chunk_id')
                        sec = row.get('sec_idx')
                        params = '?chunk=' + str(chunk) if chunk is not None else '?section=' + str(sec)
                        if chunk is None and sec is not None:
                            with engine.db() as src:
                                raw = src.execute('SELECT tx FROM sections WHERE doc_id=? AND sec_idx=?', (doc, sec)).fetchone()
                            snip = row.get('snip', '')
                            if raw and snip:
                                start = raw['tx'].find(snip)
                                if start >= 0:
                                    params += '&start=' + str(start) + '&end=' + str(start + len(snip))
                        row['evidence_url'] = (public_base or request.url_root.rstrip('/')) + path + params
                    payload['backend_note'] = '공개 기술문서 근거 검색이며 현장 고장 진단 결과가 아닙니다.'
                    response.set_data(json.dumps(payload, ensure_ascii=False, allow_nan=False))
                    if payload.get('status') == 'unavailable':
                        response.status_code = 503
        return response

    @app.get('/api/health')
    def public_health():
        data = json.loads(engine._get_stats())
        vector = data.get('vector_index', {})
        ok = bool(data.get('total_docs')) and vector.get('status') == 'built' and vector.get('source_verified')
        return jsonify(status='ok' if ok else 'not_ready', corpus_docs=data.get('total_docs'),
                       vector_index=vector,
                       note='메타데이터 준비 상태입니다. 검색 실행 성공은 별도 확인해야 합니다.'), 200 if ok else 503

    @app.get('/api/status')
    def public_status():
        import importlib.util
        data = json.loads(engine._get_stats())
        vector = data.get('vector_index', {})
        warnings = []
        if vector.get('status') != 'built' or not vector.get('source_verified'):
            warnings.append('벡터 인덱스 또는 원문 검증이 준비되지 않았습니다')
        for dependency in ('sentence_transformers', 'huggingface_hub', 'torch'):
            if importlib.util.find_spec(dependency) is None:
                warnings.append('서버 의존성 누락: ' + dependency)
        return jsonify(status=('unavailable' if not data.get('total_docs') else 'degraded' if warnings else 'ok'),
                       corpus={'datasets':data.get('datasets'), 'documents':data.get('total_docs'), 'sections':data.get('total_sections')},
                       vector_index=vector, warnings=warnings,
                       note='인덱스와 의존성 상태이며, 실제 검색 실행 성공은 별도 확인해야 합니다.')

    @app.get('/evidence/<doc_id>')
    def evidence_page(doc_id):
        if not re.fullmatch(r'[0-9]+-(가이드|설명서)', doc_id):
            return Response('문서를 찾을 수 없습니다', status=404)
        con = engine.db()
        try:
            doc = con.execute('SELECT * FROM docs WHERE id=?', (doc_id,)).fetchone()
            if not doc:
                return Response('문서를 찾을 수 없습니다', status=404)
            chunk = request.args.get('chunk')
            section = request.args.get('section')
            try:
                chunk_id = int(chunk) if chunk is not None else None
                sec_idx = int(section) if section is not None else None
            except ValueError:
                return Response('근거 구간 번호가 올바르지 않습니다', status=400)
            excerpts = []
            if chunk_id is not None:
                if not os.path.exists(engine.VEC_DB_PATH):
                    return Response('청크 인덱스가 준비되지 않았습니다', status=503)
                with sqlite3.connect('file:' + engine.VEC_DB_PATH + '?mode=ro', uri=True) as vec:
                    vec.row_factory = sqlite3.Row
                    row = vec.execute('SELECT * FROM chunks WHERE chunk_id=? AND doc_id=?', (chunk_id, doc_id)).fetchone()
                if not row:
                    return Response('해당 문서의 근거 청크를 찾을 수 없습니다', status=404)
                raw = con.execute('SELECT tx FROM sections WHERE doc_id=? AND sec_idx=?', (doc_id, row['sec_idx'])).fetchone()
                if not raw or raw['tx'][row['start_char']:row['end_char']] != row['raw_text']:
                    return Response('원문과 청크가 일치하지 않습니다', status=503)
                excerpts.append((row['label'], row['raw_text'], f"청크 {chunk_id} · 원문 위치 {row['start_char']}–{row['end_char']}"))
            else:
                sql = 'SELECT sec_idx,label,tx FROM sections WHERE doc_id=?'
                args = [doc_id]
                if sec_idx is not None:
                    sql += ' AND sec_idx=?'
                    args.append(sec_idx)
                sql += ' ORDER BY sec_idx LIMIT 30'
                rows = con.execute(sql, args).fetchall()
                if sec_idx is not None and not rows:
                    return Response('해당 문서의 구간을 찾을 수 없습니다', status=404)
                for row in rows:
                    text = row['tx'] or ''
                    start, end = 0, min(len(text), 1600)
                    if request.args.get('start') is not None or request.args.get('end') is not None:
                        try:
                            start = int(request.args['start'])
                            end = int(request.args['end'])
                            if sec_idx is None or not 0 <= start < end <= len(text) or end-start > 5000:
                                raise ValueError()
                        except (ValueError, KeyError):
                            return Response('근거 원문 위치가 올바르지 않습니다', status=400)
                    excerpts.append((row['label'], text[start:end], '섹션 ' + str(row['sec_idx']) + f' · 원문 위치 {start}–{end}'))
        finally:
            con.close()
        title = html.escape(doc['title'])
        pieces = []
        for label, text, location in excerpts:
            rendered = bleach.clean(markdown.markdown(text or '', extensions=['tables', 'fenced_code']),
                tags=['p','br','strong','em','ul','ol','li','table','thead','tbody','tr','th','td','pre','code','h1','h2','h3','h4','blockquote'],
                attributes={'th':['rowspan','colspan'],'td':['rowspan','colspan']}, strip=True)
            pieces.append('<section><h2>' + html.escape(label or '(서두)') + '</h2><p>' + html.escape(location) +
                          '</p>' + rendered + '<details><summary>저장된 원문 텍스트 보기</summary><pre>' + html.escape(text or '') + '</pre></details></section>')
        page = '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + title + '</title><style>body{font:16px/1.7 system-ui,sans-serif;margin:2rem auto;padding:0 1rem;max-width:1000px;color:#1b273a}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6fa;padding:1rem;border-radius:8px}section{margin:2rem 0}.note{background:#fff5d6;padding:1rem}h1{font-size:1.7rem}</style><body><p>제조 Copilot · RAG 근거 열람</p><h1>' + title + '</h1><p class="note">RAG에 저장된 공개 기술문서의 근거 발췌입니다. 검색 결과만으로 설비 고장이나 화재 발생을 확정할 수 없습니다. 원본 파일·이미지 전체를 재배포하는 페이지가 아닙니다.</p>' + ''.join(pieces) + '<footer>원출처: AI Hub 공개 기술문서 · 검색 인덱스의 문서·구간 정보와 대조됨</footer></body></html>'
        return Response(page, content_type='text/html; charset=utf-8')
