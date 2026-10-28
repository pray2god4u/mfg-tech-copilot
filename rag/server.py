#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mfg-rag MCP 서버 — AI Hub 기술문서 RAG 질의 도구 (제조 중심 전분야)
- SQLite FTS5(trigram) 인덱스: ~/.oneai/workspace/mfg-tech-copilot/rag/mfg_corpus.db
- 코퍼스: 데이터셋 83종 · 설명서/활용가이드 153건 · 섹션 5,101개
- 도구: search_docs, search_sections, get_document, list_datasets, get_stats,
       recommend_datasets, compare_datasets, semantic_search(벡터), hybrid_search(하이브리드)
- 동의어 확장 검색(예: 화재→열폭주·발화·탄화), 3글자 미만 단어는 LIKE 스캔
"""
import os, re, json, sqlite3, sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vector_runtime
import retrieval_v2
from mcp.server.fastmcp import FastMCP

DB_PATH = os.path.expanduser(os.getenv('MFG_CORPUS_PATH', str(Path(__file__).resolve().parent / 'mfg_corpus_20261026_expanded_2.db')))
VEC_DB_PATH = os.path.expanduser(os.getenv('MFG_VECTOR_PATH', str(Path(__file__).resolve().parent / 'mfg_chunks_20261028_v2.db')))
VEC_MODEL = "BAAI/bge-m3"
mcp = FastMCP("mfg-rag")

# 범용 단어 — 검색 그룹에서 제외 (모든 문서에 걸려 노이즈)
STOPWORDS = {
    "데이터", "데이터셋", "자료", "구축", "구성", "활용", "관련", "위한", "용도",
    "제공", "수집", "구축된", "대한", "있는", "있음", "필요", "가능", "사용",
}

# 제조·AI 데이터 도메인 동의어 사전 — 토큰별 대안어 (FTS OR 확장에 사용)
SYNONYMS = {
    "예지보전": ["예측정비", "예방정비", "PHM", "상태감시"],
    "예측": ["예지", "추정", "예측모델"],
    "고장": ["결함", "이상", "고장진단", "fault"],
    "불량": ["결함", "이물", "스크래치"],
    "검사": ["검증", "품질검사", "측정"],
    "품질": ["불량", "결함", "품질검사"],
    "화재": ["열폭주", "발화", "탄화", "연기"],
    "열폭주": ["발화", "화재", "과열", "탄화"],
    "탄화": ["발화", "화재", "카본화"],
    "과열": ["발열", "온도상승", "열폭주"],
    "배터리": ["이차전지", "리튬이온", "셀"],
    "이차전지": ["배터리", "리튬이온"],
    "수명": ["잔존수명", "수명예측", "RUL", "SOH"],
    "이송장치": ["OHT", "AGV", "반송", "운반"],
    "OHT": ["이송장치", "반송", "호이스트"],
    "AGV": ["이송장치", "무인운반", "주행"],
    "반송": ["이송", "운반", "OHT"],
    "용접": ["용접로봇", "용단", "비드"],
    "조립": ["체결", "결합", "피팅"],
    "센서": ["계측", "측정", "모니터링"],
    "진동": ["가속도", "소음", "가진"],
    "온도": ["열화상", "열", "발열"],
    "열화상": ["온도", "적외선", "서멀"],
    "이미지": ["영상", "사진", "카메라"],
    "영상": ["이미지", "동영상", "비디오"],
    "라벨링": ["라벨", "태깅", "정답", "주석"],
    "라벨": ["라벨링", "태깅", "클래스"],
    "검수": ["품질검증", "교차검증", "확인"],
    "구축": ["수집", "구성", "제작"],
    "멀티모달": ["복합", "융합", "다중"],
    "실장": ["SMT", "표면실장", "마운팅"],
    "SMT": ["실장", "표면실장"],
    "로봇": ["매니퓰레이터", "티칭", "팔"],
    "티칭": ["교시", "경로생성", "로봇"],
    "설비": ["기계", "장비", "기기"],
    "공정": ["라인", "제조", "단계"],
    "예측모델": ["학습모델", "AI모델", "딥러닝"],
    "딥러닝": ["신경망", "학습", "AI"],
    "데이터셋": ["데이터", "코퍼스"],
    "P&ID": ["배관", "계장", "도면"],
    "OCR": ["문자인식", "글자인식", "판독"],
}

def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def expand_query(q: str):
    """동의어 확장 쿼리 파싱.
    반환: (fts_groups, like_groups)
      - fts_groups: FTS MATCH 문자열 목록 (그룹 간 AND, 그룹 내 동의어 OR).
        토큰이 3글자 이상이면 FTS 그룹 — 단, 3글자 미만 동의어는 제외
        (trigram은 2글자 prefix 불가, MATCH는 OR 절 안에 쓸 수 없음)
      - like_groups: LIKE 대안어 목록의 목록 (그룹 간 AND, 그룹 내 OR).
        2글자 토큰은 원래 단어 + 동의어 전부를 LIKE OR로 검색
    """
    q = re.sub(r"[^\w\s]", " ", q).strip()  # 단어문자(한글·영숫자·_)와 공백만 유지
    fts_groups, like_groups = [], []
    for t in q.split():
        if t in STOPWORDS:
            continue
        syns = SYNONYMS.get(t, [])
        if len(t) >= 3:
            long_syns = [s for s in syns if len(s) >= 3]
            fts_groups.append("(" + " OR ".join(s + "*" for s in [t] + long_syns) + ")")
        elif len(t) >= 2:
            alts = [t] + [s for s in syns if len(s) >= 2]
            like_groups.append(alts)
    return (fts_groups, like_groups)

def _apply_groups(sql: str, args: list, fts_col: str, like_col: str,
                  fts_groups, like_groups):
    """확장 쿼리 그룹을 WHERE 절에 적용 (그룹 간 AND, 그룹 내 OR)"""
    for g in fts_groups:
        sql += f" AND {fts_col} MATCH ?"
        args.append(g)
    for group in like_groups:
        sql += " AND (" + " OR ".join(f"{like_col} LIKE ? ESCAPE '\\'" for _ in group) + ")"
        args.extend("%" + t + "%" for t in group)
    return sql, args

def _search_docs(query: str, dtype: str = "", sn: str = "", limit: int = 10) -> str:
    """문서 단위 검색 본체"""
    con = db()
    fts_groups, like_groups = expand_query(query)
    limit = max(1, min(int(limit), 30))
    if not fts_groups and not like_groups:
        con.close()
        return json.dumps({"error": "검색어가 비어 있거나 특수문자뿐입니다"}, ensure_ascii=False)
    sql = """
    SELECT d.id, d.sn, d.title, d.dtype, d.ds_name, d.nsec,
           snippet(docs_fts, 3, '『', '』', '…', 40) AS snip,
           bm25(docs_fts) AS score
    FROM docs_fts
    JOIN docs_text ON docs_text.rowid = docs_fts.rowid
    JOIN docs d ON d.id = docs_text.id
    WHERE 1=1
    """
    args = []
    sql, args = _apply_groups(sql, args, "docs_fts", "docs_text.tx",
                              fts_groups, like_groups)
    if dtype:
        sql += " AND d.dtype = ?"; args.append(dtype)
    if sn:
        sql += " AND d.sn = ?"; args.append(sn)
    sql += " ORDER BY score LIMIT ?"; args.append(limit)
    rows = [dict(r) for r in con.execute(sql, args)]
    con.close()
    if not rows:
        fallback = json.loads(retrieval_v2.lexical_search(_engine_context(), query, dtype, sn, int(limit)))
        seen = set()
        for candidate in fallback.get('results', []):
            if candidate['doc_id'] in seen:
                continue
            seen.add(candidate['doc_id'])
            with db() as src:
                doc = src.execute('SELECT id,sn,title,dtype,ds_name,nsec FROM docs WHERE id=?',
                                  (candidate['doc_id'],)).fetchone()
            rows.append({**dict(doc), 'snip': candidate['snip'], 'score': candidate['score'],
                         'source_url': candidate['source_url']})
    return json.dumps({"query": query, "shown": len(rows), "results": rows}, ensure_ascii=False)

def _search_sections(query: str, dtype: str = "", sn: str = "", limit: int = 10) -> str:
    """섹션 단위 검색 본체"""
    con = db()
    fts_groups, like_groups = expand_query(query)
    limit = max(1, min(int(limit), 30))
    if not fts_groups and not like_groups:
        con.close()
        return json.dumps({"error": "검색어가 비어 있거나 특수문자뿐입니다"}, ensure_ascii=False)
    sql = """
    SELECT s.doc_id, d.title, d.dtype, d.ds_name, s.sec_idx, s.level, s.label,
           snippet(sec_fts, 1, '『', '』', '…', 48) AS snip,
           bm25(sec_fts) AS score
    FROM sec_fts JOIN sections s ON s.rowid = sec_fts.rowid
    JOIN docs d ON d.id = s.doc_id
    WHERE 1=1
    """
    args = []
    sql, args = _apply_groups(sql, args, "sec_fts", "s.tx",
                              fts_groups, like_groups)
    if dtype:
        sql += " AND d.dtype = ?"; args.append(dtype)
    if sn:
        sql += " AND d.sn = ?"; args.append(sn)
    sql += " ORDER BY score LIMIT ?"; args.append(limit)
    rows = [dict(r) for r in con.execute(sql, args)]
    con.close()
    if not rows:
        return retrieval_v2.lexical_search(_engine_context(), query, dtype, sn, int(limit))
    return json.dumps({"query": query, "shown": len(rows), "results": rows}, ensure_ascii=False)

def _get_document(doc_id: str, full: bool = False) -> str:
    """문서 1건 전체 조회 본체"""
    con = db()
    r = con.execute("SELECT * FROM docs WHERE id = ?", (doc_id,)).fetchone()
    if not r:
        con.close()
        return json.dumps({"error": f"문서 없음: {doc_id}", "hint": "list_datasets 으로 id 확인"}, ensure_ascii=False)
    doc = dict(r)
    max_len = 1000000 if full else 200
    secs = [dict(s) for s in con.execute(
        "SELECT sec_idx, level, label, tx FROM sections WHERE doc_id = ? ORDER BY sec_idx",
        (doc_id,))]
    for s in secs:
        if len(s["tx"]) > max_len:
            s["tx"] = s["tx"][:max_len] + ("…" if not full else "")
    doc["sections"] = secs
    doc["section_count"] = len(secs)
    con.close()
    return json.dumps(doc, ensure_ascii=False)

def _list_datasets() -> str:
    """데이터셋 목록 (현재 80종) 본체"""
    con = db()
    rows = [dict(r) for r in con.execute("""
        SELECT sn, ds_name,
               SUM(CASE WHEN dtype='설명서' THEN 1 ELSE 0 END) AS manuals,
               SUM(CASE WHEN dtype='가이드' THEN 1 ELSE 0 END) AS guides,
               SUM(nsec) AS sections, SUM(nbytes) AS bytes,
               GROUP_CONCAT(id) AS doc_ids
        FROM docs GROUP BY sn ORDER BY sn DESC""")]
    con.close()
    return json.dumps({"dataset_count": len(rows), "datasets": rows}, ensure_ascii=False)

def _recommend_datasets(requirement: str, limit: int = 5) -> str:
    """요구사항 기반 데이터셋 추천 본체 — 그룹별 개별 쿼리 후 파이썬에서 점수 합산.
    점수 = Σ (그룹별 히트 섹션 수 × 가중치): FTS 그룹 2.0, LIKE 그룹 1.0.
    모든 그룹(토큰)에 히트가 있는 데이터셋이 상위 — AND 요구사항을 소프트하게 반영."""
    con = db()
    fts_groups, like_groups = expand_query(retrieval_v2.normalize_query(requirement))
    limit = max(1, min(int(limit), 10))
    if not fts_groups and not like_groups:
        con.close()
        return json.dumps({"error": "요구사항이 비어 있거나 특수문자뿐입니다"}, ensure_ascii=False)

    scores = {}   # sn -> {"score", "hits", "labels"}
    def add_hits(rows, weight):
        for r in rows:
            sn = r["sn"]
            e = scores.setdefault(sn, {"score": 0.0, "hits": 0, "labels": set()})
            e["score"] += weight
            e["hits"] += r["hits"]
            for lab in (r["labels"] or "").split(" | "):
                if lab and len(e["labels"]) < 8:
                    e["labels"].add(lab)

    for g in fts_groups:
        rows = con.execute("""
            SELECT d.sn, COUNT(*) AS hits, GROUP_CONCAT(s.label, ' | ') AS labels
            FROM sec_fts JOIN sections s ON s.rowid = sec_fts.rowid
            JOIN docs d ON d.id = s.doc_id
            WHERE sec_fts MATCH ? GROUP BY d.sn""", (g,)).fetchall()
        add_hits(rows, 2.0)
    for group in like_groups:
        cond = " OR ".join("s.tx LIKE ? ESCAPE '\\'" for _ in group)
        rows = con.execute(f"""
            SELECT d.sn, COUNT(*) AS hits, GROUP_CONCAT(s.label, ' | ') AS labels
            FROM sections s JOIN docs d ON d.id = s.doc_id
            WHERE {cond} GROUP BY d.sn""",
            ["%" + t + "%" for t in group]).fetchall()
        add_hits(rows, 1.0)

    n_groups = len(fts_groups) + len(like_groups)
    name_map = {r["sn"]: r["ds_name"] for r in con.execute(
        """SELECT sn, COALESCE(MAX(CASE WHEN dtype='설명서' THEN ds_name END), MAX(ds_name)) AS ds_name
           FROM docs GROUP BY sn""")}
    ranked = []
    for sn, e in scores.items():
        # 전체 그룹 중 몇 개에 걸쳤는지 비중(coverage)을 반영해 정렬 안정화
        ranked.append({"sn": sn, "ds_name": name_map.get(sn, ""),
                       "score": round(e["score"], 2),
                       "hit_sections": e["hits"], "evidence_labels": sorted(e["labels"])})
    ranked.sort(key=lambda x: -x["score"])
    ranked = ranked[:limit]
    con.close()
    return json.dumps({
        "requirement": requirement, "shown": len(ranked),
        "note": "score는 토큰별 히트 가중합(높을수록 관련). evidence_labels는 근거 섹션 라벨",
        "results": ranked}, ensure_ascii=False)

def _compare_datasets(sn_a: str, sn_b: str) -> str:
    """두 데이터셋 비교 본체 — 구성·문서·규모·주요 섹션 라벨 비교"""
    con = db()
    out = {}
    for sn in (sn_a, sn_b):
        r = con.execute("""
            SELECT sn, ds_name,
                   SUM(CASE WHEN dtype='설명서' THEN 1 ELSE 0 END) AS manuals,
                   SUM(CASE WHEN dtype='가이드' THEN 1 ELSE 0 END) AS guides,
                   SUM(nsec) AS sections, SUM(nbytes) AS bytes,
                   GROUP_CONCAT(id) AS doc_ids
            FROM docs WHERE sn = ? GROUP BY sn""", (sn,)).fetchone()
        if not r:
            con.close()
            return json.dumps({"error": f"데이터셋 없음: {sn}", "hint": "list_datasets 으로 sn 확인"}, ensure_ascii=False)
        info = dict(r)
        # 주요 섹션 라벨 (문서 앞부분 위주 12개 — 구성 파악용)
        labels = [row[0] for row in con.execute("""
            SELECT DISTINCT s.label FROM sections s JOIN docs d ON d.id = s.doc_id
            WHERE d.sn = ? AND s.label != '(서두)' AND s.level <= 3
            ORDER BY s.sec_idx LIMIT 12""", (sn,))]
        info["section_labels"] = labels
        out[sn] = info
    con.close()
    return json.dumps({"compare": [out[sn_a], out[sn_b]]}, ensure_ascii=False)

def _get_stats() -> str:
    """코퍼스 통계 본체"""
    con = db()
    stats = {
        "total_docs": con.execute("SELECT COUNT(*) FROM docs").fetchone()[0],
        "total_sections": con.execute("SELECT COUNT(*) FROM sections").fetchone()[0],
        "datasets": con.execute("SELECT COUNT(DISTINCT sn) FROM docs").fetchone()[0],
        "by_dtype": [dict(r) for r in con.execute(
            "SELECT dtype, COUNT(*) AS cnt FROM docs GROUP BY dtype")],
        "db_bytes": os.path.getsize(DB_PATH),
    }
    con.close()
    stats["vector_index"] = vector_runtime.stats(_engine_context())
    return json.dumps(stats, ensure_ascii=False)

# ---- 청크 벡터 검색: 원문/모델 검증 및 메모리 캐시는 vector_runtime 담당 ----
def _engine_context():
    return SimpleNamespace(DB_PATH=DB_PATH, VEC_DB_PATH=VEC_DB_PATH, VEC_MODEL=VEC_MODEL,
                           db=db, _semantic_search=_semantic_search)


def _semantic_search(query: str, dtype: str = "", sn: str = "", limit: int = 10) -> str:
    """청크 벡터 검색: 원문 전체 범위와 모델/원문 버전 검증."""
    return vector_runtime.semantic_search(_engine_context(), query, dtype, sn, limit)

def _hybrid_search(query: str, dtype: str = "", sn: str = "", limit: int = 10) -> str:
    """자연어 어휘 후보 + 청크 벡터, 오류 시 경고를 포함한 어휘 폴백."""
    return retrieval_v2.hybrid_search(_engine_context(), query, dtype, sn, limit)

# ---- MCP 도구 등록 ----
@mcp.tool()
def search_docs(query: str, dtype: str = "", sn: str = "", limit: int = 10) -> str:
    """문서 단위 검색 — AI Hub 기술문서 (규모는 get_stats 참고)(설명서·활용가이드)에서 제목·본문을 통합 검색한다.
    query: 검색어(예: SMT 공정, 배터리 열폭주, 용접), dtype: '설명서'|'가이드' 필터,
    sn: 데이터셋 번호 필터(예: 71900), limit: 최대 건수(기본 10, 최대 30)"""
    return _search_docs(query, dtype, sn, limit)

@mcp.tool()
def search_sections(query: str, dtype: str = "", sn: str = "", limit: int = 10) -> str:
    """섹션 단위 검색 — 코퍼스 섹션에서 문구를 찾는다. 데이터 구성·라벨링 기준·
    품질 검수 절차 등 실무 내용을 원문 그대로 수집할 때 사용.
    query: 검색어, dtype: '설명서'|'가이드' 필터, sn: 데이터셋 번호 필터, limit: 최대 건수"""
    return _search_sections(query, dtype, sn, limit)

@mcp.tool()
def get_document(doc_id: str, full: bool = False) -> str:
    """문서 1건 전체 조회 — doc_id(예: '71900-설명서')로 섹션 전체를 반환.
    full=false(기본)면 섹션 본문 200자 요약, full=true면 본문 전체"""
    return _get_document(doc_id, full)

@mcp.tool()
def list_datasets() -> str:
    """데이터셋 목록 — 데이터셋별 문서 수·섹션 수·doc_ids. sn 필터나 doc_id 확인에 사용"""
    return _list_datasets()

@mcp.tool()
def recommend_datasets(requirement: str, limit: int = 5) -> str:
    """요구사항 기반 데이터셋 추천 — 요구 문장(예: 'OHT 화재 예측용 센서 데이터')을 받아
    관련도 높은 데이터셋과 근거 섹션 라벨을 반환. 제안서·기획서 작성 시 후보 데이터셋 선정에 사용.
    requirement: 요구사항 문장, limit: 최대 추천 수(기본 5, 최대 10)"""
    return _recommend_datasets(requirement, limit)

@mcp.tool()
def compare_datasets(sn_a: str, sn_b: str) -> str:
    """두 데이터셋 비교 — 구성(설명서/가이드 수)·규모(섹션·바이트)·주요 섹션 라벨을 나란히 반환.
    sn_a·sn_b: 데이터셋 번호(예: '71802', '71918')"""
    return _compare_datasets(sn_a, sn_b)

@mcp.tool()
def semantic_search(query: str, dtype: str = "", sn: str = "", limit: int = 10) -> str:
    """의미(벡터) 검색 — 임베딩 유사도로 어휘가 달라도 의미가 같은 섹션을 찾는다.
    최초 호출 시 임베딩 모델을 로드해 수 초가 걸릴 수 있다.
    query: 검색 문장(예: '설비가 뜨거워지는 문제'), dtype: '설명서'|'가이드',
    sn: 데이터셋 번호 필터, limit: 최대 건수(기본 10, 최대 30)"""
    return _semantic_search(query, dtype, sn, limit)

@mcp.tool()
def hybrid_search(query: str, dtype: str = "", sn: str = "", limit: int = 10) -> str:
    """하이브리드 검색 — 어휘(FTS+동의어) 검색과 의미(벡터) 검색을 RRF로 결합해 반환.
    정확한 용어 매치와 의미 유사 결과를 모두 잡을 때 사용.
    query: 검색어·문장, dtype: '설명서'|'가이드', sn: 데이터셋 번호 필터,
    limit: 최대 건수(기본 10, 최대 30)"""
    return _hybrid_search(query, dtype, sn, limit)

@mcp.tool()
def get_stats() -> str:
    """코퍼스 통계 — 총 문서·섹션 수, 문서 유형별 분포"""
    return _get_stats()

if __name__ == "__main__":
    mcp.run()
