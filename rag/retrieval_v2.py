"""Independent, read-only section retrieval; no model loading or index mutation.

Integration: lexical_search(server_module, query, ...), hybrid_search(server_module,
query, ...). The engine supplies DB_PATH and _semantic_search. Returned snippets
and labels are slices of the corpus originals, never reconstructed summaries.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
from contextlib import closing
from pathlib import Path
from urllib.parse import quote

__all__ = ["normalize_query", "lexical_search", "hybrid_search", "clear_cache"]

# Small, explicit lemmas and paraphrases, not a learned diagnosis dictionary.
_PHRASES = (
    (r"뜨거워(?:지는|진|졌다|져요|지고|지다)?", "과열"),
    (r"온도(?:가)?\s*(?:올라(?:가는|간|요)|상승(?:하는)?)", "과열"),
    (r"멈추(?:는|다|고|면)|멈췄(?:다|어요)?|멈춤", "정지 고장"),
    (r"전류(?:가)?\s*(?:튀(?:는|다)|급증(?:하는)?)", "전류 이상"),
    (r"떨리(?:는|다)|떨림", "진동"),
    (r"라인(?=이|의|에서|\s|$)", "공정"),
    (r"예측용(?=\s|$)", "예측"),
)
_STOP = frozenset("데이터 데이터셋 자료 구축 구성 활용 관련 위한 용도 제공 수집 구축된 대한 있는 있음 필요 가능 사용 문제 원인 이유 알려줘 알려주세요 찾아줘 찾아주세요 보여줘 무엇 뭐 왜 어떤 어떻게 좀 해줘 주세요 알고 싶어 싶어요 발생 발생하는 하는 되는 있는지 알려 줄 설명 검색 및 와 과 의 에서 으로 에 대한".split())
_SUFFIXES = ("에서는", "으로는", "에서", "에게", "으로", "까지", "부터", "하는", "되는", "인가요", "인지", "나요", "이랑", "이", "가", "을", "를", "은", "는", "의", "와", "과", "에", "도")
_SYNONYMS = {
    "설비": ("기계", "장비"), "공정": ("라인", "제조공정"),
    "과열": ("발열", "온도상승", "열폭주", "overheat"),
    "정지": ("중단", "멈춤", "고장", "fault"),
    "고장": ("결함", "이상", "고장진단", "fault"),
    "화재": ("열폭주", "발화", "탄화", "연기", "fire"),
    "열폭주": ("발화", "화재", "과열"),
    "배터리": ("이차전지", "리튬이온", "battery"),
    "이차전지": ("배터리", "리튬이온"),
    "oht": ("이송장치", "호이스트", "overhead hoist"),
    "이송장치": ("oht", "agv", "반송장치", "운반장치"),
    "agv": ("무인운반", "무인이송", "agv"),
    "센서": ("계측", "측정", "sensor", "sensor_data"),
    "전류": ("current", "전류값"), "전압": ("voltage",),
    "온도": ("열화상", "발열", "temperature"),
    "진동": ("가속도", "소음", "vibration"),
    "예지보전": ("예측정비", "예방정비", "phm", "상태감시"),
    "예측": ("예지", "추정", "prediction"),
    "수명": ("잔존수명", "rul", "soh"),
    "불량": ("결함", "스크래치"), "라벨링": ("라벨", "태깅", "주석"),
    "라벨": ("라벨링", "태깅", "클래스"),
    "이미지": ("영상", "사진", "카메라"), "영상": ("이미지", "동영상"),
    "검수": ("품질검증", "교차검증"), "용접": ("비드", "용접로봇"),
    "smt": ("실장", "표면실장"), "p&id": ("배관", "계장", "도면"),
}
_LEMMAS = set(_SYNONYMS) | _STOP | {s for v in _SYNONYMS.values() for s in v}
_ASSETS = frozenset(("oht", "이송장치", "agv", "배터리", "이차전지", "smt", "용접"))
_CACHE = {}
_LOCK = threading.RLock()
_NOTE = "검색 결과는 문서 후보이며 실제 설비의 고장 원인·진단을 입증하지 않습니다. 원문과 현장 측정값을 확인하세요."


def normalize_query(query: str) -> str:
    """Normalize a few grounded Korean paraphrases and particles; keep unknowns.

    Single-character terms are deliberately ignored (열 must not match 열화,
    열거, 배열, etc.). No caller query or source text is modified in place.
    """
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    text = query.strip()
    for pattern, replacement in _PHRASES:
        text = re.sub(pattern, replacement, text)
    tokens = re.findall(r"P&ID|[\w]+", text, flags=re.IGNORECASE)
    out = []
    for token in tokens:
        key = token.lower()
        if key not in _LEMMAS:
            for suffix in _SUFFIXES:
                if key.endswith(suffix) and len(key) - len(suffix) >= 2:
                    stem = key[:-len(suffix)]
                    # A known lemma is safe; unknowns strip only unambiguous particles.
                    if stem in _LEMMAS or suffix in ("에서", "으로", "을", "를", "은", "는", "의"):
                        key = stem
                        break
        if key in _STOP or len(key) < 2:
            continue
        display = key.upper() if key in ("oht", "agv", "smt", "phm", "rul", "soh", "p&id") else key
        if display not in out:
            out.append(display)
    return " ".join(out)


def clear_cache():
    """Drop only in-memory snapshots (useful after changing engine.DB_PATH)."""
    with _LOCK:
        _CACHE.clear()


def _corpus(engine, include_keys=False):
    path = Path(engine.DB_PATH).expanduser().resolve(strict=True)
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key][1 if include_keys else 0]
        # Even a stale/missing path cannot silently create a database.
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as con:
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA query_only=ON")
            rows = tuple(dict(r) for r in con.execute("""
                SELECT s.doc_id,s.sec_idx,s.level,s.label,s.tx,
                       d.title,d.dtype,d.ds_name,d.sn
                FROM sections s JOIN docs d ON d.id=s.doc_id
                ORDER BY s.doc_id,s.sec_idx
            """))
        valid = []
        seen = set()
        for r in rows:
            tx, label = r["tx"] or "", r["label"] or ""
            compact = lambda s: re.sub(r"[\W_]+", "", s).lower()
            if not tx.strip() or (compact(tx) and compact(tx) == compact(label)):
                continue
            ident = (r["doc_id"], label, tx)
            if ident in seen:
                continue
            seen.add(ident)
            r["sn"] = str(r["sn"])
            r["_body"] = tx.lower()
            r["_context"] = (r["title"] + " " + r["ds_name"]).lower()
            r["_label"] = label.lower()
            valid.append(r)
        # Bound cache to the active corpus; no externally shared mutable rows.
        _CACHE.clear()
        _CACHE[key] = (tuple(valid), frozenset((r["doc_id"], r["sec_idx"]) for r in rows))
        return _CACHE[key][1 if include_keys else 0]


def _contains(text, term):
    if len(term) < 2:
        return False
    if re.search(r"[a-z0-9]", term):
        return bool(re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", text))
    return term in text


def _groups(normalized):
    return [(term, tuple(dict.fromkeys((term,) + _SYNONYMS.get(term, ()))))
            for term in normalized.lower().split()]


def _hit(text, group):
    return max((1.0 if i == 0 else .75 for i, t in enumerate(group) if _contains(text, t)), default=0.0)


def _asset_ok(row, groups):
    context = row["_context"] + " " + row["_body"]
    return all(any(_contains(context, t) for t in alternatives)
               for term, alternatives in groups if term in _ASSETS)


def _snip(row, groups, width=700):
    """A contiguous *original* tx slice, including JSON keys/labels unchanged."""
    body = row["_body"]
    # Pick an evidence-dense window, not simply the first asset mention in a
    # very long introduction. The selected window is still an exact raw slice.
    starts = {0}
    for _, alts in groups:
        for term in alts:
            if not _contains(body, term):
                continue
            for i, match in enumerate(re.finditer(re.escape(term), body)):
                starts.add(max(0, match.start() - 100))
                if i >= 63:
                    break
    def window_score(start):
        window = body[start:start + width]
        return sum((1.0 + (.5 if term not in _ASSETS and term not in ("설비", "공정") else 0)) * _hit(window, alts)
                   for term, alts in groups)
    start = min(starts, key=lambda s: (-window_score(s), s))
    return row["tx"][start:start + width]


def _public(row, groups):
    out = {k: row[k] for k in ("doc_id", "sec_idx", "title", "dtype", "ds_name", "sn", "label", "level")}
    out["snip"] = _snip(row, groups)
    out["source_url"] = "https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=" + quote(row["sn"])
    out["source"] = out["source_url"]
    return out


def _prepare(engine, query, dtype, sn, limit):
    if not isinstance(dtype, str) or dtype not in ("", "설명서", "가이드"):
        raise ValueError("dtype must be '', '설명서', or '가이드'")
    if isinstance(sn, bool) or not isinstance(sn, (str, int)):
        raise ValueError("sn must be an empty string or a numeric dataset id")
    sn = str(sn).strip()
    if sn and not re.fullmatch(r"\d+", sn):
        raise ValueError("sn must contain only digits")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    limit = min(limit, 30)
    normalized = normalize_query(query)
    groups = _groups(normalized)
    rows = _corpus(engine) if groups else ()
    rows = tuple(r for r in rows if (not dtype or r["dtype"] == dtype) and (not sn or r["sn"] == sn))
    return normalized, groups, rows, sn, limit


def _rank(rows, groups):
    scored = []
    for row in rows:
        if not _asset_ok(row, groups):
            continue
        body = [_hit(row["_body"], a) for _, a in groups]
        title = [_hit(row["_context"], a) for _, a in groups]
        label = [_hit(row["_label"], a) for _, a in groups]
        # Context cannot make an empty/unrelated section evidence by itself.
        if not any(body):
            continue
        coverage = sum(bool(b or t or l) for b, t, l in zip(body, title, label)) / len(groups)
        score = 6 * coverage ** 2 + sum(body) / len(groups) + .65 * sum(title) / len(groups) + .4 * sum(label) / len(groups)
        # Topic terms in actual text are stronger evidence than an asset title.
        topics = [b for (term, _), b in zip(groups, body) if term not in _ASSETS and term not in ("설비", "공정")]
        if topics:
            score += 2 * sum(bool(b) for b in topics) / len(topics)
        scored.append((score, row, coverage))
    scored.sort(key=lambda x: (-x[0], x[1]["doc_id"], x[1]["sec_idx"]))
    return scored


def _unsupported_cause(query, rows, groups):
    if not re.search(r"원인|이유|왜", query):
        return False
    # A soft match on '설비' must not claim evidence about an unknown cause.
    return any(not any(_hit(r["_body"] + " " + r["_context"], alts) for r in rows)
               for _, alts in groups)


def _payload(query, normalized, mode, results, status="ok", warnings=None):
    return {"query": query, "normalized_query": normalized, "mode": mode,
            "status": status, "shown": len(results), "results": results,
            "evidence_status": "candidate_only" if results else "no_evidence",
            "diagnosis_supported": False, "note": _NOTE if results else "해당 조건에서 뒷받침할 문서 근거를 찾지 못했습니다. 검색 실패는 원인의 부재를 입증하지 않습니다.",
            "warnings": warnings or []}


def _dump(data):
    return json.dumps(data, ensure_ascii=False, allow_nan=False)


def _error(query, mode, exc):
    out = _payload(query, "", mode, [], "error", [f"{type(exc).__name__}: {exc}"])
    out["error"] = f"{type(exc).__name__}: {exc}"
    return _dump(out)


def lexical_search(engine, query, dtype="", sn="", limit=10):
    """Cached section scan with soft synonym coverage and strict metadata filters."""
    try:
        normalized, groups, rows, sn, limit = _prepare(engine, query, dtype, sn, limit)
        if not groups or _unsupported_cause(query, rows, groups):
            return _dump(_payload(query, normalized, "lexical_v2", [], "no_evidence"))
        ranked = _rank(rows, groups)[:limit]
        results = []
        for score, row, coverage in ranked:
            result = _public(row, groups)
            result.update(score=round(score, 6), coverage=round(coverage, 4), matched="lex")
            results.append(result)
        return _dump(_payload(query, normalized, "lexical_v2", results, "ok" if results else "no_evidence"))
    except Exception as exc:
        return _error(query, "lexical_v2", exc)


def hybrid_search(engine, query, dtype="", sn="", limit=10):
    """RRF(k=60), lexical=1 and vector=.4, >=30 candidates per branch.

    Semantic errors, malformed responses, nonfinite scores, and stale section
    IDs explicitly degrade to lexical results. Canonical corpus metadata always
    wins over vector metadata; semantic results cannot bypass filters/assets.
    This function never imports/downloads an embedding model itself.
    """
    try:
        normalized, groups, rows, sn, limit = _prepare(engine, query, dtype, sn, limit)
        if not groups or _unsupported_cause(query, rows, groups):
            return _dump(_payload(query, normalized, "hybrid_v2", [], "no_evidence"))
        candidates = max(30, limit)
        lex = _rank(rows, groups)[:candidates]
        lexical_results = []
        for score, row, coverage in lex:
            result = _public(row, groups)
            result.update(score=round(score, 6), coverage=round(coverage, 4), matched="lex")
            lexical_results.append(result)
        canonical = {(r["doc_id"], r["sec_idx"]): r for r in rows}
        try:
            response = engine._semantic_search(normalized, dtype=dtype, sn=sn, limit=candidates)
            sem = json.loads(response) if isinstance(response, str) else response
            if not isinstance(sem, dict) or sem.get("error") or sem.get("status") in ("error", "degraded"):
                raise ValueError(f"semantic search failed: {sem}")
            if not isinstance(sem.get("results"), list):
                raise ValueError("semantic response must contain a results list")
            if sem.get("warnings"):
                raise ValueError(f"semantic warnings: {sem['warnings']}")
            sem_rows, seen = [], set()
            for item in sem["results"]:
                if not isinstance(item, dict):
                    raise ValueError("malformed semantic result")
                doc_id, sec_idx = item.get("doc_id"), item.get("sec_idx")
                if not isinstance(doc_id, str) or isinstance(sec_idx, bool) or not isinstance(sec_idx, int):
                    raise ValueError("semantic result needs doc_id and integer sec_idx")
                if "score" in item and (isinstance(item["score"], bool) or not isinstance(item["score"], (int, float)) or not math.isfinite(item["score"])):
                    raise ValueError("nonfinite or invalid semantic score")
                key = (doc_id, sec_idx)
                row = canonical.get(key)
                if row is None:
                    # A known but out-of-scope/empty section is filtered, a stale
                    # key is an index error and must not be silently swallowed.
                    all_keys = _corpus(engine, include_keys=True)
                    if key not in all_keys:
                        raise ValueError(f"stale semantic section: {key}")
                    continue
                if key not in seen and _asset_ok(row, groups):
                    seen.add(key)
                    sem_rows.append(row)
        except Exception as exc:
            out = _payload(query, normalized, "hybrid_v2", lexical_results[:limit], "degraded",
                           [f"semantic unavailable: {type(exc).__name__}: {exc}"])
            out["fallback"] = "lexical_v2"
            out["candidate_limit"] = candidates
            return _dump(out)
        merged = {}
        for weight, branch, ranked in ((1.0, "lex", [x[1] for x in lex]), (.4, "sem", sem_rows)):
            for rank, row in enumerate(ranked, 1):
                key = (row["doc_id"], row["sec_idx"])
                entry = merged.setdefault(key, [0.0, row, []])
                entry[0] += weight / (60 + rank)
                entry[2].append(branch)
        ordered = sorted(merged.values(), key=lambda x: (-x[0], x[1]["doc_id"], x[1]["sec_idx"]))[:limit]
        results = []
        for score, row, matched in ordered:
            result = _public(row, groups)
            result.update(rrf_score=round(score, 8), matched="+".join(matched))
            results.append(result)
        out = _payload(query, normalized, "hybrid_v2", results, "ok" if results else "no_evidence")
        out.update(candidate_limit=candidates, rrf_k=60, lexical_weight=1.0, vector_weight=.4)
        return _dump(out)
    except Exception as exc:
        return _error(query, "hybrid_v2", exc)
