"""원문 청크 벡터 검색. 기존 FTS 원문은 변경하지 않는다."""
import hashlib
import json
import os
import sqlite3
import threading
from pathlib import Path

_lock = threading.RLock()
_state = {}


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def _load(engine):
    import numpy as np
    path = engine.VEC_DB_PATH
    if not os.path.isfile(path):
        raise FileNotFoundError('청크 벡터 인덱스가 준비되지 않았습니다')
    signature = (path, os.stat(path).st_mtime_ns, os.stat(path).st_size,
                 engine.DB_PATH, os.stat(engine.DB_PATH).st_mtime_ns)
    with _lock:
        if _state.get('signature') == signature:
            return _state
        con = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        con.row_factory = sqlite3.Row
        try:
            if con.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                raise ValueError('벡터 DB 무결성 오류')
            meta = dict(con.execute('SELECT key,value FROM meta').fetchall())
            if meta.get('source_sha256') != _sha(engine.DB_PATH):
                raise ValueError('원문 인덱스와 벡터 인덱스 버전이 다릅니다')
            if meta.get('model') != engine.VEC_MODEL or not meta.get('revision'):
                raise ValueError('임베딩 모델 또는 revision 메타가 일치하지 않습니다')
            rows = [dict(r) for r in con.execute('SELECT * FROM chunks ORDER BY chunk_id')]
        finally:
            con.close()
        if not rows:
            raise ValueError('청크 인덱스가 비어 있습니다')
        dim = int(meta['dim'])
        mat = np.vstack([np.frombuffer(r.pop('embedding'), dtype=np.float32) for r in rows])
        if mat.shape != (len(rows), dim) or not np.isfinite(mat).all():
            raise ValueError('벡터 차원 또는 유한성 검사 실패')
        if not np.allclose(np.linalg.norm(mat, axis=1), 1.0, atol=1e-4):
            raise ValueError('벡터 정규화 검사 실패')
        # 캐시에 있는 동일 revision만 사용: 최초 질의에도 원문 외부 전송 없음.
        from build_chunk_index import load_model, MODEL, REVISION
        if meta['model'] != MODEL or meta['revision'] != REVISION:
            raise ValueError('빌더와 런타임의 모델 revision이 다릅니다')
        requested_device = os.getenv('MFG_VECTOR_DEVICE', 'auto')
        if requested_device == 'auto':
            import torch
            requested_device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        if requested_device not in ('cpu', 'mps'):
            raise ValueError('MFG_VECTOR_DEVICE는 auto, cpu 또는 mps여야 합니다')
        model, local_model_path = load_model(device=requested_device)
        model_dim = (model.get_embedding_dimension()
                     if hasattr(model, 'get_embedding_dimension')
                     else model.get_sentence_embedding_dimension())
        if model_dim != dim:
            raise ValueError('모델 출력 차원과 인덱스 차원이 다릅니다')
        with engine.db() as src:
            docs = {r['id']: dict(r) for r in src.execute(
                'SELECT id,sn,title,dtype,ds_name FROM docs')}
        if any(r['doc_id'] not in docs for r in rows):
            raise ValueError('원문에 없는 청크 문서가 존재합니다')
        _state.clear()
        _state.update(signature=signature, meta=meta, rows=rows, mat=mat,
                      model=model, docs=docs)
        return _state


def semantic_search(engine, query, dtype='', sn='', limit=10):
    limit = max(1, min(int(limit), 30))
    if not str(query).strip() or not any(c.isalnum() for c in str(query)):
        return json.dumps({'error': '검색어가 비어 있거나 특수문자뿐입니다',
                           'shown': 0, 'results': []}, ensure_ascii=False)
    if dtype and dtype not in ('설명서', '가이드'):
        return json.dumps({'error': '문서 유형은 설명서 또는 가이드입니다',
                           'shown': 0, 'results': []}, ensure_ascii=False)
    try:
        import numpy as np
        state = _load(engine)
        from build_chunk_index import clean_text
        cleaned_query = clean_text(query)
        tokens = state['model'].tokenizer(cleaned_query, truncation=False)['input_ids']
        if len(tokens) > state['model'].max_seq_length:
            raise ValueError('질의가 토큰 제한을 넘었습니다. 검색 질문을 줄여 주세요')
        with _lock:
            q = state['model'].encode([cleaned_query], normalize_embeddings=True,
                                      show_progress_bar=False)[0].astype(np.float32)
        if q.shape != (state['mat'].shape[1],) or not np.isfinite(q).all():
            raise ValueError('질의 벡터 차원 또는 유한성 검사 실패')
        sims = state['mat'] @ q
        order = np.argsort(-sims, kind='stable')
        results, seen = [], set()
        for i in order:
            row = state['rows'][int(i)]
            doc = state['docs'][row['doc_id']]
            if dtype and doc['dtype'] != dtype:
                continue
            if sn and str(doc['sn']) != str(sn):
                continue
            # 인접 겹침 청크 중복 억제 (동일 원문 섹션 내 중첩 50% 이상).
            span = (row['doc_id'], row['sec_idx'], row['start_char'], row['end_char'])
            duplicate = any(a == span[0] and b == span[1] and
                max(0, min(d, span[3])-max(c, span[2])) /
                max(1, min(d-c, span[3]-span[2])) > .5 for a,b,c,d in seen)
            if duplicate:
                continue
            seen.add(span)
            results.append({**doc, 'doc_id': row['doc_id'],
                            'sec_idx': row['sec_idx'], 'chunk_id': row['chunk_id'],
                            'label': row['label'], 'start_char': row['start_char'],
                            'end_char': row['end_char'], 'snip': row['raw_text'][:600],
                            'score': round(float(sims[int(i)]), 6),
                            'url': 'https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn='+str(doc['sn'])})
            if len(results) >= limit:
                break
        return json.dumps({'query': query, 'mode': 'semantic_chunk', 'status': 'ok',
                           'shown': len(results), 'results': results,
                           'note': '유사도는 진단 정확도나 적합성 확률이 아닙니다. 원문을 확인하세요.'}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({'query': query, 'mode': 'semantic_chunk', 'status': 'unavailable',
                           'error': str(exc), 'shown': 0, 'results': []}, ensure_ascii=False)


def stats(engine):
    if not os.path.isfile(engine.VEC_DB_PATH):
        return {'status': 'unavailable', 'reason': '청크 벡터 인덱스 없음'}
    try:
        con = sqlite3.connect(f'file:{engine.VEC_DB_PATH}?mode=ro', uri=True)
        try:
            meta = dict(con.execute('SELECT key,value FROM meta').fetchall())
            count = con.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
        finally:
            con.close()
        return {'status': 'built', 'chunks': count, 'model': meta['model'],
                'dim': int(meta['dim']), 'revision': meta.get('revision'),
                'bytes': os.path.getsize(engine.VEC_DB_PATH),
                'source_verified': meta.get('source_sha256') == _sha(engine.DB_PATH),
                'note': '빌드 상태이며 현재 앱의 신규 도구 연결 상태와는 별개입니다'}
    except Exception as exc:
        return {'status': 'unavailable', 'reason': str(exc)}
