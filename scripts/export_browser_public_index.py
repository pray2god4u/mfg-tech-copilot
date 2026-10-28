"""Publish irreversible lexical postings + <=200-character evidence, preserving local full corpus.
Existing mean-pooled e5-small vectors are reused without modifying the desktop BGE index.
"""
import json, sqlite3, re, hashlib, math
from pathlib import Path
BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / 'webapp/static/browser-index'
OUT = BASE / 'webapp/static/browser-index-public-v1'
SYN = {'화재':'화재 발화 불티 연기','예측':'예지 진단 조기감지','이송장치':'oht 반송 로봇','센서':'sensor 계측','과열':'온도 상승 발열','열폭주':'발화 온도 급상승','라벨링':'라벨 태깅 검수','멈추는':'정지 다운 중단','멈추':'정지 다운 중단','뜨거워지는':'과열 발열 온도','뜨거워짐':'과열 발열 온도'}
STOP = '알려줘 추천해줘 비교해줘 데이터 사업의 후보를 조기 문제 원인 때문에 발생하는 그리고 그'.split()
def toks(text):
    out=[]
    for w in re.sub(r'[^가-힣a-z0-9]+',' ',text.lower()).split():
        out.append(w);out.extend(SYN.get(w,'').split())
    return set(w for w in out if len(w)>=2 and w not in STOP)
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    meta=json.loads((SOURCE/'meta.json').read_text())
    assert digest(SOURCE/'vectors.bin') == meta['vectors_sha256']
    db=sqlite3.connect('file:'+str(BASE/'rag/mfg_chunks_20261028_v2.db')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    rows=[dict(r) for r in db.execute('SELECT chunk_id,doc_id,sec_idx,label,start_char,end_char,raw_text FROM chunks ORDER BY chunk_id')];db.close()
    assert len(rows)==meta['count']
    old=json.loads((SOURCE/'chunks.json').read_text())
    assert [r['chunk_id'] for r in rows]==[c['i'] for c in old]
    db=sqlite3.connect('file:'+str(BASE/'rag/mfg_corpus_20261026_expanded_2.db')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    docs=[dict(r) for r in db.execute('SELECT id,sn,title,dtype,ds_name FROM docs ORDER BY id')];db.close()
    postings={};chunks=[]
    for i,r in enumerate(rows):
        raw=r['raw_text'] or '';low=raw.lower()
        for w in sorted(toks(raw)):
            postings.setdefault(w,[]).append([i,min(10,low.count(w))])
        chunks.append({'i':r['chunk_id'],'d':r['doc_id'],'s':r['sec_idx'],'l':str(r['label'] or '')[:200],'a':r['start_char'],'b':r['end_char'],'t':raw[:200]})
    assert all(len(c['t'])<=200 for c in chunks)
    (OUT/'vectors.bin').write_bytes((SOURCE/'vectors.bin').read_bytes())
    for name,value in [('docs.json',docs),('chunks.json',chunks),('lexical.json',{'synonyms':SYN,'stop':STOP,'postings':postings})]:
        (OUT/name).write_text(json.dumps(value,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    meta.update({'schema':1,'browser_model':'Xenova/multilingual-e5-small','browser_model_revision':'761b726dd34fb83930e26aab4e9ac3899aa1fa78','browser_dtype':'q8','max_length':512,'excerpt_limit':200,'lexical_algorithm':'expanded_token_count_log_coverage_v1','public_text_policy':'chunk_first_200_chars_only; unordered lexical postings, no full raw text','source':'AI 허브(aihub.or.kr)','assets':{n:{'bytes':(OUT/n).stat().st_size,'sha256':digest(OUT/n)} for n in ['vectors.bin','docs.json','chunks.json','lexical.json']}})
    (OUT/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'count':len(chunks),'dimension':meta['dim'],'max_excerpt_chars':max(len(c['t']) for c in chunks),'lexical_terms':len(postings),'assets':meta['assets'],'output':str(OUT)},ensure_ascii=False))
if __name__=='__main__':main()
