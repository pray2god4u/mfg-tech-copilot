import hashlib,json,sqlite3,sys,os
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'rag'))
from build_chunk_index import MODEL,REVISION

def prepare():
    source=BASE/'rag'/'mfg_corpus_20261026_expanded_2.db'
    vector=BASE/'rag'/'mfg_chunks_20261028_v2.db'
    if not source.is_file() or not vector.is_file():
        raise RuntimeError('원문과 사전 구축된 청크 인덱스가 배포 파일에 필요합니다')
    with sqlite3.connect('file:'+str(vector)+'?mode=ro',uri=True) as c:
        meta=dict(c.execute('SELECT key,value FROM meta'))
        count=c.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
        assert c.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    assert meta['model']==MODEL and meta['revision']==REVISION
    assert hashlib.sha256(source.read_bytes()).hexdigest()==meta['source_sha256']
    assert count==int(meta['chunks']) and int(meta['dim'])==1024
    print('INDEX_VERIFIED',count,flush=True)
    if '--verify-only' in sys.argv:return
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=MODEL,revision=REVISION,allow_patterns=[
        '*.json','*.txt','*.model','*.safetensors','1_Pooling/*','2_Normalize/*'])
    from build_chunk_index import load_model
    model,path=load_model(device='cpu')
    print('MODEL_READY',str(path),flush=True)

if __name__=='__main__':prepare()
