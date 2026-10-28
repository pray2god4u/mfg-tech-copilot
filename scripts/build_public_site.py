import argparse,sys,json,sqlite3,re,shutil,hashlib
from pathlib import Path
from urllib.parse import urlsplit
import importlib.util
from flask import render_template
BASE=Path(__file__).resolve().parents[1]

def rewrite(text,prefix):
    def link(m):
        kind,url,quote=m.group(1),m.group(2),m.group(3)
        if not url.startswith('/') or url.startswith('//'):return m.group(0)
        path,sep,fragment=url.partition('#')
        path,query_sep,query=path.partition('?')
        target=None
        if path=='/':target='index.html'
        elif path in ('/search','/datasets','/demo'):target=path[1:]+'.html'
        elif path.startswith('/static/'):target=path[1:]
        elif path.startswith('/dataset/') or path.startswith('/document/'):target=path[1:]+'.html'
        if target is None:return m.group(0)
        return kind+'="'+prefix+target+(('?'+query) if query_sep else '')+(('#'+fragment) if sep else '')+'"'
    return re.sub(r'(href|src|action)=[\"\']([^\"\']+)([\"\'])',link,text)

def build(api_base=''):
    if api_base:
        u=urlsplit(api_base)
        if u.scheme!='https' or u.hostname in ('localhost','127.0.0.1') or not u.hostname or u.username:
            raise ValueError('공개 HTTPS 검색 서버 주소가 필요합니다')
    out=BASE/'docs';out.mkdir(exist_ok=True)
    spec=importlib.util.spec_from_file_location('public_site_builder',BASE/'webapp'/'app.py')
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    app=mod.app;client=app.test_client()
    with sqlite3.connect('file:'+str(Path(mod.engine.DB_PATH))+'?mode=ro',uri=True) as c:
        c.row_factory=sqlite3.Row
        docs=[dict(r) for r in c.execute('SELECT * FROM docs ORDER BY id')]
        sns=sorted({d['sn'] for d in docs})
    pages=[('/', 'index.html'),('/search','search.html'),('/datasets','datasets.html'),('/demo','demo.html')]
    pages += [('/dataset/'+str(sn),'dataset/'+str(sn)+'.html') for sn in sns]
    for route,name in pages:
        r=client.get(route);assert r.status_code==200,(route,r.status_code)
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True)
        prefix='../' if '/' in name else './'
        target.write_text(rewrite(r.get_data(as_text=True),prefix),encoding='utf-8')
    with sqlite3.connect('file:'+mod.engine.VEC_DB_PATH+'?mode=ro',uri=True) as v:
        v.row_factory=sqlite3.Row
        for d in docs:
            with mod.engine.db() as src:
                sections=[dict(r) for r in src.execute('SELECT sec_idx,label,tx FROM sections WHERE doc_id=? ORDER BY sec_idx',(d['id'],))]
            chunks=[dict(r) for r in v.execute('SELECT chunk_id,label,sec_idx,start_char,end_char,raw_text FROM chunks WHERE doc_id=? ORDER BY chunk_id',(d['id'],))]
            groups={}
            for s in sections:s['overview']=(s.pop('tx') or '')[:200]
            for c in chunks:
                c['excerpt']=c.pop('raw_text')[:200];groups.setdefault(c['sec_idx'],[]).append(c)
            with app.test_request_context('/'):
                page=render_template('public_document.html',doc=d,sections=sections,chunks=chunks,chunks_by_section=groups)
            target=out/'document'/(d['id']+'.html');target.parent.mkdir(exist_ok=True)
            target.write_text(rewrite(page,'../'),encoding='utf-8')
    for source in (BASE/'webapp'/'static').rglob('*'):
        if source.is_file():
            relative=source.relative_to(BASE/'webapp'/'static')
            if relative.parts[0]=='browser-index':continue # local full-text experimental assets MUST NOT be published
            target=out/'static'/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
    if api_base:raise ValueError('This public build uses free browser search, not a paid search backend')
    config={'api_base':'','deployment_pending':False,'browser_search':True,'index_path':'static/browser-index-public-v1/','model':'Xenova/multilingual-e5-small','model_dtype':'q8','query_sent_to_search_server':False}
    assert not (out/'static/browser-index').exists(), 'Private raw browser index present in public output'
    public_chunks=json.loads((out/'static/browser-index-public-v1/chunks.json').read_text())
    assert all(len(c['t'])<=200 for c in public_chunks)
    (out/'static'/'search-config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
    (out/'.nojekyll').write_text('')
    assert len(list((out/'document').glob('*.html')))==len(docs)
    for f in out.rglob('*.html'):
        for url in re.findall(r'(?:href|src|action)=[\"\']([^\"\']+)',f.read_text()):
            if url.startswith(('http:','https:','mailto:','data:','#','//')):continue
            path=url.split('#')[0].split('?')[0]
            if path:assert (f.parent/path).exists(),(str(f),url)
        assert 'localhost:' not in f.read_text() and 'file:///' not in f.read_text()
    print(json.dumps({'pages':len(list(out.rglob('*.html'))),'documents':len(docs),'config':config,'source_verified':True,'output':str(out)},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--api-base',default='');a=p.parse_args();build(a.api_base)
