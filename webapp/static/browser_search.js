/* Free client-side retrieval. Full raw text stays local: public snippets <=200 characters.
 * e5-small passage vectors: attention-mask mean + L2, query prefix 'query: '.
 * RRF weights lexical=1, semantic=.35. Retrieval is not equipment diagnosis.
 */
const site = new URL('../', import.meta.url);
const asset = path => new URL(path, site).href;
const $ = id => document.getElementById(id);
const state = { config:null, meta:null, docs:[], docMap:new Map(), chunks:[], lex:null, vectors:null,
  corePromise:null, vectorPromise:null, modelPromise:null, pipe:null, generation:0, inference:Promise.resolve() };
const node = (tag,text,cls) => {const n=document.createElement(tag);if(text!==undefined)n.textContent=String(text);if(cls)n.className=cls;return n;};
function status(title,detail){$('serviceTitle').textContent=title;$('serviceStatus').textContent=detail;}
function capabilities(){status(state.pipe?'브라우저 의미 검색 준비 완료':'무료 브라우저 어휘 검색 준비 완료',state.pipe?'의미·하이브리드·어휘 검색을 이 기기에서 실행합니다. 검색 API 서버를 사용하지 않습니다.':'어휘·카탈로그는 즉시 사용 가능합니다. 의미·하이브리드를 선택하면 최초 약 135MB 모델과 10MB 벡터, 추가 런타임을 내려받습니다.');}
async function json(url){const r=await fetch(url);if(!r.ok)throw new Error('파일 다운로드 실패 (HTTP '+r.status+')');return r.json();}
async function checked(name){
  const info=state.meta.assets[name];if(!info)throw new Error('인덱스 검증 정보 없음');
  const r=await fetch(asset(state.config.index_path+name));if(!r.ok)throw new Error(name+' HTTP '+r.status);
  const bytes=await r.arrayBuffer();if(bytes.byteLength!==info.bytes)throw new Error(name+' 크기 불일치');
  if(!globalThis.crypto?.subtle)throw new Error('무결성 검증에 HTTPS 또는 localhost가 필요합니다');
  const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');
  if(hash!==info.sha256)throw new Error(name+' SHA-256 불일치');
  return name.endsWith('.json')?JSON.parse(new TextDecoder().decode(bytes)):bytes;
}
function loadCore(){
  if(state.corePromise)return state.corePromise;
  state.corePromise=(async()=>{
    state.config=await json(asset('static/search-config.json'));
    if(state.config.browser_search!==true||state.config.api_base)throw new Error('무료 브라우저 검색 설정이 아닙니다');
    if(state.config.index_path!=='static/browser-index-public-v1/')throw new Error('승인되지 않은 인덱스 경로');
    state.meta=await json(asset(state.config.index_path+'meta.json'));
    if(state.meta.dim!==384||state.meta.pooling!=='mean_attention_mask'||state.meta.query_prefix!=='query: '||state.meta.excerpt_limit!==200)throw new Error('인덱스 전처리 불일치');
    // A small catalog remains usable even if the lexical asset fails.
    state.docs=await checked('docs.json');state.docMap=new Map(state.docs.map(d=>[d.id,d]));
    const select=$('sn');const present=new Set(Array.from(select.options,o=>o.value));
    [...new Set(state.docs.map(d=>String(d.sn)))].sort((a,b)=>Number(a)-Number(b)).forEach(sn=>{if(!present.has(sn)){const o=node('option',sn);o.value=sn;select.append(o);}});
    [state.chunks,state.lex]=await Promise.all([checked('chunks.json'),checked('lexical.json')]);
    if(state.chunks.length!==state.meta.count||state.chunks.some(c=>typeof c.t!=='string'||[...c.t].length>200||!state.docMap.has(c.d)))throw new Error('공개 발췌 구조 불일치');
    $('indexStatus').textContent=state.meta.count.toLocaleString()+'개 청크 · 384차원 e5-small · 공개 발췌 최대 200자 · SHA-256 검증 완료';
    return state;
  })();return state.corePromise;
}
async function loadVectors(){
  if(!state.vectorPromise)state.vectorPromise=(async()=>{
    const bytes=await checked('vectors.bin');const v=new Float32Array(bytes);
    if(v.length!==state.meta.count*state.meta.dim)throw new Error('벡터 차원 불일치');
    for(const x of v)if(!Number.isFinite(x))throw new Error('유효하지 않은 벡터');
    state.vectors=v;return v;
  })().catch(e=>{state.vectorPromise=null;throw e;});
  return state.vectorPromise;
}
async function loadModel(){
  if(!state.modelPromise)state.modelPromise=(async()=>{
    status('의미 검색 모델 준비 중','최초 다운로드 약 135MB + 벡터 10MB + 런타임. 네트워크·기기 환경에 따라 몇 분 걸릴 수 있습니다.');
    const T=await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2/dist/transformers.min.js');
    T.env.allowLocalModels=false;
    T.env.backends.onnx.wasm.numThreads=1;
    state.pipe=await T.pipeline('feature-extraction',state.meta.browser_model,{revision:state.meta.browser_model_revision,dtype:'q8',device:'wasm',progress_callback:p=>{
      if(p.status==='progress'&&p.total) $('indexStatus').textContent='모델 '+p.file+' · '+Math.round(100*p.loaded/p.total)+'%';
    }});
    return state.pipe;
  })().catch(e=>{state.modelPromise=null;state.pipe=null;throw e;});
  return state.modelPromise;
}
export function tokenize(query,lex){
  const stop=new Set(lex.stop),out=[];
  for(const w of String(query).toLowerCase().replace(/[^가-힣a-z0-9]+/g,' ').split(/\s+/)){out.push(w);if(lex.synonyms[w])out.push(...lex.synonyms[w].split(' '));}
  return [...new Set(out.filter(w=>w.length>=2&&!stop.has(w)))];
}
function allowed(ci,filter){const d=state.docMap.get(state.chunks[ci].d);return(!filter.dtype||d.dtype===filter.dtype)&&(!filter.sn||String(d.sn)===String(filter.sn));}
export function lexicalRank(query,lex,eligible=()=>true,k=20){
  const words=tokenize(query,lex),scores=new Map();
  if(!words.length)return [];
  for(const w of words)for(const [i,n] of lex.postings[w]||[]){if(!eligible(i))continue;const v=scores.get(i)||[0,0];v[0]+=Math.log(1+n);v[1]++;scores.set(i,v);}
  return [...scores].map(([i,[hit,coverage]])=>({i,score:hit*(.5+coverage/words.length)})).filter(r=>r.score>0).sort((a,b)=>b.score-a.score||a.i-b.i).slice(0,k);
}
export function reciprocalRank(lex,sem){
  const scores=new Map();lex.forEach((r,i)=>scores.set(r.i,(scores.get(r.i)||0)+1/(61+i)));
  sem.forEach((r,i)=>scores.set(r.i,(scores.get(r.i)||0)+.35/(61+i)));
  return [...scores].map(([i,score])=>({i,score})).sort((a,b)=>b.score-a.score||a.i-b.i);
}
async function semanticRank(query,filter,job){
  await Promise.all([loadVectors(),loadModel()]);
  if(job!==state.generation)return [];
  const run=state.inference.catch(()=>{}).then(async()=>{
    if(job!==state.generation)return [];
    const tensor=await state.pipe(state.meta.query_prefix+query,{pooling:'mean',normalize:true,truncation:true,max_length:state.meta.max_length});
    const q=tensor.data;if(q.length!==state.meta.dim||Array.from(q).some(x=>!Number.isFinite(x)))throw new Error('질의 임베딩 차원/값 불일치');
    let norm=0;for(const x of q)norm+=x*x;if(Math.abs(norm-1)>.01)throw new Error('질의 정규화 불일치');
    const ranked=[];
    for(let i=0;i<state.chunks.length;i++){if(!allowed(i,filter))continue;let score=0;const offset=i*state.meta.dim;for(let d=0;d<q.length;d++)score+=state.vectors[offset+d]*q[d];ranked.push({i,score});}
    return ranked.sort((a,b)=>b.score-a.score||a.i-b.i).slice(0,10);
  });state.inference=run;return run;
}
function cleanSnippet(text){
  // Remove known presentation tags only; keep literal values and render with textContent.
  return String(text).replace(/<\/?(?:table|thead|tbody|tfoot|tr|td|th|span|div|p|br|h[1-6]|ul|ol|li|b|strong|em|i|a|img|sup|sub)\b[^>]*>/gi,' ').replace(/<\/?(?:table|tr|td|th|span|div|p|br)\b[^>]*$/gi,'').replace(/&nbsp;/gi,' ').replace(/\s+/g,' ').trim();
}
function evidence(d,chunk){const url=new URL('document/'+encodeURIComponent(d.id)+(location.pathname.endsWith('.html')?'.html':''),site);if(chunk!==undefined)url.hash='chunk-'+chunk;return url.href;}
function render(rows,mode,query,warning=''){
  $('results').replaceChildren();$('warnings').textContent=warning;
  $('resultCount').textContent=rows.length+'건 · 표시된 상위 결과';
  const labels={hybrid:'하이브리드',semantic:'의미 (벡터)',lexical:'어휘',catalog:'카탈로그'};
  $('matchCounts').textContent='실행 방식: '+labels[mode]+' · 사용자 브라우저에서 처리 · 후보 근거이며 진단 아님';
  $('empty').hidden=rows.length>0;$('empty').textContent='결과가 없습니다. 단어·필터를 바꾸어 검색하세요.';
  for(const r of rows){
    const c=mode==='catalog'?null:state.chunks[r.i];const d=c?state.docMap.get(c.d):r;
    const card=node('article',undefined,'room-result');card.dataset.docId=d.id;
    const meta=node('div');meta.append(node('p',d.dtype,'record-institution'),node('p',d.ds_name||d.title,'record-source'));
    const body=node('div'),a=node('a',c?.l||d.title,'record-title');a.href=evidence(d,c?.i);
    body.append(a,node('p',cleanSnippet(c?.t||d.title),'record-snippet'),node('p','출처: '+(d.ds_name||d.title)+' · AI 허브(aihub.or.kr)','record-source'));
    const action=node('div',undefined,'record-action');const proof=node('a','사이트 내 근거 발췌 →');proof.href=evidence(d,c?.i);action.append(proof);
    if(/^\d+$/.test(String(d.sn))){const original=node('a','AI Hub 메타데이터 ↗');original.href='https://aihub.or.kr/aihubdata/data/view.do?dataSetSn='+encodeURIComponent(d.sn);original.rel='noopener noreferrer';original.target='_blank';action.append(original);}
    card.append(meta,body,action);$('results').append(card);
  }
  $('results').dataset.mode=mode;
}
function catalog(query,filter){const words=String(query).toLowerCase().split(/\s+/).filter(Boolean);return state.docs.filter(d=>(!filter.dtype||d.dtype===filter.dtype)&&(!filter.sn||String(d.sn)===String(filter.sn))&&(!words.length||words.some(w=>(d.title+' '+d.ds_name+' '+d.id).toLowerCase().includes(w)))).slice(0,50);}
async function search(query,requested='hybrid',filter={}){
  const job=++state.generation;query=String(query).trim().slice(0,500);
  $('error').hidden=true;$('warnings').textContent='';$('loading').hidden=false;$('cancel').disabled=false;
  const mode=['hybrid','semantic','lexical','catalog'].includes(requested)?requested:'hybrid';
  try{
    await loadCore();if(job!==state.generation)return {cancelled:true};
    if(mode==='catalog'||!query){const rows=catalog(query,filter);render(rows,'catalog',query);capabilities();return {mode:'catalog',rows};}
    const lex=lexicalRank(query,state.lex,i=>allowed(i,filter));let rows=lex,actual=mode,warning='';
    if(mode==='semantic'||mode==='hybrid'){
      try{const sem=await semanticRank(query,filter,job);rows=mode==='semantic'?sem:reciprocalRank(lex,sem);}
      catch(e){actual='lexical';rows=lex;warning='의미 검색을 실행할 수 없어 어휘 검색으로 대체했습니다. '+e.message;}
    }
    if(job!==state.generation)return {cancelled:true};
    rows=rows.slice(0,10);render(rows,actual,query,warning);capabilities();
    if(warning)status('어휘 검색으로 대체 실행','의미 검색 준비/실행 실패. 결과는 어휘 검색이며 의미 검색 성공으로 표시하지 않습니다.');
    return {mode:actual,warning,rows:rows.map(r=>({...r,doc_id:state.chunks[r.i].d,chunk_id:state.chunks[r.i].i}))};
  }catch(e){
    if(job!==state.generation)return {cancelled:true};
    if(state.docs.length){const rows=catalog(query,filter);render(rows,'catalog',query,'검색 인덱스 로드 실패: 카탈로그만 조회했습니다. '+e.message);status('카탈로그 대체 조회','어휘·의미 검색 인덱스를 사용할 수 없습니다.');return {mode:'catalog',rows};}
    $('errorText').textContent='검색 준비 실패: '+e.message;$('error').hidden=false;status('검색 준비 실패','파일 다운로드를 확인하고 페이지를 다시 열어 주세요.');return {error:e.message};
  }finally{if(job===state.generation){$('loading').hidden=true;$('cancel').disabled=true;}}
}
function formFilter(){return {dtype:$('dtype').value,sn:$('sn').value};}
function saveQuery(q,mode,filter){const p=new URLSearchParams();if(q)p.set('q',q);p.set('mode',mode);if(filter.dtype)p.set('dtype',filter.dtype);if(filter.sn)p.set('sn',filter.sn);history.replaceState(null,'',location.pathname+'?'+p);}
$('searchForm').addEventListener('submit',e=>{e.preventDefault();const q=$('q').value,mode=$('mode').value,filter=formFilter();saveQuery(q,mode,filter);search(q,mode,filter);});
$('cancel').addEventListener('click',()=>{state.generation++;$('loading').hidden=true;$('cancel').disabled=true;$('warnings').textContent='검색 결과 표시를 취소했습니다. 이미 시작한 모델 다운로드/추론은 계속될 수 있습니다.';});
$('reset').addEventListener('click',()=>{state.generation++;$('q').value='';$('dtype').value='';$('sn').value='';$('mode').value='hybrid';saveQuery('','hybrid',{});search('','catalog',{});});
const params=new URLSearchParams(location.search);$('q').value=params.get('q')||'';$('mode').value=params.get('mode')||'hybrid';if(!$('mode').value)$('mode').value='hybrid';$('dtype').value=params.get('dtype')||'';
status('무료 브라우저 검색 준비 중','메타데이터와 제한 발췌·어휘 인덱스를 검증하고 있습니다.');
const ready=loadCore().then(()=>{$('sn').value=params.get('sn')||'';capabilities();return true;}).catch(()=>false);
window.MfgBrowserSearch={search,ready,inspect:()=>({count:state.meta?.count,dim:state.meta?.dim,modelReady:!!state.pipe,fullTextPublished:false})};
ready.then(()=>{if(params.get('q'))search($('q').value,$('mode').value,{dtype:params.get('dtype')||'',sn:params.get('sn')||''});else search('','catalog',{});});
