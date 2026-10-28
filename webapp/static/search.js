/* Trusted deployment config only; no URL-supplied API endpoint. No HTML injection. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const publicStatic = location.pathname.endsWith('.html');
  const site = new URL(publicStatic ? './' : '/', location.href);
  let base = '', catalog = [], controller = null, generation = 0, ready = false;
  const node = (tag, text, cls) => { const n = document.createElement(tag); if (text !== undefined) n.textContent = String(text ?? ''); if (cls) n.className = cls; return n; };
  function error(text) { $('errorText').textContent = text; $('error').hidden = false; }
  function endpoint(path) { return base + '/api/' + path; }
  async function json(url, signal) {
    const r = await fetch(url, {signal, credentials:'omit', redirect:'error', headers:{Accept:'application/json'}});
    if (!r.ok) throw new Error('HTTP ' + r.status);
    return r.json();
  }
  function describe(v) { return typeof v === 'object' ? JSON.stringify(v) : String(v ?? '정보 없음'); }
  function evidence(r) {
    if (r.evidence_url && base) {
      try {
        const direct = new URL(r.evidence_url, base + '/');
        if (direct.origin === new URL(base).origin && direct.pathname.startsWith('/evidence/')) return direct.href;
      } catch (_) {}
    }
    const id = String(r.doc_id ?? r.id ?? '');
    const fragment = r.chunk_id !== undefined && r.chunk_id !== null ? 'chunk-' + encodeURIComponent(String(r.chunk_id)) : 'section-' + encodeURIComponent(String(r.sec_idx ?? 0));
    return new URL('document/' + encodeURIComponent(id) + (publicStatic ? '.html' : '') + '#' + fragment, site).href;
  }
  function matched(r, type) {
    const shorthand = type === 'lexical' ? 'lex' : 'sem';
    return String(r.matched ?? '').split('+').includes(shorthand) || r[type + '_matched'] === true || r[type + '_rank'] != null || r[type + '_score'] != null || (Array.isArray(r.matched_by) && r.matched_by.includes(type));
  }
  function render(rows, mode) {
    $('results').replaceChildren();
    let lexical = 0, semantic = 0;
    for (const r of rows) {
      const card = node('article', undefined, 'room-result');
      const meta = node('div'); meta.append(node('p', String(r.sn ?? '') + ' · ' + String(r.dtype ?? ''), 'record-institution'), node('p', r.ds_name ?? r.doc_id ?? r.id, 'record-source'));
      const body = node('div'); const link = node('a', r.label ?? r.title ?? r.doc_id, 'record-title'); link.href = evidence(r);
      body.append(link, node('p', r.snip ?? '', 'record-snippet'));
      const tags = node('div', undefined, 'record-tags');
      if (matched(r, 'lexical') || mode === 'lexical') { lexical++; tags.append(node('span', '어휘 일치', 'badge manual')); }
      if (matched(r, 'semantic') || mode === 'semantic') { semantic++; tags.append(node('span', '의미 일치', 'badge guide')); }
      if (mode === 'catalog') tags.append(node('span', '메타데이터만', 'badge'));
      if (mode === 'hybrid' && !matched(r,'lexical') && !matched(r,'semantic')) tags.append(node('span', '일치 유형 미제공'));
      body.append(tags);
      const action = node('div');
      action.append(node('p', Number.isFinite(r.score) ? '검색 점수 ' + r.score.toFixed(4) : '진단 아님', 'record-facts'));
      const proof = node('a', '사이트 내 근거 발췌 →'); proof.href = evidence(r); action.append(proof);
      if (/^\d+$/.test(String(r.sn ?? ''))) { const original = node('a', 'AI Hub 메타데이터 ↗'); original.href = 'https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=' + encodeURIComponent(r.sn); original.className='secondary-source'; original.rel='noopener noreferrer'; original.target='_blank'; action.append(original); }
      card.append(meta, body, action); $('results').append(card);
    }
    $('resultCount').textContent = rows.length + '건' + (mode === 'catalog' ? ' · 카탈로그' : ' · 상위 10건');
    $('matchCounts').textContent = mode === 'catalog' ? '카탈로그 메타데이터 검색 — 본문·벡터 검색 아님' : '어휘 일치 ' + lexical + ' · 의미 일치 ' + semantic + ' (중복 포함; 미제공 유형은 집계하지 않음)';
    $('empty').hidden = rows.length > 0; $('empty').textContent = '일치하는 결과가 없습니다. 다른 검색어·필터를 시도해 주세요.';
  }
  function stop() { if (controller) controller.abort(); generation++; controller = null; $('loading').hidden=true; $('cancel').disabled=true; }
  async function search() {
    stop(); const request = generation;
    $('error').hidden = true; $('empty').hidden = true; $('results').replaceChildren(); $('warnings').textContent=''; $('resultCount').textContent='—'; $('matchCounts').textContent='어휘 일치 — · 의미 일치 —';
    const q = $('q').value.trim(), mode = $('mode').value, dtype = $('dtype').value, sn = $('sn').value;
    const p = new URLSearchParams({q, mode, dtype, sn}); history.replaceState(null, '', '?' + p.toString());
    if (!q) { error('검색어를 입력해 주세요.'); return; }
    if (mode === 'catalog') {
      const norm = s => String(s ?? '').toLowerCase().replace(/\s+/g,'');
      const words = q.split(/\s+/).map(norm).filter(Boolean);
      const rows = catalog.filter(d => (!dtype || dtype === d.dtype) && (!sn || sn === String(d.sn)) && words.every(w => norm([d.title,d.ds_name,...(d.labels || [])].join(' ')).includes(w))).map(d => ({...d, doc_id:d.id, sec_idx:d.first_sec_idx ?? 0, snip:'메타데이터: ' + (d.labels || []).slice(0,4).join(' · ')}));
      render(rows, mode); return;
    }
    if (!ready || !base) { error('검색 API 배포 대기/연결 불가: 본문·벡터 검색을 실행하지 않았습니다. 검색 방식을 카탈로그 (메타데이터만)로 선택하면 공개 목록을 조회할 수 있습니다.'); return; }
    controller = new AbortController(); const signal=controller.signal;
    const timeout=setTimeout(() => { if (request === generation) { stop(); error('검색 시간 초과: 서버 상태를 확인하거나 다시 시도해 주세요.'); } }, 45000);
    $('loading').hidden=false; $('cancel').disabled=false;
    try {
      const params = new URLSearchParams({q,dtype,sn,limit:'10'});
      const data = await json(endpoint(mode === 'lexical' ? 'sections' : mode) + '?' + params, signal);
      if (request !== generation) return;
      $('warnings').textContent = [data.status ? '검색 상태: ' + describe(data.status) : '', ...(Array.isArray(data.warnings) ? data.warnings.map(describe) : data.warnings ? [describe(data.warnings)] : [])].filter(Boolean).join(' · ');
      if (data.error || data.status === 'unavailable') throw new Error(describe(data.error || '검색 엔진 사용 불가'));
      if (!Array.isArray(data.results)) throw new Error('잘못된 검색 응답: results 누락');
      render(data.results, mode);
      if (data.status === 'degraded') $('serviceTitle').textContent='검색 서버 제한 동작 (degraded)';
    } catch(e) { if (request === generation && e.name !== 'AbortError') error('검색 실패 — 결과를 생성하지 않았습니다: ' + e.message); }
    finally { clearTimeout(timeout); if (request === generation) { $('loading').hidden=true; $('cancel').disabled=true; controller=null; } }
  }
  async function init() {
    const params = new URLSearchParams(location.search);
    $('q').value = params.get('q') ?? $('q').value;
    for (const id of ['mode','dtype']) if ([...$(id).options].some(o => o.value === params.get(id))) $(id).value=params.get(id);
    try {
      const data = await json(new URL('static/catalog.json',site)); catalog = data.documents || [];
      for (const d of data.datasets || []) { const o = node('option', d.sn + ' ' + d.ds_name.slice(0,22)); o.value=d.sn; $('sn').append(o); }
      if ([...$('sn').options].some(o => o.value === params.get('sn'))) $('sn').value=params.get('sn');
    } catch(e) { error('카탈로그 로드 실패: ' + e.message); }
    try {
      const config = await json(new URL('static/search-config.json',site));
      if (!config.api_base || config.deployment_pending === true) {
        $('serviceTitle').textContent='검색 API 배포 대기'; $('serviceStatus').textContent='승인된 검색 서버 주소가 아직 설정되지 않았습니다. 본문·벡터 검색은 실행되지 않습니다. 카탈로그 (메타데이터만)는 별도로 이용할 수 있습니다.';
      } else {
        const url = new URL(config.api_base);
        if (url.protocol !== 'https:' || url.username || url.password || url.search || url.hash) throw new Error('API 설정은 인증정보 없는 HTTPS 주소여야 합니다');
        base=url.href.replace(/\/$/,'');
        const controller=new AbortController(); const timeout=setTimeout(() => controller.abort(),20000);
        let status; try { status=await json(endpoint('status'),controller.signal); } finally { clearTimeout(timeout); }
        ready=status.status !== 'unavailable';
        $('serviceTitle').textContent = !ready ? '검색 서버 사용 불가 (unavailable)' : status.status === 'degraded' ? '검색 서버 제한 동작 (degraded)' : '검색 서버 연결됨';
        $('serviceStatus').textContent = '서버 상태: ' + describe(status.status ?? '응답 수신') + ' · 진단 기능 없음';
        $('indexStatus').textContent = '코퍼스: ' + describe(status.corpus) + ' · 벡터 인덱스: ' + describe(status.vector_index) + (status.warnings ? ' · ' + describe(status.warnings) : '');
      }
    } catch(e) { ready=false; $('serviceTitle').textContent='검색 서버 연결 불가'; $('serviceStatus').textContent='설정/네트워크/CORS/서버 상태 확인이 필요합니다: ' + e.message; }
    if ($('q').value.trim()) search();
  }
  $('searchForm').addEventListener('submit', e => {e.preventDefault(); search();});
  $('cancel').addEventListener('click', () => {stop(); error('검색 요청을 취소했습니다.');});
  $('reset').addEventListener('click', () => {stop(); $('q').value=''; $('dtype').value=''; $('sn').value=''; $('mode').value='hybrid'; $('results').replaceChildren(); $('resultCount').textContent='—'; $('error').hidden=true; $('empty').hidden=true; $('warnings').textContent=''; history.replaceState(null,'',location.pathname);});
  init();
})();
