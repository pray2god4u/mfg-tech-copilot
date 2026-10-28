#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI Hub 신규 데이터셋 모니터링 — 제조·로봇 분야 신규 공개 데이터셋 확인
실행: 매주 cron. 스냅샷(data/monitor/known_datasets.json)과 비교해 신규·변동 보고.
"""
import json, os, urllib.request, sys
from datetime import datetime

KEY = os.environ.get("AIHUB_API_KEY", "74684F49-E5C8-4C8F-B046-D4C565006EF4")
BASE = "https://aihub.or.kr"
HEADERS = {"X-API-KEY": KEY, "X-MCP-Client": "oneai-desktop", "X-MCP-Tool": "searchDataSets"}
SNAP = os.path.expanduser("~/.oneai/workspace/mfg-tech-copilot/data/monitor/known_datasets_complete_20261026.json")
REALMS = {"REALM012": "제조", "REALM011": "로봇"}

def val(node):
    return node.get("value") if isinstance(node, dict) else node

def fetch_realm(realm):
    out = {}
    offset = 0
    while True:
        url = f"{BASE}/mcp/dataSetList.do?srchDataRealmCode={realm}&recordCountPerPage=200&firstIndex={offset}"
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.loads(r.read().decode("utf-8"))
        if d.get('result') != 'SUCCESS':
            raise RuntimeError(d.get('message', 'AI Hub 조회 실패'))
        items = d.get("data", {}).get("list", {}).get("value", [])
        before = len(out)
        for it in items:
            sn = str(val(it.get("dataSetSn")))
            out[sn] = {"name": val(it.get("dataNm")), "realm": REALMS.get(realm, realm), "url": val(it.get("url"))}
        if len(items) < 200:
            break
        if len(out) == before:
            raise RuntimeError('페이지 진행이 멈춰 전체 조회를 확인할 수 없습니다')
        offset += len(items)
    return out

def main():
    current = {}
    for realm in REALMS:
        current.update(fetch_realm(realm))

    known = {}
    if os.path.exists(SNAP):
        with open(SNAP) as f:
            known = json.load(f)

    baseline = not os.path.exists(SNAP)
    new_sns = [] if baseline else sorted(set(current) - set(known))
    gone_sns = sorted(set(known) - set(current))
    renamed = [sn for sn in set(current) & set(known)
               if current[sn]["name"] != known[sn].get("name")]

    report = {
        "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total": len(current),
        "baseline_created": baseline,
        "new": [{"sn": sn, **current[sn]} for sn in new_sns],
        "removed": [{"sn": sn, **known[sn]} for sn in gone_sns],
        "renamed": [{"sn": sn, "old": known[sn].get("name"), "new": current[sn]["name"]}
                    for sn in renamed],
    }

    # 스냅샷 갱신
    os.makedirs(os.path.dirname(SNAP), exist_ok=True)
    with open(SNAP, "w") as f:
        json.dump(current, f, ensure_ascii=False, indent=1)

    print(json.dumps(report, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    sys.exit(main())
