# 제조현장 기술지원 Copilot (mfg-tech-copilot)

AI 허브(aihub.or.kr) 기술문서 **83종 데이터셋 · 153건 설명서/활용가이드 · 5,101개 섹션**을
검색·추천·비교하는 웹 서비스입니다. SQLite FTS5(trigram) 인덱스 + 동의어 확장 검색을 사용합니다.

## 주요 기능
- 문서/섹션 통합 검색 (동의어 확장: 화재→열폭주·탄화·발화 등)
- 요구사항 기반 데이터셋 추천 (근거 섹션 라벨 포함)
- 두 데이터셋 구성·규모 비교
- 시연 질문 6종 제공 (도입 검토 시연용)

## 로컬 실행
```bash
cd webapp
pip install -r requirements.txt
python app.py            # http://127.0.0.1:8142
```

## Render 배포
1. GitHub 저장소 연결 (New → Web Service)
2. `render.yaml` 설정 자동 인식 (rootDir: webapp, gunicorn 시작)
3. 배포 후 `https://<서비스명>.onrender.com` 공개 URL 발급

## API
| 엔드포인트 | 설명 |
|---|---|
| `/api/stats` | 코퍼스 통계 |
| `/api/search?q=` | 문서 검색 (dtype·sn·limit) |
| `/api/sections?q=` | 섹션 검색 |
| `/api/recommend?q=` | 데이터셋 추천 |
| `/api/compare?a=&b=` | 데이터셋 비교 |
| `/api/document/<doc_id>` | 문서 원문 (full=1 전체) |
| `/api/datasets` | 데이터셋 목록 |

## 데이터 출처 및 라이선스
본 서비스의 문서 내용은 **AI 허브(aihub.or.kr)** 데이터셋 설명서·구축활용가이드에서 발췌·색인했습니다.
- 출처 표기 의무: 화면·답변에 AI 허브 명칭 표기
- 원본 데이터 재배포 금지 — 색인·발췌 제공에 한정
- 본 서비스는 실시간 설비 예측이 아닌 기술문서 검색·검토 지원 도구입니다.
