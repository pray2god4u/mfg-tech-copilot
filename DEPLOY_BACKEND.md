# 제조 Copilot 검색 서버 배포

## 현재 구조
- 공개 화면: https://pray2god4u.github.io/mfg-tech-copilot/ (GitHub Pages 유지)
- 검색 화면은 어휘 / 의미 / 하이브리드 모드를 제공하지만, 공개 검색 서버 주소가 설정되기 전에는 연결 대기 상태를 표시합니다.
- 실시간 검색: Flask + BAAI/bge-m3 CPU 검색 API. 사전 구축된 6,542청크 인덱스와 원문 해시를 검증합니다.
- RAG 출처: 정적 문서별 제한 발췌 페이지와 실시간 API 근거 페이지. AI Hub 링크는 보조 출처입니다.

## 1. 검색 서버 생성 (사용자 대시보드 작업)
Render에서 New → Web Service → 저장소 `pray2god4u/mfg-tech-copilot`을 선택합니다.
- 배포 방식: Docker / 저장소 루트의 `Dockerfile`
- 헬스체크: `/api/health`
- 환경변수: `PORT=10000`, `MFG_VECTOR_DEVICE=cpu`, `MFG_ALLOWED_ORIGINS=https://pray2god4u.github.io`, `MFG_TRUST_PROXY=1`
- `render-backend.yaml`은 별도 서버용 참고 Blueprint입니다. 기존 `render.yaml`은 보존했습니다.
- BGE 모델을 빌드 단계에서 다운로드하고 CPU로 로딩합니다. 최초 빌드는 오래 걸릴 수 있습니다.
- 초기 서버 메모리는 4GB 이상을 권장하되 실제 부하 테스트로 확인해야 합니다. 무료 소형 서버에서의 정상 작동을 보장하지 않습니다. 유료 서비스 생성 전 대시보드의 요금·메모리를 직접 확인하세요.
- 서버 생성·요금 결제는 자동 수행하지 않았습니다.

## 2. 배포 검증
배포가 끝난 뒤 실제 HTTPS 서비스 주소를 확인합니다. 다음이 모두 필요합니다.
- `/api/health`: 원문·인덱스 준비 확인
- `/api/status`: 의존성·코퍼스 상태 확인 (이것만으로 검색 성공을 보장하지 않음)
- `/api/semantic?q=OHT%20화재`: `status=ok`, 결과 존재
- `/api/hybrid?q=OHT%20화재`: `status=ok`, 경고 없음, 어휘+벡터 분기 확인
- 반환된 `evidence_url`을 열어 같은 문서의 실제 근거 구간 확인
- GitHub Pages 화면에서 CORS 허용 확인

## 3. 공개 사이트 연결
실제 검색 서버 HTTPS URL을 assistant에 알려주세요. 서버 검색을 검증한 후 사이트 설정과 GitHub Pages를 갱신할 수 있습니다.
개발자가 직접 연결할 경우:
`uv run --no-project --python 3.12 --with flask --with markdown --with bleach python scripts/build_public_site.py --api-base 실제_HTTPS_서비스주소`
생성된 `docs/`를 저장소의 `main`에 커밋·푸시합니다.
서버 URL은 공개 정보이며 API 키·인증 토큰을 넣지 마세요. localhost·PC 파일 주소는 공개 사이트 설정에 사용하지 않습니다.

## 보안·운영 한계
- 공개 API는 입력 길이·결과 수·허용 Origin·요청 빈도를 제한합니다. 단일 서버 프로세스 기준으로 제한하며 다중 인스턴스 공유 제한은 별도 필요합니다.
- 사용자 개인정보나 사내 기밀을 공개 검색창에 입력하지 마세요. 서비스 제공자의 요청 로그 정책도 확인하세요.
- 원문 전체의 공개 재배포가 아닌 제한된 RAG 근거 발췌 제공입니다.
- 검색 결과는 고장 진단이나 화재 발생 확정 근거가 아닙니다. `candidate_only` 및 한계 안내를 유지합니다.
- 모델은 공개 HF 저장소에서 내려받고 질의 시 로컬 CPU로 실행됩니다. 질의나 RAG 원문을 외부 LLM에 전송하지 않습니다.
- 로컬 CPU API 검증은 통과했으나, 실제 Render 배포·컨테이너 빌드·공개 접속 검증은 별도 단계입니다.
