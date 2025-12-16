# 통합형 인공지능 설계 개요

## 문제 정의
쉽고 간편하고, 필요에 따라 유동적으로 기능을 추가·삭제할 수 있는 통합형 인공지능 플랫폼이 필요하다.

## 목표
- **모듈식 구조**: 기능을 플러그인 단위로 추가·삭제 가능
- **핫스왑**: 서비스 중단 없이 모듈 교체/비활성화
- **일관된 인터페이스**: 모든 모듈이 동일한 계약(Manifest & Handler)을 따른다
- **관측성/안전성**: 모듈별 상태, 리소스, 정책을 중앙에서 제어

## 아키텍처 요약
1. **Core Orchestrator**: 라우팅·스케줄링·리소스 할당, 공통 인증/로깅 제공  
2. **Module Registry**: Manifest 기반 모듈 등록/버전 관리, enable/disable 플래그 지원  
3. **Skill Plugins**: 실제 기능 구현체. 컨테이너, 함수, 또는 패키지 단위로 배포  
4. **Adapter Layer**: 외부 채널(API, Chat, Webhook)을 Core에 연결  
5. **Memory & Context Store**: 대화/작업 컨텍스트를 모듈 간 공유  
6. **Policy Guard**: 권한/레이트리밋/감사 로깅 일괄 적용

## 모듈 Manifest 예시
```yaml
name: sentiment-analyzer
version: 0.1.0
enabled: true                            # optional, 없으면 기본 true, 비활성화 시 false
entrypoint: sentiment:handler            # 예: sentiment:handler(패키지:메서드) 또는 https://api.example.com/invoke(HTTP)
intents:
  - classify-sentiment
inputs:
  text: string
outputs:
  label: { type: string, enum: [positive, neutral, negative] }
  score: number
resources:
  cpu: 100m
  memory: 128Mi
policies:
  auth: service-token
  rate_limit: 50/min                     # requests per minute
healthcheck: /health
```

### 추가/삭제 플로우
1) Manifest 작성 → Registry에 등록  
2) Core가 Registry를 워치하며 새 모듈을 동적으로 로드/핫스왑  
3) 모듈 비활성화 시 `enabled: false` 혹은 Manifest 제거 → Core가 라우팅 제외  
4) Adapter는 Core의 라우팅 테이블을 참조하므로 채널별 수정 없이 즉시 반영

## 최소 API 계약 (의사 코드)
```python
registry.register(manifest)
core.reload()  # 핫스왑 적용

response = core.invoke(
    intent="classify-sentiment",
    payload={"text": "서비스 정말 좋아요!"}
)

registry.disable("sentiment-analyzer")
core.reload()
```

## 운영 체크리스트
- 모듈별 메트릭/로그 수집 및 대시보드화
- 배포 전 계약 검증(스키마/헬스체크)
- 롤백: 이전 Manifest로 즉시 되돌리기
- 샌드박스/권한 최소화로 안전성 확보

## 다음 단계
- Manifest 스키마와 레퍼런스 구현체 패키지화
- `registry` 및 `core`의 실 구현 코드/테스트 추가
