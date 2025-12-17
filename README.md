# AI 논문 주입형 대화 챗봇

간단한 CLI 도구로 논문(PDF/텍스트)을 주입해두고, 저장된 내용 기반으로 질의응답을 할 수 있습니다.

## 요구사항 충족 방법
- `python app.py add <파일경로>` 명령으로 논문을 손쉽게 추가합니다.
- 추가된 논문의 단락을 토큰 기반으로 검색하여, 질문과 가장 유사한 부분을 반환해 대화에 활용합니다.

## 빠른 시작
1. 의존성 설치
   ```bash
   pip install -r requirements.txt
   ```
2. 논문 추가
   ```bash
   python app.py add sample.pdf --title "샘플 논문"
   ```
3. 대화
   ```bash
   python app.py chat "이 논문의 핵심 기여는?"
   ```

## 기타 명령
- 저장된 논문 목록: `python app.py list --verbose`
- 텍스트 파일도 동일하게 추가할 수 있습니다.

## 참고
- 저장 데이터는 `data/papers.json`에 보관되며, 로컬에서만 사용됩니다.
