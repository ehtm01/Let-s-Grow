# Linear LLM Studio

웹에서 에이전트를 만들고 순서대로 실행하는 Streamlit 대시보드입니다.

## 로컬 실행

Python 3.12 환경에서 두 파일을 같은 폴더에 저장하고 실행합니다.

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Streamlit Community Cloud 배포

1. GitHub 저장소의 같은 폴더에 `app.py`와 `requirements.txt`를 업로드합니다.
2. https://share.streamlit.io 에 로그인하고 새 앱을 만듭니다.
3. 저장소와 브랜치를 고른 뒤 Main file path에 `app.py`를 지정합니다. 하위 폴더에 올렸다면 해당 경로를 지정합니다.
4. Advanced settings에서 Python 3.12를 선택하고 Deploy를 누릅니다.
5. 배포된 웹페이지의 사이드바에 본인의 OpenAI API key를 입력합니다. 키를 GitHub에 올릴 필요가 없습니다.

## 사용 순서

1. 기본 분석가/작성자를 편집하거나 에이전트를 추가·삭제합니다.
2. 위/아래 버튼으로 순서를 정하고 이름, 모델 ID, System prompt, 추가 prompt, 최대 출력 토큰을 설정합니다.
3. 원하는 에이전트만 RAG를 켜고 파일을 업로드 영역에 끌어 놓습니다.
4. User prompt를 입력하고 워크플로 실행을 누릅니다.
5. 단계별 결과에서 입력·추가 요청·검색 근거를 확인합니다. 결과 JSON과 최종 답변 TXT를 다운로드할 수 있습니다.

첫 단계는 User prompt를 받습니다. 그다음 단계는 바로 앞 단계의 출력, 자신의 추가 prompt, 자신의 검색 문단만 입력으로 받습니다. 원래 User prompt와 전체 이력이 자동으로 재첨부되지는 않습니다. System prompt는 각 API 호출의 instructions로 전달됩니다.

## RAG와 파일 지원

- PDF(텍스트 PDF), TXT, Markdown, CSV, DOCX(본문과 표).
- UTF-8 및 CP949 텍스트. 스캔 PDF OCR, 이미지 해석, 암호화 PDF는 지원하지 않습니다.
- 1,200자 문단과 200자 중첩으로 분할한 뒤 문자 2~4-gram TF-IDF 코사인 유사도로 검색합니다. 한국어를 지원하며 별도 임베딩 API나 벡터 DB가 필요 없습니다. 의미 기반 임베딩 검색과 달리 표현이 완전히 다른 동의어는 놓칠 수 있습니다.
- 에이전트당 최대 5개·총 20 MB, 파일당 10 MB·추출 텍스트 20만 자, PDF 200페이지까지입니다. CSV는 표 전체의 수치 분석이 아니라 텍스트 문단 검색으로 처리합니다.
- 검색 근거에는 파일명과 문단 번호가 붙습니다. 문단 번호는 원본 PDF 페이지 번호가 아닙니다.
- 파일이 변경되면 해당 에이전트의 검색 인덱스를 다시 만듭니다. 다른 사용자와 공유되는 전역 캐시는 사용하지 않습니다.

## 세션과 API

- API key는 비밀번호 입력창에서 받고 현재 Streamlit 세션 메모리에만 유지합니다. 코드가 키를 파일·환경변수·결과 다운로드에 저장하지 않습니다.
- 파일은 앱 서버 메모리에서 처리합니다. 실행 시 프롬프트와 검색된 문단은 OpenAI에 전송됩니다. API 요청에는 `store=False`를 지정하지만 OpenAI의 별도 데이터 보존 정책을 대체하는 것은 아닙니다.
- 키·설정·파일·결과는 영구 저장하지 않습니다. 새로고침·연결 종료·서버 재시작 시 사라질 수 있습니다. 사이드바의 세션 초기화 버튼으로 지울 수 있습니다.
- 기본 모델 ID는 `gpt-4.1-mini`이며 UI에서 접근 가능한 Responses API 텍스트 모델 ID로 변경할 수 있습니다. 계정별 모델 접근 권한과 API 결제가 필요합니다.
- 각 단계에서 API 사용료가 발생합니다. 실패하면 뒤 단계는 실행하지 않고 앞 단계의 성공 결과를 유지합니다. 재실행은 첫 단계부터 다시 호출합니다.
- 응답이 출력 한도 등으로 미완료되면 다음 단계로 넘기지 않습니다. 한도를 늘리거나 입력을 줄여 재실행하세요.

## 검증 범위

Python 3.12 환경에서 고정된 requirements 의존성을 설치해 Streamlit AppTest 및 단위 테스트로 에이전트 추가·삭제·순서 변경, 설정/업로드 유지, 순차 전달, 추가 prompt, 파일 추출/검색, RAG 파일 변경 반영, 오류 중단, 키 초기화를 검증했습니다. 실제 API key를 사용하는 유료 호출과 실제 Community Cloud 배포, 브라우저 화면의 시각 검수는 수행하지 않았습니다.

공식 참고 문서:
- https://developers.openai.com/api/docs/guides/text
- https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
