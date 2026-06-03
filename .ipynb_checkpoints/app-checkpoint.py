import os
import logging
import streamlit as st
# 1. 환경 변수 로드 라이브러리 불러오기
from dotenv import load_dotenv
from google import genai

# 2. .env 파일의 내용을 시스템 환경 변수로 로딩
load_dotenv()

# 3. 환경 변수에서 구글 API 키 꺼내오기
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# ---- [접근 로그 인프라 설정] ----
# 경로를 './app_user_access.log'로 설정하여 스크립트와 동일한 폴더에 로그가 남습니다.
logging.basicConfig(
    filename='./app_user_access.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    encoding='utf-8'
)

# 웹브라우저 탭 타이틀 및 아이콘 설정
st.set_page_config(page_title="Gemini Chatbot UI", page_icon="♊")
st.title("2Team Custom UI (Gemini Version)")

# 4. API 키가 정상적으로 로드되었는지 확인
if not GOOGLE_API_KEY:
    st.error(".env 파일에서 GEMINI_API_KEY를 찾을 수 없습니다. 설정을 확인해 주세요.")
    st.stop()

# 5. 채팅 히스토리 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 새로고침할 때마다 중복으로 찍히지 않도록 최초 진입 시에만 접속 로그 기록
if "logged_in" not in st.session_state:
    logging.info("📢 새로운 유저가 웹 페이지에 접속했습니다.")
    st.session_state.logged_in = True

# 기존 대화 기록 화면에 다시 그리기
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력창 생성 및 처리
if prompt := st.chat_input("Gemini에게 무엇이든 물어보세요!"):
    
    # 사용자 입력 표시 및 저장
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # AI의 답변 생성 프로세스
    with st.chat_message("assistant"):
        try:
            # 6. 환경 변수에서 가져온 키로 구글 클라이언트 초기화
            client = genai.Client(api_key=GOOGLE_API_KEY)
            
            # 스트리밍 제너레이터 정의
            def gemini_stream_generator():
                response_stream = client.models.generate_content_stream(
                    model='gemini-2.5-flash',
                    contents=prompt
                )
                for chunk in response_stream:
                    yield chunk.text

            # 실시간 출력 및 저장
            response_text = st.write_stream(gemini_stream_generator())
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            
            # ---- [스트리밍 완료 후 로그 기록 실행] ----
            logging.info(f"💬 [USER QUESTION]: {prompt}")
            logging.info(f"🤖 [SYSTEM ANSWER]: Gemini 스트리밍 응답 생성 완료 (포트: 8443)")
            
        except Exception as e:
            st.error(f"API 호출 중 에러가 발생했습니다: {e}")