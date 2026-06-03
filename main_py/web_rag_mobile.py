import os
import sys
import logging
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import LOG_PATH
from service import GenerationService

# 로그 설정
logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8",
)

# Streamlit 설정
st.set_page_config(
    page_title="국룰:RFP 맥잡기",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&family=Space+Mono:wght@400;700&display=swap');

:root {
    --bg-primary  : #0a0f1e;
    --bg-card     : #111d35;
    --accent      : #2563eb;
    --accent-light: #3b82f6;
    --text-primary: #e2e8f0;
    --text-muted  : #94a3b8;
    --border      : #1e3a5f;
}

html, body, .stApp {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Noto Sans KR', sans-serif !important;
}

/* 사이드바 완전 숨김 */
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"]  { display: none !important; }

/* 타이틀 */
.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    color: #cbd5e1;
    border-bottom: 2px solid var(--accent);
    padding-bottom: 0.4rem;
    margin-bottom: 0.3rem;
}
.sub-caption {
    font-size: 0.7rem;
    color: var(--text-muted);
    font-family: 'Space Mono', monospace;
    margin-bottom: 1rem;
}

/* 채팅 메시지 */
.stChatMessage {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

/* 입력창 */
.stChatInputContainer, .stTextInput input {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
    caret-color: white !important;
}
.stChatInputContainer textarea:focus,
.stChatInputContainer input:focus,
.stChatInputContainer > div:focus-within,
.stChatInputContainer > div > div:focus-within {
    border-color: white !important;
    outline: none !important;
    box-shadow: 0 0 0 1px white !important;
}
* :focus {
    outline-color: white !important;
}
/* 버튼 */
.stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Noto Sans KR', sans-serif !important;
}
.stButton > button:hover {
    background: #1d4ed8 !important;
}

/* 상단 여백 줄이기 */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
    max-width: 100% !important;
}

/* status */
.stStatusWidget {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
}
</style>
""", unsafe_allow_html=True)

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# 서비스 초기화
@st.cache_resource
def load_service():
    return GenerationService()

try:
    with st.spinner("로딩 중..."):
        svc = load_service()
except Exception as e:
    st.error(f"초기화 오류: {e}")
    st.stop()

# 타이틀
st.markdown('<div class="main-title">국룰:RFP 맥잡기</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-caption">RFP 문서 기반 AI 어시스턴트</div>', unsafe_allow_html=True)

# 대화 초기화 버튼
if st.button("새 대화", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

# 이전 대화 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 입력창
if prompt := st.chat_input("RFP에 대해 질문하세요"):
    logging.info(f"[MOBILE][USER QUESTION]: {prompt}")
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
        if m["role"] in ("user", "assistant")
    ]

    with st.chat_message("assistant"):
        try:
            answer_chunks = []
            sources_text  = ""

            with st.status("검색 중...", expanded=False) as status:
                for chunk in svc.stream(prompt, history=history if history else None):
                    if chunk["type"] == "meta":
                        pass
                    elif chunk["type"] == "chunk":
                        answer_chunks.append(chunk["data"])
                    elif chunk["type"] == "done":
                        sources_text = chunk["data"]["sources_text"]
                status.update(label="완료", state="complete")

            answer_full = "".join(answer_chunks)
            if sources_text and "[출처]" not in answer_full:
                answer_full += "\n" + sources_text

            st.markdown(answer_full)

            st.session_state.messages.append({"role": "assistant", "content": answer_full})
            logging.info("[MOBILE][SYSTEM ANSWER]: 답변 생성 완료")

        except Exception as e:
            logging.error(f"[MOBILE][SYSTEM ERROR]: {e}")
            st.error(f"오류: {e}")
