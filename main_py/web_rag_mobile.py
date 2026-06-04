import os
import sys
import logging
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import LOG_PATH
from api_client import stream_chat, is_server_ready
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# 로그 설정
logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8",
)

st.set_page_config(
    page_title="국룰 : AI와 함께 공공입찰 맥잡기",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

:root {
    --bg-primary  : #0a0f1e;
    --bg-card     : #111d35;
    --accent      : #2563eb;
    --text-primary: #e2e8f0;
    --text-muted  : #94a3b8;
    --border      : #1e3a5f;
}

html, body, .stApp {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Pretendard', sans-serif !important;
}

section[data-testid="stSidebar"]  { display: none !important; }
[data-testid="collapsedControl"]  { display: none !important; }

.main-title {
    font-family: 'Pretendard', sans-serif;
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
    font-family: 'Pretendard', sans-serif;
    margin-bottom: 1rem;
}

.stChatMessage {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

.stChatInputContainer textarea,
.stChatInputContainer input {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
    caret-color: white !important;
}
.stChatInputContainer textarea:focus,
.stChatInputContainer input:focus {
    border-color: white !important;
    outline: none !important;
    box-shadow: 0 0 0 1px white !important;
}

.stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Pretendard', sans-serif !important;
}
.stButton > button:hover {
    background: #1d4ed8 !important;
}

.block-container {
    padding-top: 3rem !important;
    padding-bottom: 1rem !important;
    max-width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = "local"
if "llm_model" not in st.session_state:
    st.session_state.llm_model = ""

# 서버 상태 확인
if not is_server_ready():
    st.error("API 서버가 준비되지 않았습니다. fastapi_server.py를 먼저 실행해주세요.")
    st.stop()

# 타이틀
st.markdown('<div class="main-title">국룰 : AI와 함께 공공입찰 맥잡기</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-caption">RFP 문서 기반 AI 어시스턴트</div>', unsafe_allow_html=True)

# LLM 선택
provider_options = {
    "local"      : "로컬 (Phi-4)",
    "openai"     : "GPT-4o-mini",
    "gemini"     : "Gemini 2.5 Flash",
    "openrouter" : "gemma-4-26b-a4b-it(Openrouter)",
}
col_model, col_new = st.columns([3, 1])
with col_model:
    selected = st.selectbox(
        "모델",
        options=list(provider_options.keys()),
        format_func=lambda x: provider_options[x],
        index=list(provider_options.keys()).index(st.session_state.llm_provider),
        label_visibility="collapsed",
    )
    if selected != st.session_state.llm_provider:
        st.session_state.llm_provider = selected
with col_new:
    if st.button("새 대화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 이전 대화 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 입력창
if prompt := st.chat_input("질문하세요"):
    logging.info(f"[MOBILE][USER]: {prompt}")
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

            key_map = {"openai": os.getenv("OPENAI_API_KEY",""), "gemini": os.getenv("GEMINI_API_KEY",""), "openrouter": os.getenv("OPENROUTER_API_KEY",""), "local": ""}
            api_key = key_map.get(st.session_state.llm_provider, "")
            llm_cfg = {"provider": st.session_state.llm_provider, "api_key": api_key, "model": st.session_state.llm_model}
            with st.status("검색 중...", expanded=False) as status:
                for chunk in stream_chat(prompt, history=history if history else None, llm_config=llm_cfg):
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
            logging.info("[MOBILE][ANSWER]: 완료")

        except Exception as e:
            logging.error(f"[MOBILE][ERROR]: {e}")
            st.error(f"오류: {e}")
