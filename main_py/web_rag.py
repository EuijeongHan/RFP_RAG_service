import os
import sys
import logging
import streamlit as st
import pandas as pd

# main_py 폴더를 경로에 추가
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
logging.info("새로운 유저가 RAG 페이지에 접속했습니다.")

# Streamlit 설정
st.set_page_config(
    page_title="입찰메이트 RAG Dashboard",
    layout="wide",
)
st.title("입찰메이트 RAG Dashboard")
st.caption("Gemma4 E4B + LoRA + ChromaDB + BM25 + KURE-v1")

# Session State
if "messages"      not in st.session_state: st.session_state.messages      = []
if "last_meta"     not in st.session_state: st.session_state.last_meta     = {}
if "last_sources"  not in st.session_state: st.session_state.last_sources  = []
if "last_latency"  not in st.session_state: st.session_state.last_latency  = {}

# Sidebar
st.sidebar.header("RAG 설정")
top_k_display = st.sidebar.slider("Top-K (표시용)", 1, 10, 5)

if st.session_state.last_meta:
    st.sidebar.divider()
    st.sidebar.subheader("마지막 검색 정보")
    meta = st.session_state.last_meta
    st.sidebar.write(f"필터: {meta.get('filter', {})}")
    st.sidebar.write(f"재작성: {meta.get('rewritten_query', '')}")
    sub_q = meta.get("sub_queries", [])
    if len(sub_q) > 1:
        st.sidebar.write(f"서브쿼리: {sub_q}")

if st.session_state.last_latency:
    st.sidebar.divider()
    st.sidebar.subheader("레이턴시")
    lat = st.session_state.last_latency
    st.sidebar.write(f"재작성: {lat.get('rewrite_ms', 0)}ms")
    st.sidebar.write(f"검색:   {lat.get('retrieval_ms', 0)}ms")
    st.sidebar.write(f"생성:   {lat.get('generation_ms', 0)}ms")
    st.sidebar.write(f"총:     {lat.get('total_ms', 0)}ms")

if st.sidebar.button("대화 초기화"):
    st.session_state.messages     = []
    st.session_state.last_meta    = {}
    st.session_state.last_sources = []
    st.session_state.last_latency = {}
    st.rerun()

# 서비스 초기화 (서버 시작 시 1회)
@st.cache_resource
def load_service():
    return GenerationService()

try:
    with st.spinner("파이프라인 로딩 중..."):
        svc = load_service()
    st.sidebar.success("파이프라인 로드 완료")
except Exception as e:
    st.error(f"초기화 오류: {e}")
    st.stop()

# 통계
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("대화 턴", len([m for m in st.session_state.messages if m["role"] == "user"]))
with col2:
    st.metric("Top-K", top_k_display)
with col3:
    total_ms = st.session_state.last_latency.get("total_ms", 0)
    st.metric("마지막 응답 (ms)", total_ms if total_ms else "-")

# Tabs
tab1, tab2, tab3 = st.tabs(["Chat", "Retrieved Docs", "Analytics"])

# Chat 탭
with tab1:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("질문을 입력하세요"):
        logging.info(f"[USER QUESTION]: {prompt}")
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        history = GenerationService.format_history(st.session_state.messages[:-1])

        with st.chat_message("assistant"):
            try:
                answer_chunks = []
                sources_text  = ""
                meta_info     = {}

                with st.status("RAG 처리 중...", expanded=True) as status:
                    for chunk in svc.stream(prompt, history=history if history else None):
                        if chunk["type"] == "meta":
                            data = chunk["data"]
                            meta_info = data
                            st.write(f"검색 완료 - 필터: {data['filter']} | 재작성: {data['rewritten_query']}")
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
                st.session_state.last_meta    = meta_info
                st.session_state.last_sources = meta_info.get("sources", [])
                st.session_state.last_latency = {}

                logging.info("[SYSTEM ANSWER]: RAG 답변 생성 완료")

            except Exception as e:
                logging.error(f"[SYSTEM ERROR]: {e}")
                st.error(f"RAG 처리 오류: {e}")

# Retrieved Docs 탭
with tab2:
    st.subheader("검색된 문서 (출처)")
    sources = st.session_state.get("last_sources", [])
    if sources:
        for s in sources:
            label = f"[{s['rank']}] {s.get('agency','')} {s.get('year','')} - {s.get('project','')}"
            with st.expander(label):
                st.write(f"Score: {s.get('score', 0):.4f}")
    else:
        st.info("아직 검색된 문서가 없습니다.")

    st.divider()
    st.subheader("마지막 검색 메타 정보")
    meta = st.session_state.get("last_meta", {})
    if meta:
        st.json(meta)
    else:
        st.info("아직 검색 이력이 없습니다.")

# Analytics 탭
with tab3:
    st.subheader("대화 분석")
    messages  = st.session_state.messages
    user_msgs = [m for m in messages if m["role"] == "user"]
    asst_msgs = [m for m in messages if m["role"] == "assistant"]

    a1, a2, a3 = st.columns(3)
    with a1: st.metric("총 질문 수", len(user_msgs))
    with a2: st.metric("총 답변 수", len(asst_msgs))
    with a3:
        avg_len = round(sum(len(m["content"]) for m in asst_msgs) / len(asst_msgs)) if asst_msgs else 0
        st.metric("평균 답변 길이 (자)", avg_len)

    st.divider()
    st.subheader("답변 길이 추이")
    if asst_msgs:
        df = pd.DataFrame({
            "턴": list(range(1, len(asst_msgs) + 1)),
            "답변 길이": [len(m["content"]) for m in asst_msgs],
        })
        st.bar_chart(data=df, x="턴", y="답변 길이")
        st.dataframe(df)
    else:
        st.info("아직 분석할 데이터가 없습니다.")
