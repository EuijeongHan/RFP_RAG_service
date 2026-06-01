import os
import sys
import json
import uuid
import logging
import datetime
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import LOG_PATH
from api_client import stream_chat, compute_metrics, is_server_ready
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8",
)

SESSION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sessions")
os.makedirs(SESSION_DIR, exist_ok=True)


def list_sessions():
    files = sorted(
        [f for f in os.listdir(SESSION_DIR) if f.endswith(".json")],
        key=lambda f: os.path.getmtime(os.path.join(SESSION_DIR, f)),
        reverse=True,
    )
    sessions = []
    for f in files:
        path = os.path.join(SESSION_DIR, f)
        try:
            with open(path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            sessions.append({
                "id"      : f.replace(".json", ""),
                "title"   : data.get("title", "제목 없음"),
                "created" : data.get("created", ""),
                "messages": data.get("messages", []),
            })
        except Exception:
            continue
    return sessions


def save_session(session_id, messages):
    path = os.path.join(SESSION_DIR, f"{session_id}.json")
    title = "새 대화"
    for m in messages:
        if m["role"] == "user":
            title = m["content"][:30] + ("..." if len(m["content"]) > 30 else "")
            break
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "title"   : title,
            "created" : datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "messages": messages,
        }, f, ensure_ascii=False, indent=2)


def delete_session(session_id):
    path = os.path.join(SESSION_DIR, f"{session_id}.json")
    if os.path.exists(path):
        os.remove(path)


def compute_metrics_llm(question, answer, context, openai_key):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        prompt = f"""다음 RAG 시스템의 출력을 평가하세요. 각 지표를 0.0~1.0 점수로 평가하고 JSON으로만 응답하세요.

질문: {question}
답변: {answer[:500]}
컨텍스트: {context[:500]}

JSON 형식:
{{"Faithfulness": 0.0, "Answer Relevancy": 0.0, "Context Precision": 0.0, "Context Recall": 0.0}}"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}


st.set_page_config(
    page_title="국룰:RFP 맥잡기",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&family=Space+Mono:wght@400;700&display=swap');

:root {
    --bg-primary   : #0a0f1e;
    --bg-secondary : #0d1526;
    --bg-card      : #111d35;
    --bg-hover     : #162040;
    --accent       : #2563eb;
    --accent-light : #3b82f6;
    --accent-glow  : #1d4ed8;
    --text-primary : #e2e8f0;
    --text-muted   : #94a3b8;
    --border       : #1e3a5f;
}

html, body, .stApp {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Noto Sans KR', sans-serif !important;
}

section[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #cbd5e1;
    border-bottom: 2px solid var(--accent);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}
.sub-caption {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: -0.8rem;
    margin-bottom: 1rem;
    font-family: 'Space Mono', monospace;
}

.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 8px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: var(--accent);
    border-radius: 3px 0 0 3px;
}
.metric-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-family: 'Space Mono', monospace;
}
.metric-value {
    font-size: 1.4rem;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
}
.metric-bar {
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    margin-top: 6px;
    overflow: hidden;
}
.metric-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.6s ease;
}

.stChatMessage {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

.stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Noto Sans KR', sans-serif !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: var(--accent-glow) !important;
    box-shadow: 0 0 12px rgba(37, 99, 235, 0.4) !important;
}

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

.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary) !important;
    border-bottom: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-muted) !important;
    font-family: 'Noto Sans KR', sans-serif !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent-light) !important;
    border-bottom: 2px solid var(--accent-light) !important;
}

hr { border-color: var(--border) !important; }

.section-header {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: var(--text-muted);
    font-family: 'Space Mono', monospace;
    padding: 8px 0 6px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 10px;
}

div[data-baseweb="base-input"]:focus-within {
    border-color: white !important;
    outline: none !important;
    box-shadow: 0 0 0 2px white !important;
}
div[data-baseweb="textarea"]:focus-within {
    border-color: white !important;
    outline: none !important;
    box-shadow: 0 0 0 2px white !important;
}
</style>
""", unsafe_allow_html=True)

defaults = {
    "messages"     : [],
    "last_meta"    : {},
    "last_sources" : [],
    "last_metrics" : {},
    "last_context" : "",
    "last_question": "",
    "last_answer"  : "",
    "session_id"   : str(uuid.uuid4()),
    "openai_key"   : os.getenv("OPENAI_API_KEY", ""),
    "gemini_key"   : os.getenv("GEMINI_API_KEY", ""),
    "openrouter_key": os.getenv("OPENROUTER_API_KEY", ""),
    "llm_provider" : "local",
    "llm_model"    : "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if not is_server_ready():
    st.error("API 서버(포트 2026)가 준비되지 않았습니다. fastapi_server.py를 먼저 실행해주세요.")
    st.stop()

with st.sidebar:
    st.markdown('<div class="main-title">국룰:RFP 맥잡기</div>', unsafe_allow_html=True)

    if st.button("+ 새 대화", use_container_width=True):
        if st.session_state.messages:
            save_session(st.session_state.session_id, st.session_state.messages)
        for k, v in defaults.items():
            st.session_state[k] = v
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.markdown('<div class="section-header">대화 기록</div>', unsafe_allow_html=True)
    sessions = list_sessions()
    if sessions:
        for s in sessions[:20]:
            col_s, col_d = st.columns([5, 1])
            with col_s:
                if st.button(
                    f"{s['title']}\n{s['created']}",
                    key=f"ses_{s['id']}",
                    use_container_width=True,
                ):
                    if st.session_state.messages:
                        save_session(st.session_state.session_id, st.session_state.messages)
                    st.session_state.messages   = s["messages"]
                    st.session_state.session_id = s["id"]
                    st.rerun()
            with col_d:
                if st.button("x", key=f"del_{s['id']}"):
                    delete_session(s["id"])
                    st.rerun()
    else:
        st.markdown('<p style="color:var(--text-muted);font-size:0.8rem;">아직 대화 기록이 없습니다.</p>', unsafe_allow_html=True)

    st.markdown('<div class="section-header" style="margin-top:16px;">LLM 설정</div>', unsafe_allow_html=True)

    provider_options = {
        "local"      : "로컬 (Phi-4-mini)",
        "openai"     : "OpenAI GPT-4o-mini",
        "gemini"     : "Gemini 2.5 Flash",
        "openrouter" : "OpenRouter",
    }
    selected_provider = st.selectbox(
        "LLM 선택",
        options=list(provider_options.keys()),
        format_func=lambda x: provider_options[x],
        index=list(provider_options.keys()).index(st.session_state.llm_provider),
    )
    if selected_provider != st.session_state.llm_provider:
        st.session_state.llm_provider = selected_provider

    if selected_provider == "openrouter":
        model_input = st.text_input(
            "OpenRouter 모델 (선택)",
            value=st.session_state.llm_model,
            placeholder="ex) anthropic/claude-3-haiku",
        )
        if model_input != st.session_state.llm_model:
            st.session_state.llm_model = model_input

    if selected_provider != "local":
        key_map = {"openai": "openai_key", "gemini": "gemini_key", "openrouter": "openrouter_key"}
        key_attr = key_map[selected_provider]
        if st.session_state[key_attr]:
            st.markdown('<p style="color:#10b981;font-size:0.75rem;">외부 LLM 활성화됨</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:#ef4444;font-size:0.75rem;">API Key 미설정 (.env 확인)</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:var(--text-muted);font-size:0.75rem;">로컬 모델 사용 중</p>', unsafe_allow_html=True)

    st.markdown('<div class="section-header" style="margin-top:16px;">평가 설정</div>', unsafe_allow_html=True)
    if st.session_state.openai_key:
        st.markdown('<p style="color:#10b981;font-size:0.75rem;">LLM 평가 활성화됨</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:var(--text-muted);font-size:0.75rem;">임베딩 기반 평가 사용 중</p>', unsafe_allow_html=True)


st.markdown('<div class="main-title">국룰:RFP 맥잡기</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-caption">{provider_options.get(st.session_state.llm_provider, "로컬")} · ChromaDB · BM25 · KURE-v1</div>', unsafe_allow_html=True)

chat_col, metric_col = st.columns([3, 1])

with metric_col:
    st.markdown('<div class="section-header">답변 품질 평가</div>', unsafe_allow_html=True)
    metrics = st.session_state.last_metrics

    def metric_color(val):
        if val >= 0.75: return "#10b981"
        if val >= 0.5:  return "#f59e0b"
        return "#ef4444"

    if metrics and "error" not in metrics:
        for label, val in metrics.items():
            color = metric_color(val)
            pct   = int(val * 100)
            st.markdown(f"""
<div class="metric-card">
  <div class="metric-label">{label}</div>
  <div class="metric-value" style="color:{color}">{val:.2f}</div>
  <div class="metric-bar">
    <div class="metric-fill" style="width:{pct}%;background:{color}"></div>
  </div>
</div>""", unsafe_allow_html=True)
    elif metrics and "error" in metrics:
        st.warning(f"평가 오류: {metrics['error']}")
    else:
        for label in ["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall"]:
            st.markdown(f"""
<div class="metric-card">
  <div class="metric-label">{label}</div>
  <div class="metric-value" style="color:var(--text-muted)">-</div>
  <div class="metric-bar"><div class="metric-fill" style="width:0%;background:var(--border)"></div></div>
</div>""", unsafe_allow_html=True)

    if st.session_state.last_meta:
        st.markdown('<div class="section-header" style="margin-top:16px;">검색 정보</div>', unsafe_allow_html=True)
        meta = st.session_state.last_meta
        st.markdown(f'<p style="font-size:0.75rem;color:var(--text-muted)">필터: {meta.get("filter", {})}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:0.75rem;color:var(--text-muted)">재작성: {meta.get("rewritten_query", "")}</p>', unsafe_allow_html=True)
        sub_q = meta.get("sub_queries", [])
        if len(sub_q) > 1:
            st.markdown(f'<p style="font-size:0.75rem;color:var(--text-muted)">서브쿼리: {sub_q}</p>', unsafe_allow_html=True)


with chat_col:
    tab1, tab2, tab3 = st.tabs(["Chat", "Retrieved Docs", "Analytics"])

    with tab1:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("질문하세요"):
            logging.info(f"[ADMIN][USER]: {prompt}")
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
                    meta_info     = {}
                    context_text  = ""

                    step_icons = {1: "1. 쿼리 재작성", 2: "2. 문서 검색", 3: "3. Reranker", 4: "4. 답변 생성"}
                    key_map  = {"openai": "openai_key", "gemini": "gemini_key", "openrouter": "openrouter_key", "local": None}
                    key_attr = key_map.get(st.session_state.llm_provider)
                    api_key  = st.session_state[key_attr] if key_attr else ""
                    llm_cfg  = {
                        "provider": st.session_state.llm_provider,
                        "api_key" : api_key,
                        "model"   : st.session_state.get("llm_model", ""),
                    }
                    with st.status("RAG 처리 중...", expanded=True) as status:
                        current_step = 0
                        for chunk in stream_chat(prompt, history=history if history else None, llm_config=llm_cfg):
                            if chunk["type"] == "progress":
                                step    = chunk["data"]["step"]
                                message = chunk["data"]["message"]
                                if step != current_step:
                                    current_step = step
                                    st.write(f"**{step_icons.get(step, str(step))}**")
                                st.caption(message)
                            elif chunk["type"] == "meta":
                                meta_info    = chunk["data"]
                                context_text = chunk["data"].get("context", "")
                            elif chunk["type"] == "chunk":
                                answer_chunks.append(chunk["data"])
                            elif chunk["type"] == "done":
                                sources_text = chunk["data"]["sources_text"]
                        status.update(label="처리 완료", state="complete")

                    answer_full = "".join(answer_chunks)
                    if sources_text and "[출처]" not in answer_full:
                        answer_full += "\n" + sources_text

                    st.markdown(answer_full)

                    st.session_state.messages.append({"role": "assistant", "content": answer_full})
                    st.session_state.last_meta     = meta_info
                    st.session_state.last_sources  = meta_info.get("sources", [])
                    st.session_state.last_context  = context_text
                    st.session_state.last_question = prompt
                    st.session_state.last_answer   = answer_full

                    # 평가 지표 - API 서버에서 계산
                    if context_text:
                        if st.session_state.openai_key:
                            metrics = compute_metrics_llm(
                                prompt, answer_full, context_text,
                                st.session_state.openai_key
                            )
                        else:
                            metrics = compute_metrics(
                                question=prompt,
                                answer  =answer_full,
                                context =context_text,
                                sources =meta_info.get("sources", []),
                            )
                        st.session_state.last_metrics = metrics

                    save_session(st.session_state.session_id, st.session_state.messages)
                    logging.info("[ADMIN][ANSWER]: 완료")
                    st.rerun()

                except Exception as e:
                    logging.error(f"[ADMIN][ERROR]: {e}")
                    st.error(f"RAG 처리 오류: {e}")

    with tab2:
        st.markdown('<div class="section-header">검색된 문서</div>', unsafe_allow_html=True)
        sources = st.session_state.get("last_sources", [])
        if sources:
            for s in sources:
                label = f"[{s['rank']}] {s.get('agency','')} {s.get('year','')} - {s.get('project','')}"
                with st.expander(label):
                    st.write(f"Score: {s.get('score', 0):.4f}")
        else:
            st.info("아직 검색된 문서가 없습니다.")

        if st.session_state.last_context:
            st.markdown('<div class="section-header" style="margin-top:16px;">사용된 컨텍스트</div>', unsafe_allow_html=True)
            st.code(st.session_state.last_context[:5000])

    with tab3:
        st.markdown('<div class="section-header">대화 분석</div>', unsafe_allow_html=True)
        messages  = st.session_state.messages
        user_msgs = [m for m in messages if m["role"] == "user"]
        asst_msgs = [m for m in messages if m["role"] == "assistant"]

        a1, a2, a3 = st.columns(3)
        with a1: st.metric("총 질문 수", len(user_msgs))
        with a2: st.metric("총 답변 수", len(asst_msgs))
        with a3:
            avg_len = round(sum(len(m["content"]) for m in asst_msgs) / len(asst_msgs)) if asst_msgs else 0
            st.metric("평균 답변 길이 (자)", avg_len)

        if asst_msgs:
            st.divider()
            df = pd.DataFrame({
                "턴": list(range(1, len(asst_msgs) + 1)),
                "답변 길이": [len(m["content"]) for m in asst_msgs],
            })
            st.bar_chart(data=df, x="턴", y="답변 길이")
