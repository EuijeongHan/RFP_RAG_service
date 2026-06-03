import os
import sys
import json
import uuid
import logging
import datetime
import numpy as np
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import LOG_PATH, EMBED_MODEL_ID
from service import GenerationService

# 로그 설정
logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8",
)

# 세션 저장 경로
SESSION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sessions")
os.makedirs(SESSION_DIR, exist_ok=True)

OPENAI_EVAL_ENABLED = False  # OpenAI 키 입력 시 True로 스위칭


# 세션 파일 관리
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


# 평가 지표 계산 (임베딩 기반)
@st.cache_resource
def get_eval_embed_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(
        EMBED_MODEL_ID,
        cache_folder="/mnt/gukrul/hf_cache/hub",
    )


def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def compute_metrics(question, answer, context, sources, openai_key=None):
    if openai_key:
        return compute_metrics_llm(question, answer, context, openai_key)
    return compute_metrics_embedding(question, answer, context, sources)


def compute_metrics_embedding(question, answer, context, sources):
    try:
        model = get_eval_embed_model()
        q_emb  = model.encode([question], normalize_embeddings=True)[0]
        a_emb  = model.encode([answer],   normalize_embeddings=True)[0]
        c_emb  = model.encode([context],  normalize_embeddings=True)[0]

        faithfulness     = round(cosine_sim(a_emb, c_emb), 3)
        answer_relevancy = round(cosine_sim(a_emb, q_emb), 3)
        context_recall   = round(cosine_sim(c_emb, q_emb), 3)

        # Context Precision: Reranker 점수 상위 비율
        scores = [s.get("score", 0) for s in sources]
        context_precision = round(float(np.mean(scores)) if scores else 0.0, 3)

        return {
            "Faithfulness"     : faithfulness,
            "Answer Relevancy" : answer_relevancy,
            "Context Precision": context_precision,
            "Context Recall"   : context_recall,
        }
    except Exception as e:
        return {"error": str(e)}


def compute_metrics_llm(question, answer, context, openai_key):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)

        prompt = f"""다음 RAG 시스템의 출력을 평가하세요. 각 지표를 0.0~1.0 사이 점수로 평가하고 JSON으로만 응답하세요.

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


# Streamlit 설정
st.set_page_config(
    page_title="국룰:RFP 맥잡기",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS - 짙은 파란색 테마
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
    --success      : #10b981;
    --warning      : #f59e0b;
    --danger       : #ef4444;
}

html, body, .stApp {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Noto Sans KR', sans-serif !important;
}

/* 사이드바 */
section[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* 타이틀 */
.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #cbd5e1;  
    letter-spacing: -0.5px;
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

/* 세션 아이템 */
.session-item {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 6px;
    cursor: pointer;
    transition: all 0.2s;
}
.session-item:hover {
    background: var(--bg-hover);
    border-color: var(--accent);
}
.session-title {
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.session-date {
    font-size: 0.7rem;
    color: var(--text-muted);
    margin-top: 2px;
}

/* 메트릭 카드 */
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
    color: var(--text-primary);
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

/* 채팅 */
.stChatMessage {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

/* 버튼 */
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

/* 탭 */
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

/* 구분선 */
hr {
    border-color: var(--border) !important;
}

/* 섹션 헤더 */
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

/* expander */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
}
</style>
""", unsafe_allow_html=True)


# Session State 초기화
if "messages"      not in st.session_state: st.session_state.messages      = []
if "last_meta"     not in st.session_state: st.session_state.last_meta     = {}
if "last_sources"  not in st.session_state: st.session_state.last_sources  = []
if "last_metrics"  not in st.session_state: st.session_state.last_metrics  = {}
if "last_context"  not in st.session_state: st.session_state.last_context  = ""
if "last_question" not in st.session_state: st.session_state.last_question = ""
if "last_answer"   not in st.session_state: st.session_state.last_answer   = ""
if "session_id"    not in st.session_state: st.session_state.session_id    = str(uuid.uuid4())
if "openai_key"    not in st.session_state: st.session_state.openai_key    = ""


# 왼쪽 사이드바 - 세션 목록
with st.sidebar:
    st.markdown('<div class="main-title">국룰:RFP 맥잡기</div>', unsafe_allow_html=True)

    if st.button("+ 새 대화", use_container_width=True):
        if st.session_state.messages:
            save_session(st.session_state.session_id, st.session_state.messages)
        st.session_state.messages      = []
        st.session_state.last_meta     = {}
        st.session_state.last_sources  = []
        st.session_state.last_metrics  = {}
        st.session_state.last_context  = ""
        st.session_state.last_question = ""
        st.session_state.last_answer   = ""
        st.session_state.session_id    = str(uuid.uuid4())
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

    st.markdown('<div class="section-header" style="margin-top:16px;">평가 설정</div>', unsafe_allow_html=True)
    openai_key_input = st.text_input(
        "OpenAI API Key (선택)",
        value=st.session_state.openai_key,
        type="password",
        placeholder="sk-... (LLM 기반 평가 활성화)",
    )
    if openai_key_input != st.session_state.openai_key:
        st.session_state.openai_key = openai_key_input

    if st.session_state.openai_key:
        st.markdown('<p style="color:#10b981;font-size:0.75rem;">LLM 평가 활성화됨</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:var(--text-muted);font-size:0.75rem;">임베딩 기반 평가 사용 중</p>', unsafe_allow_html=True)


# 서비스 초기화
@st.cache_resource
def load_service():
    return GenerationService()

try:
    with st.spinner("파이프라인 로딩 중..."):
        svc = load_service()
except Exception as e:
    st.error(f"초기화 오류: {e}")
    st.stop()


# 메인 타이틀
st.markdown('<div class="main-title">국룰:RFP 맥잡기</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-caption">Gemma4 E4B · LoRA · ChromaDB · BM25 · KURE-v1</div>', unsafe_allow_html=True)

# 메인 레이아웃: 채팅(좌) + 평가 지표(우)
chat_col, metric_col = st.columns([3, 1])

# 오른쪽 - 평가 지표 패널
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


# 왼쪽 - 채팅
with chat_col:
    tab1, tab2, tab3 = st.tabs(["Chat", "Retrieved Docs", "Analytics"])

    with tab1:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("RFP에 대해 질문하세요"):
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
                    context_text  = ""

                    with st.status("RAG 처리 중...", expanded=True) as status:
                        for chunk in svc.stream(prompt, history=history if history else None):
                            if chunk["type"] == "meta":
                                data = chunk["data"]
                                meta_info    = data
                                context_text = data.get("context", "")
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
                    st.session_state.last_meta     = meta_info
                    st.session_state.last_sources  = meta_info.get("sources", [])
                    st.session_state.last_context  = context_text
                    st.session_state.last_question = prompt
                    st.session_state.last_answer   = answer_full

                    # 평가 지표 계산
                    if context_text:
                        metrics = compute_metrics(
                            question  = prompt,
                            answer    = answer_full,
                            context   = context_text,
                            sources   = meta_info.get("sources", []),
                            openai_key= st.session_state.openai_key or None,
                        )
                        st.session_state.last_metrics = metrics

                    # 세션 자동 저장
                    save_session(st.session_state.session_id, st.session_state.messages)

                    logging.info("[SYSTEM ANSWER]: RAG 답변 생성 완료")
                    st.rerun()

                except Exception as e:
                    logging.error(f"[SYSTEM ERROR]: {e}")
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
