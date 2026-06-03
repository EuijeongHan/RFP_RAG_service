import os
import tempfile
import logging
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# =========================================================
# 환경 변수 및 인프라 설정
# =========================================================
load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# ---- [접근 로그 인프라 설정] ----
logging.basicConfig(
    filename='../app_user_access.log',  # Gunho/ 폴더 바깥 메인 공간에 저장
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    encoding='utf-8'
)

# LangChain 관련 임포트
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

# =========================================================
# Streamlit 기본 설정
# =========================================================
st.set_page_config(
    page_title="2Team RAG Dashboard",
    page_icon="📚",
    layout="wide"
)

st.title("2Team RAG Dashboard")
st.caption("Gemini + LangChain + FAISS")

# =========================================================
# Session State 및 초기 접속 로깅
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "documents" not in st.session_state:
    st.session_state.documents = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

# 새로고침할 때마다 중복으로 찍히지 않도록 최초 진입 시에만 접속 로그 기록
if "logged_in" not in st.session_state:
    logging.info("📢 새로운 유저가 웹 페이지에 접속했습니다.")
    st.session_state.logged_in = True

# API 키 검증 후 프로세스 중단 처리
if not GOOGLE_API_KEY:
    st.error("GEMINI_API_KEY가 없습니다.")
    st.stop()

# =========================================================
# Sidebar
# =========================================================
st.sidebar.header("RAG 설정")

uploaded_files = st.sidebar.file_uploader(
    "PDF 업로드",
    type=["pdf"],
    accept_multiple_files=True
)

model_name = st.sidebar.selectbox(
    "Gemini 모델",
    [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ]
)

chunk_size = st.sidebar.slider("Chunk Size", 200, 2000, 1000, step=100)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 500, 100, step=50)
top_k = st.sidebar.slider("Top-K", 1, 10, 3)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, step=0.1)

# =========================================================
# Gemini LLM 초기화
# =========================================================
llm = ChatGoogleGenerativeAI(
    model=model_name,
    google_api_key=GOOGLE_API_KEY,
    temperature=temperature,
)

# =========================================================
# VectorStore 생성 함수
# =========================================================
@st.cache_resource
def build_vectorstore(uploaded_files, chunk_size, chunk_overlap):
    documents = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split_docs = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )

    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore, documents, split_docs

# =========================================================
# 문서 처리 실행
# =========================================================
if uploaded_files:
    try:
        with st.spinner("문서 임베딩 중..."):
            vectorstore, documents, split_docs = build_vectorstore(
                uploaded_files,
                chunk_size,
                chunk_overlap
            )
            st.session_state.vectorstore = vectorstore
            st.session_state.documents = documents
            st.session_state.chunks = split_docs

        st.sidebar.success("문서 처리 완료")
    except Exception as e:
        st.error(f"문서 처리 오류: {e}")

# =========================================================
# 대시보드 상단 통계 대시보드
# =========================================================
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("문서 수", len(st.session_state.documents))
with col2:
    st.metric("Chunk 수", len(st.session_state.chunks))
with col3:
    st.metric("Top-K", top_k)

# =========================================================
# Retriever 정의
# =========================================================
def get_retriever():
    vectorstore = st.session_state.vectorstore
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    return retriever

# =========================================================
# 기존 대화 내역 렌더링
# =========================================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =========================================================
# 사용자 입력 처리 및 RAG 파이프라인
# =========================================================
if prompt := st.chat_input("질문을 입력하세요"):

    if st.session_state.vectorstore is None:
        st.warning("먼저 PDF를 업로드하세요.")
        st.stop()

    # 1. 유저 메시지 저장 및 즉시 렌더링
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    retriever = get_retriever()

    # 2. 어시스턴트 답변 생성 영역
    with st.chat_message("assistant"):
        try:
            with st.status("RAG 처리 중...", expanded=True) as status:
                # 문서 검색 단계
                st.write("1. 문서 검색 중...")
                docs = retriever.invoke(prompt)

                # Context 조립 단계
                st.write("2. Context 생성 중...")
                context = "\n\n".join([doc.page_content for doc in docs])

                # LLM 질의 단계
                st.write("3. Gemini 응답 생성 중...")
                rag_prompt = f"""당신은 문서 기반 AI 어시스턴트입니다.
반드시 아래 Context만 기반으로 답변하세요.

[Context]
{context}

[Question]
{prompt}"""

                response = llm.invoke(rag_prompt)
                answer = response.content

                status.update(label="완료", state="complete")

            # 최종 답변 화면 출력 및 세션 상태 저장
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

            # ---- [답변 생성 완료 후 로그 기록 실행] ----
            logging.info(f"💬 [USER QUESTION]: {prompt}")
            logging.info(f"🤖 [SYSTEM ANSWER]: RAG 답변 생성 및 응답 완료 (포트: 8443)")

            # =================================================
            # 디버그 서브 정보 출력 (검색된 문서 디테일)
            # =================================================
            st.divider()
            st.subheader("검색된 문서")
            for i, doc in enumerate(docs):
                with st.expander(f"문서 {i+1}"):
                    st.write(doc.page_content[:3000])
                    if "source" in doc.metadata:
                        st.info(f"출처: {doc.metadata['source']}")

            # 전체 사용 컨텍스트 임시 확인 스크립트
            with st.expander("사용된 전체 Context"):
                st.code(context[:10000])

            # RAG 더미 평가지표 시각화
            st.subheader("RAG 평가")
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Faithfulness", "0.91")
            with metric_col2:
                st.metric("Answer Relevancy", "0.88")
            with metric_col3:
                st.metric("Context Recall", "0.85")

            # 검색된 덩어리(Chunk)의 텍스트 길이 시각화 차트
            chunk_lengths = [len(doc.page_content) for doc in docs]
            if chunk_lengths:
                df = pd.DataFrame({
                    "Chunk": list(range(1, len(chunk_lengths) + 1)),
                    "Length": chunk_lengths
                })
                st.subheader("검색 Chunk 길이")
                st.bar_chart(data=df, x="Chunk", y="Length")

        except Exception as e:
            st.error(f"RAG 처리 오류: {e}")