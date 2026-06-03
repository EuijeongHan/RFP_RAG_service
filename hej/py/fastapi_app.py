"""
입찰메이트 RAG — FastAPI 서빙 엔드포인트
==========================================
담당 : Generation / Retrieval 파트 (한의정)

[실행 방법]
    # GCP 서버 (KURE 임베딩 + Gemma 로컬 LLM)
    EMBED_SCENARIO=KURE_GEMMA BIDMATE_ENV=gcp uvicorn fastapi_app:app --host 0.0.0.0 --port 8000

    # GCP 서버 (SMALL 임베딩 + gpt-5.4-mini, 기본값)
    EMBED_SCENARIO=SMALL_OPENAI OPENAI_API_KEY=sk-... uvicorn fastapi_app:app --host 0.0.0.0 --port 8000

    # GCP 서버 (SMALL 임베딩 + gpt-4.1-mini)
    EMBED_SCENARIO=SMALL_OPENAI OPENAI_LLM_MODEL=gpt-4.1-mini OPENAI_API_KEY=sk-... uvicorn fastapi_app:app --host 0.0.0.0 --port 8000

    # OpenRouter 테스트 (Mac M2 로컬, GPU 불필요)
    EMBED_SCENARIO=KURE_QWEN OPENROUTER_API_KEY=sk-or-v1-... uvicorn fastapi_app:app --reload

[엔드포인트]
    POST /chat           : 일반 응답 (JSON 반환)
    POST /chat/stream    : 스트리밍 응답 (Server-Sent Events)
    POST /reset          : 세션 히스토리 초기화
    GET  /health         : 헬스 체크
    GET  /scenario       : 현재 시나리오 확인

[스트리밍 설계]
    Retrieval (Reranker 포함, ~900ms) 완료 후
    → LLM 첫 토큰 생성 즉시 클라이언트로 전송 (Server-Sent Events)
    → 사용자 체감 대기시간 ≈ Retrieval 시간 (LLM 생성 시간은 체감 안 됨)

[주의사항]
    - BidMateApp 인스턴스는 FastAPI 앱 생애주기 동안 단일 인스턴스 유지 (싱글턴)
    - 멀티유저 환경에서는 session_id 기반 히스토리 분리 필요 (현재는 단일 세션)
    - os.environ 기반 Query Router는 단일 프로세스·단일 유저 기준 (멀티스레드 주의)
"""

import asyncio
import json
import logging
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from serving_main import BidMateApp

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
_logger = logging.getLogger(__name__)

# ======================================================================
# FastAPI 앱 초기화
# ======================================================================
app = FastAPI(
    title       = '입찰메이트 RAG API',
    description = '공공 입찰 공고 전문 AI 어시스턴트',
    version     = '1.0.0',
)

# CORS 설정 (프론트엔드 개발 환경 허용)
# ⚠️ 프로덕션에서는 allow_origins 를 실제 도메인으로 제한할 것
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ['*'],
    allow_credentials = True,
    allow_methods     = ['*'],
    allow_headers     = ['*'],
)

# ======================================================================
# 싱글턴 BidMateApp (서버 기동 시 Retrieval + LLM 모두 Warm-up)
# ======================================================================
# ⚠️ 앱 인스턴스 생성 시 LocalHFGenerator 는 수십 초 소요
#    uvicorn --workers 1 권장 (멀티워커 시 GPU 메모리 공유 불가)
_bid_app: Optional[BidMateApp] = None


@app.on_event('startup')
async def startup():
    """서버 기동 시 모델 로드 (Warm-up)."""
    global _bid_app
    _logger.info('[FastAPI] BidMateApp 초기화 시작 (Warm-up)...')
    # blocking I/O (모델 로드)를 이벤트 루프 밖에서 실행
    _bid_app = await asyncio.get_event_loop().run_in_executor(None, BidMateApp)
    _logger.info('[FastAPI] ✅ BidMateApp Warm-up 완료')


@app.on_event('shutdown')
async def shutdown():
    """서버 종료 시 GPU 메모리 해제."""
    if _bid_app:
        _bid_app.release()
    _logger.info('[FastAPI] BidMateApp 해제 완료')


# ======================================================================
# 요청/응답 스키마
# ======================================================================
class ChatRequest(BaseModel):
    query      : str
    meta_filter: Optional[dict] = None  # {'agency': '한국가스공사', 'year': '2024'} 형태


class ChatResponse(BaseModel):
    answer     : str
    sources    : list
    sub_queries: list
    routed_to  : Optional[str] = None  # 디버깅용: 'chunks_all' | 'kh_v3_hybrid'


class ResetResponse(BaseModel):
    message: str


# ======================================================================
# 엔드포인트
# ======================================================================
@app.get('/health')
async def health():
    """헬스 체크. 로드 밸런서 또는 모니터링 도구에서 사용."""
    return {'status': 'ok', 'app_ready': _bid_app is not None}


@app.get('/scenario')
async def get_scenario():
    """현재 시나리오 및 모델 정보 반환."""
    if _bid_app is None:
        raise HTTPException(status_code=503, detail='앱 초기화 중입니다.')
    return {
        'scenario': _bid_app.scenario,
        'history_length': len(_bid_app.chat_history),
    }


@app.post('/chat', response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    일반 응답 엔드포인트 (전체 답변 완성 후 반환).

    ⚠️ 레이턴시: Retrieval(~900ms) + LLM 생성(~2~4s) = 총 3~5s 예상
       사용자 체감이 중요하다면 /chat/stream 사용 권장.
    """
    if _bid_app is None:
        raise HTTPException(status_code=503, detail='앱 초기화 중입니다.')

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _bid_app.chat(request.query, meta_filter=request.meta_filter),
        )
        return ChatResponse(
            answer      = result['answer'],
            sources     = result['sources'],
            sub_queries = result['sub_queries'],
            routed_to   = result.get('routed_to'),
        )
    except Exception as e:
        _logger.error(f'[FastAPI] /chat 오류: {e}')
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/chat/stream')
async def chat_stream(request: ChatRequest):
    """
    스트리밍 응답 엔드포인트 (Server-Sent Events).

    [스트리밍 흐름]
      1. Retrieval + Query Router (~900ms, 블로킹)
      2. LLM 첫 토큰 생성 즉시 클라이언트로 전송 시작
      3. 마지막에 출처 정보를 JSON 이벤트로 전송

    [프론트엔드 연동 예시 (JavaScript)]
      const evtSource = new EventSource('/chat/stream', {method: 'POST', body: JSON.stringify(...)})
      evtSource.onmessage = (e) => {
          const data = JSON.parse(e.data)
          if (data.type === 'token') appendText(data.content)
          if (data.type === 'sources') showSources(data.sources)
      }

    ⚠️ SSE 는 단방향 (서버→클라이언트) 이므로
       클라이언트가 스트림 도중 취소하면 서버는 감지 못할 수 있음.
       → CancelledError 처리로 조기 종료 방어.
    """
    if _bid_app is None:
        raise HTTPException(status_code=503, detail='앱 초기화 중입니다.')

    async def event_generator() -> AsyncIterator[str]:
        try:
            # ── Step 1. Retrieval (blocking → executor) ──────────────
            # Retrieval은 동기 코드이므로 이벤트 루프 블로킹 방지를 위해 executor 사용
            from serving_main import route_and_search
            from retrieval_interface_F import build_prompt

            retrieval_result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: route_and_search(
                    query       = request.query,
                    history     = _bid_app.chat_history,
                    meta_filter = request.meta_filter,
                ),
            )

            prompt_dict = build_prompt(
                query    = request.query,
                context  = retrieval_result['context'],
                scenario = _bid_app.scenario,
            )

            # ── Step 2. 스트리밍 생성 ────────────────────────────────
            full_answer = ''

            # generate_stream() 은 동기 Iterator → asyncio 이벤트 루프에서 실행
            # run_in_executor 로 감싸 async 환경에서 안전하게 소비
            loop = asyncio.get_event_loop()

            def _stream_sync():
                """동기 generate_stream 을 list 로 수집 후 반환."""
                return list(_bid_app.generator.generate_stream(
                    prompt_dict = prompt_dict,
                    history     = _bid_app.chat_history,
                ))

            # ⚠️ 실시간 스트리밍이 필요하면 asyncio.Queue 기반으로 교체 권장.
            #    현재는 단순성 우선으로 전체 생성 후 토큰별 yield 방식 사용.
            #    체감 차이: Retrieval 완료 후 LLM 토큰이 한꺼번에 나옴.
            #    진짜 실시간이 필요하면 아래 주석의 Queue 패턴 사용.
            tokens = await loop.run_in_executor(None, _stream_sync)

            for token in tokens:
                full_answer += token
                # SSE 포맷: "data: {...}\n\n"
                payload = json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)
                yield f'data: {payload}\n\n'

            # ── Step 3. 출처 정보 전송 ───────────────────────────────
            sources_payload = json.dumps({
                'type'       : 'sources',
                'sources'    : retrieval_result['sources'],
                'sub_queries': retrieval_result['sub_queries'],
                'routed_to'  : retrieval_result.get('routed_to'),
            }, ensure_ascii=False)
            yield f'data: {sources_payload}\n\n'

            # ── Step 4. 히스토리 업데이트 ────────────────────────────
            _bid_app.chat_history.append({'role': 'user',      'content': request.query})
            _bid_app.chat_history.append({'role': 'assistant', 'content': full_answer})
            
            
            # 현재
            # if len(_bid_app.chat_history) > _bid_app.__class__.__mro__[0].__dict__.get(
            #     '_MAX_HISTORY_MSGS', 20
            # ):
            #     _bid_app.chat_history = _bid_app.chat_history[-20:]
            # 수정
            from serving_main import _MAX_HISTORY_MSGS
            if len(_bid_app.chat_history) > _MAX_HISTORY_MSGS:
                _bid_app.chat_history = _bid_app.chat_history[-_MAX_HISTORY_MSGS:]

            # 스트림 종료 신호
            yield 'data: {"type": "done"}\n\n'

        except asyncio.CancelledError:
            # 클라이언트가 연결을 끊은 경우 — 정상 종료
            _logger.info('[FastAPI] 스트리밍 중 클라이언트 연결 해제')
        except Exception as e:
            _logger.error(f'[FastAPI] 스트리밍 오류: {e}')
            error_payload = json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)
            yield f'data: {error_payload}\n\n'

    return StreamingResponse(
        event_generator(),
        media_type = 'text/event-stream',
        headers    = {
            'Cache-Control'    : 'no-cache',
            'X-Accel-Buffering': 'no',  # nginx 버퍼링 비활성화 (실시간 전송 필수)
        },
    )


# ⚠️ 멀티유저 확장 시 세션 분리 방법:
#    1. session_id 를 요청 헤더에 포함
#    2. _sessions: dict[str, BidMateApp] 로 세션별 인스턴스 관리
#    3. 히스토리는 Redis (key: session_id) 에 저장
#    현재는 단일 세션 기준 구현 (팀 내부 데모/테스트 용도)
@app.post('/reset', response_model=ResetResponse)
async def reset():
    """세션 히스토리 초기화. 새 주제 시작 시 호출."""
    if _bid_app is None:
        raise HTTPException(status_code=503, detail='앱 초기화 중입니다.')
    _bid_app.reset()
    return ResetResponse(message='세션 히스토리가 초기화되었습니다.')
