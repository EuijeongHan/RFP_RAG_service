import os
import sys
import json
import logging
import asyncio
import numpy as np
from typing import AsyncGenerator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import LOG_PATH, EMBED_MODEL_ID
from service import GenerationService

logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="국룰:RFP 맥잡기 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

svc: GenerationService = None
eval_embed_model = None


@app.on_event("startup")
async def startup():
    global svc, eval_embed_model
    logger.info("GenerationService 초기화 시작")
    svc = GenerationService()
    logger.info("GenerationService 초기화 완료")

    # 평가용 임베딩 모델 로드
    logger.info("평가용 임베딩 모델 로드 시작")
    from sentence_transformers import SentenceTransformer
    eval_embed_model = SentenceTransformer(
        EMBED_MODEL_ID,
        cache_folder="/mnt/gukrul/hf_cache/hub",
        local_files_only=True,
    )
    logger.info("평가용 임베딩 모델 로드 완료")


# 요청 모델
class LLMConfig(BaseModel):
    provider: str = "local"
    api_key : str = ""
    model   : str = ""

class ChatRequest(BaseModel):
    query     : str
    history   : list = []
    llm_config: LLMConfig = LLMConfig()

class AskRequest(BaseModel):
    query     : str
    history   : list = []
    llm_config: LLMConfig = LLMConfig()


class MetricsRequest(BaseModel):
    question: str
    answer  : str
    context : str
    sources : list = []


# 스트리밍
async def stream_generator(query: str, history: list, llm_config: dict = None) -> AsyncGenerator[str, None]:
    import queue, threading
    q = queue.Queue()
    def run_stream():
        try:
            if llm_config:
                svc.set_llm_config(**llm_config)
            for chunk in svc.stream(query, history=history if history else None):
                q.put(chunk)
        except Exception as e:
            q.put({"type": "chunk", "data": f"오류: {e}"})
        finally:
            q.put(None)
    threading.Thread(target=run_stream, daemon=True).start()
    while True:
        chunk = await asyncio.get_event_loop().run_in_executor(None, q.get)
        if chunk is None:
            break
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# 엔드포인트
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    if not svc:
        raise HTTPException(status_code=503, detail="서비스 초기화 중입니다.")
    logging.info(f"[API][STREAM] query={req.query[:50]}")
    cfg = {"provider": req.llm_config.provider, "api_key": req.llm_config.api_key, "model": req.llm_config.model}
    return StreamingResponse(
        stream_generator(req.query, req.history, llm_config=cfg),
        media_type="text/event-stream",
    )


@app.post("/chat/ask")
async def chat_ask(req: AskRequest):
    if not svc:
        raise HTTPException(status_code=503, detail="서비스 초기화 중입니다.")
    logging.info(f"[API][ASK] query={req.query[:50]}")
    loop = asyncio.get_event_loop()
    cfg = {"provider": req.llm_config.provider, "api_key": req.llm_config.api_key, "model": req.llm_config.model}
    result = await loop.run_in_executor(
        None, lambda: svc.ask(req.query, history=req.history if req.history else None, llm_config=cfg)
    )
    return {
        "answer"          : result["answer"],
        "sources"         : result["sources"],
        "rewritten_query" : result["rewritten_query"],
        "meta_filter"     : result["meta_filter"],
        "sub_queries"     : result["sub_queries"],
        "latency_ms"      : result["latency_ms"],
    }


@app.post("/metrics")
async def compute_metrics(req: MetricsRequest):
    if eval_embed_model is None:
        raise HTTPException(status_code=503, detail="평가 모델 초기화 중입니다.")
    try:
        loop = asyncio.get_event_loop()

        def calc():
            q_emb = eval_embed_model.encode([req.question], normalize_embeddings=True)[0]
            a_emb = eval_embed_model.encode([req.answer],   normalize_embeddings=True)[0]
            c_emb = eval_embed_model.encode([req.context],  normalize_embeddings=True)[0]

            faithfulness      = round(cosine_sim(a_emb, c_emb), 3)
            answer_relevancy  = round(cosine_sim(a_emb, q_emb), 3)
            context_recall    = round(cosine_sim(c_emb, q_emb), 3)
            scores            = [s.get("score", 0) for s in req.sources]
            context_precision = round(float(np.mean(scores)) if scores else 0.0, 3)

            return {
                "Faithfulness"     : faithfulness,
                "Answer Relevancy" : answer_relevancy,
                "Context Precision": context_precision,
                "Context Recall"   : context_recall,
            }

        result = await loop.run_in_executor(None, calc)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {
        "status"        : "ok",
        "service_ready" : svc is not None,
        "eval_ready"    : eval_embed_model is not None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=2026)
