import time
import logging
from threading import Thread
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel

from config import (
    BASE_MODEL_ID, ADAPTER_PATH,
    MAX_TOKENS_REWRITE, MAX_TOKENS_GENERATE,
    LLM_MODEL,
)

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 프롬프트 템플릿
BASE_SYSTEM_PROMPT = """당신은 공공 입찰 RFP(제안요청서) 전문 분석 어시스턴트 입찰메이트입니다.

[핵심 원칙]
1. 반드시 아래 [검색된 문서]의 내용만을 근거로 답변하세요.
2. 문서에 없는 내용은 절대 추측하거나 생성하지 마세요.
3. 금액, 날짜, 수량 등 수치는 원문 그대로 인용하세요.
4. 답변 마지막에 반드시 [출처] 섹션을 포함하세요.
5. 확인할 수 없는 내용은 "제공된 문서에서 확인할 수 없습니다"라고 명시하세요.
"""

TYPE_INSTRUCTIONS = {
    "single": """
[답변 형식]
- 질문에 직접 답변하는 1~3문장으로 시작합니다.
- 필요 시 세부 내용을 bullet point로 정리합니다.
- 수치(예산, 기간, 인원 등)는 굵게 표시합니다.
""",
    "compare": """
[답변 형식 - 복수 기관/사업 비교]
- 기관별로 섹션을 나눠 정리합니다. (### 기관명)
- 마지막에 비교 요약 표를 작성합니다.
- 문서에 없는 기관의 정보는 "문서 없음"으로 표기합니다.
- 수치는 반드시 원문 인용합니다.
""",
    "followup": """
[답변 형식 - 후속 질문]
- 이전 대화 맥락을 간단히 재확인 후 답변합니다.
- 이전 답변과 연결되는 내용임을 명시합니다.
""",
}

REWRITE_SYSTEM_PROMPT = """당신은 공공 입찰 RFP 문서 검색 전문가입니다.
사용자의 질문을 벡터 DB + BM25 하이브리드 검색에 최적화된 쿼리로 재작성하세요.

[재작성 규칙]
1. 기관명은 공식 전체 명칭으로 확장합니다. (예: 가스공사 -> 한국가스공사)
2. 구어체, 약어를 공문서 표준 용어로 변환합니다. (예: 얼마야 -> 사업 예산 규모)
3. 핵심 명사 키워드를 공백으로 연결합니다. (조사, 어미 제거)
4. 연도가 언급된 경우 반드시 포함합니다.
5. 대화 히스토리가 있으면 맥락을 반영해 독립적인 쿼리로 재작성합니다.
6. 재작성된 쿼리만 출력합니다. (설명 없이)
"""

USER_PROMPT_TEMPLATE = """[검색된 문서]
{context}

---

[질문]
{query}

[참고사항]
- 검색 필터: {meta_filter}
- 재작성된 검색 쿼리: {rewritten_query}
"""


def format_sources(sources: list) -> str:
    lines = ["\n[출처]"]
    for s in sources:
        agency  = s.get("agency", "미상")
        year    = s.get("year", "")
        project = s.get("project", "")
        score   = s.get("score", 0)
        line = f"  [{s['rank']}] {agency}"
        if year:    line += f" {year}"
        if project: line += f" - {project}"
        line += f" (score: {score:.4f})"
        lines.append(line)
    return "\n".join(lines)


def build_prompt(query, rewritten_query, retrieval_result, history=None, query_type="single"):
    sub_queries = retrieval_result.get("sub_queries", [])
    if len(sub_queries) > 1:
        query_type = "compare"
    elif history and len(history) > 0:
        query_type = "followup"

    system_prompt = BASE_SYSTEM_PROMPT + TYPE_INSTRUCTIONS.get(query_type, TYPE_INSTRUCTIONS["single"])
    user_content  = USER_PROMPT_TEMPLATE.format(
        context         = retrieval_result["context"],
        query           = query,
        meta_filter     = retrieval_result.get("filter", {}),
        rewritten_query = rewritten_query,
    )
    messages = []
    if history:
        for h in history:
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_content})
    return system_prompt, messages


# Gemma4 클라이언트 래퍼
class _MessagesResponse:
    def __init__(self, text: str):
        self.content = [type("Block", (), {"text": text})()]


class _StreamContextManager:
    def __init__(self, system, messages, max_tokens, tokenizer, model):
        self._system     = system
        self._messages   = messages
        self._max_tokens = max_tokens
        self._tokenizer  = tokenizer
        self._model      = model
        self._streamer   = None
        self._thread     = None

    def __enter__(self):
        chat = _build_chat_input(self._system, self._messages)
        input_ids = self._tokenizer.apply_chat_template(
            chat, return_tensors="pt", add_generation_prompt=True
        ).to(DEVICE)
        self._streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            input_ids=input_ids, max_new_tokens=self._max_tokens,
            do_sample=True, temperature=0.2, streamer=self._streamer,
        )
        self._thread = Thread(target=self._model.generate, kwargs=gen_kwargs, daemon=True)
        self._thread.start()
        return self

    @property
    def text_stream(self):
        for chunk in self._streamer:
            if chunk:
                yield chunk

    def __exit__(self, *args):
        if self._thread:
            self._thread.join()


def _build_chat_input(system: str, messages: list) -> list:
    chat = []
    if system:
        chat.append({"role": "user",      "content": system})
        chat.append({"role": "assistant", "content": "알겠습니다."})
    for m in messages:
        chat.append({"role": m["role"], "content": m["content"]})
    return chat


class _MessagesNamespace:
    def __init__(self, tokenizer, model):
        self._tokenizer = tokenizer
        self._model     = model

    def create(self, model, max_tokens, system, messages):
        chat = _build_chat_input(system, messages)
        input_ids = self._tokenizer.apply_chat_template(
            chat, return_tensors="pt", add_generation_prompt=True
        ).to(DEVICE)
        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids, max_new_tokens=max_tokens, do_sample=True, temperature=0.2
            )
        new_tokens = output_ids[0][input_ids.shape[-1]:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return _MessagesResponse(text)

    def stream(self, model, max_tokens, system, messages):
        return _StreamContextManager(system, messages, max_tokens, self._tokenizer, self._model)


class _GemmaClient:
    def __init__(self, tokenizer, model):
        self.messages = _MessagesNamespace(tokenizer, model)


class BidMateGenerator:
    def __init__(self, llm_client, get_context_fn):
        self.client      = llm_client
        self.get_context = get_context_fn
        self._call_count = 0

    def _rewrite_query(self, query: str, history=None) -> str:
        history_summary = ""
        if history:
            prev_turns = []
            for h in history[-4:]:
                role = "사용자" if h["role"] == "user" else "어시스턴트"
                prev_turns.append(f"{role}: {h['content'][:100]}")
            history_summary = "\n[이전 대화]\n" + "\n".join(prev_turns) + "\n"
        user_content = f"{history_summary}\n[재작성할 질문]\n{query}"
        try:
            response = self.client.messages.create(
                model=LLM_MODEL, max_tokens=MAX_TOKENS_REWRITE,
                system=REWRITE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.warning(f"쿼리 재작성 실패 (원본 사용): {e}")
            return query

    def generate(self, query, history=None, meta_filter=None, verbose=False) -> dict:
        t_start = time.time()
        latency = {}

        t0 = time.time()
        rewritten_query = self._rewrite_query(query, history)
        latency["rewrite_ms"] = round((time.time() - t0) * 1000)

        t0 = time.time()
        retrieval_result = self.get_context(rewritten_query, history=history, meta_filter=meta_filter)
        latency["retrieval_ms"] = round((time.time() - t0) * 1000)

        if not retrieval_result.get("context", "").strip():
            return {
                "answer"          : "제공된 문서에서 관련 내용을 찾을 수 없습니다.",
                "sources"         : [],
                "rewritten_query" : rewritten_query,
                "original_query"  : query,
                "meta_filter"     : retrieval_result.get("filter", {}),
                "sub_queries"     : retrieval_result.get("sub_queries", []),
                "latency_ms"      : latency,
            }

        system_prompt, messages = build_prompt(
            query=query, rewritten_query=rewritten_query,
            retrieval_result=retrieval_result, history=history,
        )

        t0 = time.time()
        try:
            response = self.client.messages.create(
                model=LLM_MODEL, max_tokens=MAX_TOKENS_GENERATE,
                system=system_prompt, messages=messages,
            )
            answer_text = response.content[0].text.strip()
            self._call_count += 1
        except Exception as e:
            logger.error(f"LLM 호출 실패: {e}")
            answer_text = f"답변 생성 중 오류가 발생했습니다: {e}"
        latency["generation_ms"] = round((time.time() - t0) * 1000)

        sources_text = format_sources(retrieval_result.get("sources", []))
        if "[출처]" not in answer_text:
            answer_text += "\n" + sources_text

        latency["total_ms"] = round((time.time() - t_start) * 1000)

        return {
            "answer"          : answer_text,
            "sources"         : retrieval_result.get("sources", []),
            "rewritten_query" : rewritten_query,
            "original_query"  : query,
            "meta_filter"     : retrieval_result.get("filter", {}),
            "sub_queries"     : retrieval_result.get("sub_queries", []),
            "latency_ms"      : latency,
        }

    def generate_stream(self, query, history=None, meta_filter=None):
        # 1단계: 쿼리 재작성
        yield {"type": "progress", "data": {"step": 1, "message": "쿼리 재작성 중..."}}
        rewritten_query = self._rewrite_query(query, history)
        yield {"type": "progress", "data": {"step": 1, "message": f"쿼리 재작성 완료: {rewritten_query[:50]}"}}

        # 2단계: 문서 검색
        yield {"type": "progress", "data": {"step": 2, "message": "문서 검색 중..."}}
        retrieval_result = self.get_context(rewritten_query, history=history, meta_filter=meta_filter)
        meta_filter_info = retrieval_result.get("filter", {})
        sub_queries      = retrieval_result.get("sub_queries", [])
        detail = f"필터: {meta_filter_info}"
        if len(sub_queries) > 1:
            detail += f" | 서브쿼리 {len(sub_queries)}개"
        yield {"type": "progress", "data": {"step": 2, "message": f"문서 검색 완료 - {detail}"}}

        # 3단계: Reranker
        yield {"type": "progress", "data": {"step": 3, "message": f"Reranker 적용 중... (후보 {len(retrieval_result.get('sources', []))}개)"}}

        yield {"type": "meta", "data": {
            "rewritten_query" : rewritten_query,
            "filter"          : meta_filter_info,
            "sources"         : retrieval_result.get("sources", []),
            "sub_queries"     : sub_queries,
            "context"         : retrieval_result.get("context", ""),
        }}

        system_prompt, messages = build_prompt(
            query=query, rewritten_query=rewritten_query,
            retrieval_result=retrieval_result, history=history,
        )

        # 4단계: 답변 생성
        yield {"type": "progress", "data": {"step": 4, "message": "답변 생성 중..."}}

        with self.client.messages.stream(
            model=LLM_MODEL, max_tokens=MAX_TOKENS_GENERATE,
            system=system_prompt, messages=messages,
        ) as stream:
            for text_chunk in stream.text_stream:
                yield {"type": "chunk", "data": text_chunk}

        sources_text = format_sources(retrieval_result.get("sources", []))
        yield {"type": "done", "data": {"sources_text": sources_text}}


def init_generator(get_context_fn) -> BidMateGenerator:
    adapter_path = str(ADAPTER_PATH)
    if not Path(adapter_path).exists():
        raise FileNotFoundError(f"어댑터 경로 없음: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        cache_dir="/mnt/gukrul/hf_cache/hub",
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir="/mnt/gukrul/hf_cache/hub",
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    llm_client = _GemmaClient(tokenizer, model)
    return BidMateGenerator(llm_client, get_context_fn)


# 모듈 임포트 시 자동 초기화
generator: BidMateGenerator = None
