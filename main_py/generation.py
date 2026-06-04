import time
import logging
from threading import Thread
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer


from config import (
    BASE_MODEL_ID, ADAPTER_PATH,
    MAX_TOKENS_REWRITE, MAX_TOKENS_GENERATE,
    LLM_MODEL,
)

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 프롬프트 템플릿
BASE_SYSTEM_PROMPT = """당신은 공공 입찰 RFP 문서 분석 어시스턴트입니다.
아래 규칙을 반드시 지키세요.

규칙1: [검색된 문서]에 있는 내용만 답변하세요. 문서 외 지식 사용 금지.
규칙2: 금액/날짜/기간 등 수치는 문서에 나온 숫자 그대로만 쓰세요. 억원 변환 절대 금지. 괄호 안 변환도 금지. 원문 숫자만 쓰세요.
규칙3: 문서에 답이 없으면 반드시 "제공된 문서에서 확인할 수 없습니다"라고만 답하세요.
규칙4: 문서에 없는 내용을 추측하거나 지어내는 것은 절대 금지입니다.
규칙5: 답변은 간결하게 핵심만 작성하세요.
규칙6: 전화번호, 이메일, 개인정보, 비밀번호 등은 문서에 명시된 경우에만 답변하세요. 문서에 없으면 절대 생성하지 마세요.
규칙7: 문서에서 찾은 내용이 질문과 실제로 관련 있는지 반드시 확인하세요. 관련 없으면 "제공된 문서에서 확인할 수 없습니다"라고 답하세요.
"""

TYPE_INSTRUCTIONS = {
    "single": """
[답변 형식]
- 질문에 직접 답변하는 1~3문장으로 시작합니다.
- 필요 시 세부 내용을 bullet point로 정리합니다.
- 수치(예산, 기간, 인원 등)는 굵게 표시합니다.
""",
    "compare": """
[복수 기관/사업 비교 답변 규칙]
1. 질문에 언급된 각 기관별로 사업명, 예산, 기간, 주요 내용을 검색된 문서에서 찾아 정리하세요.
2. 각 기관을 ### 기관명 헤더로 구분해 작성하세요.
3. 마지막에 반드시 마크다운 비교표를 작성하세요. 예시:
| 항목 | 기관A | 기관B |
|------|-------|-------|
| 예산 | 실제값 | 실제값 |
| 기간 | 실제값 | 실제값 |
4. 수치는 문서 원문 그대로 쓰세요(억원 변환 금지).
5. 문서에 없는 항목은 "확인 불가"라고 쓰세요.
6. 같은 내용을 반복하지 마세요.
""",
    "followup": """
[답변 형식 - 후속 질문]
- 이전 대화 맥락을 참고하되, 반드시 [검색된 문서]의 내용을 기준으로 답변합니다.
- 이전 답변 내용을 절대 반복하지 마세요. 현재 질문이 요구하는 새로운 정보만 답변합니다.
- 현재 질문의 키워드가 [검색된 문서]에 있으면 그 내용만 답변합니다.
- 현재 질문에 해당하는 내용이 [검색된 문서]에 없으면 반드시 "제공된 문서에서 확인할 수 없습니다"라고만 답하세요.
- 답변은 2~5문장으로 간결하게 작성합니다.
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
7. 현재 질문에 새로운 기관명이 있으면 반드시 그 기관명을 기준으로 재작성합니다. 이전 대화의 기관명을 따르지 마세요.
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

# 답변 포멧

def format_sources(sources: list) -> str:
    if not sources:
        return ""
    
    # 기관별 대표 출처 추출 (중복 기관 제거)
    seen_agencies = set()
    unique_sources = []
    sorted_sources = sorted(sources, key=lambda s: s.get("score", 0), reverse=True)
    for s in sorted_sources:
        agency = s.get("agency", "")
        if agency and agency not in seen_agencies:
            seen_agencies.add(agency)
            unique_sources.append(s)
        elif not agency and len(unique_sources) == 0:
            unique_sources.append(s)
    
    if not unique_sources:
        unique_sources = [sorted_sources[0]]
    
    filenames = []
    for src in unique_sources:
        filename = (
            src.get("source_file") or
            src.get("file_name") or
            src.get("filename") or
            src.get("source") or
            src.get("project") or
            src.get("agency", "알 수 없는 파일")
        )
        if isinstance(filename, str) and ("/" in filename or "\\" in filename):
            filename = Path(filename).name
        filenames.append(filename)
    
    return "\n출처 : " + " / ".join(filenames)



def _trim_assistant_content(content: str, max_chars: int = 200) -> str:
    import re as _re
    content = _re.sub(r'\[검색된 문서\].*?---', '', content, flags=_re.DOTALL).strip()
    content = _re.sub(r'\[출처\].*$', '', content, flags=_re.DOTALL).strip()
    if len(content) > max_chars:
        content = content[:max_chars] + "..."
    return content

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
            if h["role"] == "user":
                messages.append({"role": "user", "content": h["content"]})
            elif h["role"] == "assistant":
                trimmed = _trim_assistant_content(h["content"])
                messages.append({"role": "assistant", "content": trimmed})
    messages.append({"role": "user", "content": user_content})
    return system_prompt, messages



LLM_PROVIDERS = {
    "local"    : {"name": "로컬 (Phi-4-mini)", "model": None},
    "openai"   : {"name": "OpenAI GPT-4o-mini", "model": "gpt-4o-mini"},
    "gemini"   : {"name": "Google Gemini 1.5 Flash", "model": "gemini-1.5-flash"},
    "openrouter": {"name": "OpenRouter", "model": "openai/gpt-4o-mini"},
}

class _APIMessagesNamespace:
    def __init__(self, provider: str, api_key: str, model: str = None):
        self._provider = provider
        self._api_key  = api_key
        self._model    = model

    def create(self, model, max_tokens, system, messages):
        if self._provider == "openai":
            # 1. 여기에 예외 처리를 삽입합니다.
            try:
                from openai import OpenAI
            except ImportError:
                return _MessagesResponse("OpenAI 라이브러리가 설치되지 않았습니다. 환경을 확인하거나 local 모드를 사용해주세요.")
                
            client = OpenAI(api_key=self._api_key)
            msgs = [{"role": "system", "content": system}] + messages
            resp = client.chat.completions.create(
                model=self._model or "gpt-4o-mini",
                messages=msgs,
                max_tokens=max_tokens,
                temperature=0.2,
            )
            text = resp.choices[0].message.content.strip()
            return _MessagesResponse(text)

        elif self._provider == "gemini":
            from google import genai as gai
            from google.genai import types
            client = gai.Client(api_key=self._api_key)
            msgs_gemini = [{"role": "user", "parts": [{"text": system}]}]
            msgs_gemini.append({"role": "model", "parts": [{"text": "알겠습니다."}]})
            for msg in messages:
                role = "user" if msg["role"] == "user" else "model"
                msgs_gemini.append({"role": role, "parts": [{"text": msg["content"]}]})
            resp = client.models.generate_content(
                model=self._model or "gemini-2.5-flash",
                contents=msgs_gemini,
                config=types.GenerateContentConfig(max_output_tokens=max(max_tokens, 2000), temperature=0.2),
            )
            return _MessagesResponse(resp.text.strip())

        elif self._provider == "openrouter":
            # 2. OpenRouter도 내부적으로 openai 라이브러리를 사용하므로 함께 처리합니다.
            try:
                from openai import OpenAI
            except ImportError:
                return _MessagesResponse("OpenAI 라이브러리가 설치되지 않아 OpenRouter를 사용할 수 없습니다.")
                
            client = OpenAI(
                api_key=self._api_key,
                base_url="https://openrouter.ai/api/v1",
            )
            msgs = [{"role": "system", "content": system}] + messages
            resp = client.chat.completions.create(
                model=self._model or "openai/gpt-4o-mini",
                messages=msgs,
                max_tokens=max_tokens,
                temperature=0.2,
            )
            text = resp.choices[0].message.content.strip()
            return _MessagesResponse(text)

    def stream(self, model, max_tokens, system, messages):
        return _APIStreamContext(self._provider, self._api_key, self._model, system, messages, max_tokens)


class _APIStreamContext:
    def __init__(self, provider, api_key, model, system, messages, max_tokens):
        self._provider   = provider
        self._api_key    = api_key
        self._model      = model
        self._system     = system
        self._messages   = messages
        self._max_tokens = max_tokens

    def __enter__(self):
        return self

    @property
    def text_stream(self):
        if self._provider == "openai":
            # 3. 스트리밍 환경에서도 가독성 있게 에러를 내뱉도록 처리합니다.
            try:
                from openai import OpenAI
            except ImportError:
                yield "OpenAI 라이브러리가 설치되지 않았습니다."
                return
                
            client = OpenAI(api_key=self._api_key)
            msgs = [{"role": "system", "content": self._system}] + self._messages
            with client.chat.completions.stream(
                model=self._model or "gpt-4o-mini",
                messages=msgs,
                max_tokens=self._max_tokens,
                temperature=0.2,
            ) as stream:
                for text in stream.text_stream:
                    if text:
                        yield text

        elif self._provider == "gemini":
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=self._api_key)
            msgs_gemini = [{"role": "user", "parts": [{"text": self._system}]}]
            msgs_gemini.append({"role": "model", "parts": [{"text": "알겠습니다."}]})
            for msg in self._messages:
                role = "user" if msg["role"] == "user" else "model"
                msgs_gemini.append({"role": role, "parts": [{"text": msg["content"]}]})
            for chunk in client.models.generate_content_stream(
                model=self._model or "gemini-2.5-flash",
                contents=msgs_gemini,
                config=types.GenerateContentConfig(max_output_tokens=max(self._max_tokens, 2000), temperature=0.2),
            ):
                if chunk.text:
                    yield chunk.text

        elif self._provider == "openrouter":
            try:
                from openai import OpenAI
            except ImportError:
                yield "OpenAI 라이브러리가 설치되지 않았습니다."
                return
                
            client = OpenAI(
                api_key=self._api_key,
                base_url="https://openrouter.ai/api/v1",
            )
            msgs = [{"role": "system", "content": self._system}] + self._messages
            with client.chat.completions.stream(
                model=self._model or "openai/gpt-4o-mini",
                messages=msgs,
                max_tokens=self._max_tokens,
                temperature=0.2,
            ) as stream:
                for text in stream.text_stream:
                    if text:
                        yield text

    def __exit__(self, *args):
        pass

class _ExternalAPIClient:
    def __init__(self, provider: str, api_key: str, model: str = None):
        self.messages = _APIMessagesNamespace(provider, api_key, model)

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
        actual_device = next(self._model.parameters()).device
        input_ids = self._tokenizer.apply_chat_template(
            chat, return_tensors="pt", add_generation_prompt=True
        )["input_ids"].to(actual_device)
        self._streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            input_ids=input_ids, max_new_tokens=self._max_tokens,
            do_sample=True, temperature=0.2, streamer=self._streamer,
            use_cache=True # 양자화 모델 멀티스레딩 스트리밍 안정화
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
        chat.append({"role": "system", "content": system})
    for m in messages:
        chat.append({"role": m["role"], "content": m["content"]})
    return chat

class _MessagesNamespace:
    def __init__(self, tokenizer, model):
        self._tokenizer = tokenizer
        self._model     = model

    def create(self, model, max_tokens, system, messages):
        chat = _build_chat_input(system, messages)
        tokenized = self._tokenizer.apply_chat_template(
            chat, return_tensors="pt", add_generation_prompt=True
        )
        actual_device = next(self._model.parameters()).device
        if hasattr(tokenized, "input_ids"):
            input_ids = tokenized.input_ids.to(actual_device)
        else:
            input_ids = tokenized.to(actual_device)
        input_len = input_ids.shape[-1]
        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids, max_new_tokens=max_tokens, do_sample=True, temperature=0.2, use_cache=True
            )
        new_tokens = output_ids[0][input_len:]
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
        self._base_client = llm_client  # 로컬 모델 클라이언트 보관
        self.get_context = get_context_fn
        self._call_count = 0

    def set_llm_config(self, provider: str, api_key: str, model: str = None):
            """런타임 LLM 전환 - 사용자가 선택한 provider에 따라 유연하게 변경"""
            if provider == "local" or not api_key:
                self.client = self._base_client  # 로컬 모델(Phi-4)로 복귀
                logger.info("LLM 구동 모드가 로컬(Phi-4)로 설정되었습니다.")
            else:
                # 외부 API 클라이언트(OpenAI, Gemini 등)로 전환
                self.client = _ExternalAPIClient(provider, api_key, model)
                logger.info(f"LLM 구동 모드가 외부 API({provider})로 전환되었습니다.")

            
    def _rewrite_query(self, query: str, history=None) -> str:
        import re
        rewritten = query

        # 구어체 → 공문서 용어 정규화
        # 오타 사전 (E타입 대응)
        typo_dict = {
            "코이까": "KOICA", "전쟈죠달": "전자조달", "전쟈": "전자",
            "예싼": "예산", "에산": "예산", "에싼": "예산",
            "뱡송": "방송", "씨스탬": "시스템", "씨스템": "시스템",
            "시스탬": "시스템", "구쭉": "구축", "구쭉": "구축",
            # 기관명 오타
            "코이까": "KOICA", "아시아물위원훼": "아시아물위원회",
            "물위원훼": "물위원회", "그렌드코리아레져": "그랜드코리아레저",
            "그렌드": "그랜드",
            # 지명 오타
            "키르기즈쓰탄": "키르기스탄", "우즈벡": "우즈베키스탄",
            "우즈백": "우즈베키스탄",
            # 용어 오타 (긴 것 먼저)
            "전쟈죠달": "전자조달", "구룹웨에": "그룹웨어", "구룹웨어": "그룹웨어",
            "시쓰탬": "시스템", "씨쓰템": "시스템", "시쓰템": "시스템",
            "씨스탬": "시스템", "씨스템": "시스템",
            "관계시스템": "관개시스템",
            # 단어 오타
            "예싼": "예산", "에산": "예산", "슴아트": "스마트",
            "기대효괍": "기대효과", "츄진": "추진", "뱡송": "방송",
            "레져": "레저", "구쭉": "구축", "프로젝": "프로젝트",
            "플젝": "프로젝트", "베네핏": "benefit",
            # 구어체/축약
            "알려주새요": "알려주세요", "있습니가": "있습니까",
            "고대": "고려대학교",
        }
        # 긴 문자열 먼저 치환 (이중치환 방지)
        for typo, correct in sorted(typo_dict.items(), key=lambda x: -len(x[0])):
            rewritten = rewritten.replace(typo, correct)

        # rapidfuzz 기반 퍼지 토큰 보정
        try:
            from retrieval import fuzzy_normalize_query
            rewritten = fuzzy_normalize_query(rewritten, threshold=82)
        except Exception:
            pass

        colloquial_map = [
            (r"비교(해줘|해주세요|해봐|해봐줘|하면|하자)?", ""),
            (r"알려(줘|주세요|줘요)?", ""),
            (r"가르쳐(줘|주세요)?", ""),
            (r"얼마(야|예요|이에요|인가요|\?)?", "예산 규모"),
            (r"얼마나\s*(돼|되나요|됩니까|\?)?", "규모"),
            (r"어디(야|예요|이에요|인가요|\?)?", "소재지"),
            (r"언제(야|예요|이에요|인가요|\?)?", "일정"),
            (r"기간이?\s*(어떻게|어때|얼마나)", "사업 기간"),
            (r"뭐(야|예요|이에요|인가요|\?)?", "내용"),
            (r"어때(요)?", "현황"),
        ]
        for pattern, replacement in colloquial_map:
            rewritten = re.sub(pattern, replacement, rewritten)

        # 히스토리에서 기관명 보완
        if history:
            for h in reversed(history[-6:]):
                if h["role"] == "user":
                    agency_pat = r"한국[가-힣]{1,6}공사|[가-힣]{2,8}공단|[가-힣]{2,8}은행|[가-힣]{2,8}공사|[가-힣]{2,8}연구원|[가-힣]{2,8}대학교|[가-힣]{2,8}의료원"
                    prev_agency = re.findall(agency_pat, h["content"])
                    curr_agency = re.findall(agency_pat, rewritten)
                    if prev_agency and not curr_agency:
                        rewritten = prev_agency[0] + " " + rewritten
                    break
        return rewritten
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
                # 기본 설정된 클라이언트로 시도
                response = self.client.messages.create(
                    model=LLM_MODEL, max_tokens=MAX_TOKENS_GENERATE,
                    system=system_prompt, messages=messages,
                )
                answer_text = response.content[0].text.strip()
                self._call_count += 1
            except Exception as e:
                # 401 인증 에러나 API 키 오류가 발생할 경우 로컬 모델로 강제 긴급 우회
                if "401" in str(e) or "api_key" in str(e).lower() or "invalid_api_key" in str(e):
                    logger.warning(f"외부 API 인증 실패(401). 로컬 모델로 자동 우회합니다. 에러: {e}")
                    try:
                        response = self._base_client.messages.create(
                            model=LLM_MODEL, max_tokens=MAX_TOKENS_GENERATE,
                            system=system_prompt, messages=messages,
                        )
                        answer_text = response.content[0].text.strip()
                    except Exception as local_err:
                        answer_text = f"로컬 백업 모델 구동 실패: {local_err}"
                else:
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
    
            # 사용자가 선택한 클라이언트 지정을 가져옵니다.
            active_client = self.client
            
            try:
                # 먼저 유저가 세팅한 모드로 스트리밍 시도 시 선언적 컨텍스트 진입 체크
                with active_client.messages.stream(
                    model=LLM_MODEL, max_tokens=MAX_TOKENS_GENERATE,
                    system=system_prompt, messages=messages,
                ) as stream:
                    for text_chunk in stream.text_stream:
                        yield {"type": "chunk", "data": text_chunk}
            except Exception as e:
                # 스트림 도중 401 키 에러가 터지면 즉시 중단하고 로컬 백업 모델로 스위칭하여 처음부터 다시 스트리밍
                if "401" in str(e) or "api_key" in str(e).lower() or "invalid_api_key" in str(e):
                    yield {"type": "progress", "data": {"step": 4, "message": "⚠️ 외부 API 키 인증 실패로 로컬 모델(Phi-4)로 자동 전환하여 답변을 생성합니다."}}
                    active_client = self._base_client
                    with active_client.messages.stream(
                        model=LLM_MODEL, max_tokens=MAX_TOKENS_GENERATE,
                        system=system_prompt, messages=messages,
                    ) as stream:
                        for text_chunk in stream.text_stream:
                            yield {"type": "chunk", "data": text_chunk}
                else:
                    yield {"type": "chunk", "data": f"\n스트리밍 생성 오류 발생: {e}"}
    
            sources_text = format_sources(retrieval_result.get("sources", []))
            yield {"type": "done", "data": {"sources_text": sources_text}}

def init_generator(get_context_fn) -> BidMateGenerator:
    adapter_path = str(ADAPTER_PATH)
    if not Path(adapter_path).exists():
        raise FileNotFoundError(f"어댑터 경로 없음: {adapter_path}")

    logger.info("Phi-4-mini-instruct 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        cache_dir="/mnt/gukrul/hf_cache/hub",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token else "<|pad|>"
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir="/mnt/gukrul/hf_cache/hub",
    )
    logger.info(f"LoRA 어댑터 로드 중... 경로: {adapter_path}")
    model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    model.eval()

    llm_client = _GemmaClient(tokenizer, model)
    return BidMateGenerator(llm_client, get_context_fn)


# 모듈 임포트 시 자동 초기화
generator: BidMateGenerator = None