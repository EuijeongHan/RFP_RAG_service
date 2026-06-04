#!/usr/bin/env python3
"""
vLLM 전환 스크립트
- config.py: VLLM_BASE_URL 추가
- generation.py: transformers 제거, OpenAI 클라이언트로 vLLM 호출
- retrieval.py: Reranker CPU로 변경 (VRAM 확보)
- web_rag_admin.py: 서브캡션 변경
"""
import re

MAIN_PY = "/mnt/gukrul/main_py"

# ── 1. config.py ──────────────────────────────────────────────
def update_config():
    path = f"{MAIN_PY}/config.py"
    with open(path) as f: src = f.read()

    # 모델 ID는 vLLM 서버 실행 시 지정하므로 참고용으로만 유지
    src = src.replace('microsoft/Phi-4-mini-instruct', 'google/gemma-4-E4B-it')
    src = src.replace('"peft_output" / "phi4-mini" / "lora_adapter"',
                      '"peft_output" / "gemma4-E4B_v2" / "lora_adapter_v2"')
    src = src.replace('MAX_TOKENS_GENERATE = 1500', 'MAX_TOKENS_GENERATE = 800')
    src = src.replace('MAX_TOKENS_GENERATE = 800', 'MAX_TOKENS_GENERATE = 800')

    # vLLM 서버 URL 추가
    if 'VLLM_BASE_URL' not in src:
        src += '\n# vLLM 서버\nVLLM_BASE_URL = "http://localhost:8100/v1"\n'

    with open(path, "w") as f: f.write(src)
    print("✓ config.py")

# ── 2. generation.py - vLLM 클라이언트 방식 ───────────────────
def update_generation():
    path = f"{MAIN_PY}/generation.py"
    with open(path) as f: src = f.read()

    # init_generator를 vLLM OpenAI 클라이언트 방식으로 교체
    old_init = re.search(r'def init_generator.*?^# 모듈', src, re.DOTALL | re.MULTILINE)
    if old_init:
        src = src.replace(old_init.group(), '''def init_generator(get_context_fn) -> BidMateGenerator:
    from openai import OpenAI
    from config import VLLM_BASE_URL, LLM_MODEL

    logger.info(f"vLLM 클라이언트 초기화 중... ({VLLM_BASE_URL})")

    # vLLM은 OpenAI 호환 API 제공 - 모델 로드 없이 HTTP 클라이언트만 생성
    openai_client = OpenAI(
        base_url=VLLM_BASE_URL,
        api_key="EMPTY",  # vLLM은 API 키 불필요
    )

    # vLLM용 클라이언트 래퍼
    llm_client = _VLLMClient(openai_client)
    logger.info("vLLM 클라이언트 초기화 완료")
    return BidMateGenerator(llm_client, get_context_fn)


# 모듈 임포트 시 자동 초기화
generator: BidMateGenerator = None
''')

    # _VLLMClient 클래스 추가 (기존 _GemmaClient 앞에)
    vllm_client_code = '''
class _VLLMMessagesResponse:
    def __init__(self, text: str):
        self.content = [type("Block", (), {"text": text})()]


class _VLLMStreamContextManager:
    def __init__(self, system, messages, max_tokens, client):
        self._system     = system
        self._messages   = messages
        self._max_tokens = max_tokens
        self._client     = client
        self._stream     = None

    def __enter__(self):
        from config import LLM_MODEL
        msgs = []
        if self._system:
            msgs.append({"role": "system", "content": self._system})
        for m in self._messages:
            msgs.append({"role": m["role"], "content": m["content"]})

        self._stream = self._client.chat.completions.create(
            model      = LLM_MODEL,
            messages   = msgs,
            max_tokens = self._max_tokens,
            temperature= 0.2,
            stream     = True,
        )
        return self

    @property
    def text_stream(self):
        for chunk in self._stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    def __exit__(self, *args):
        pass


class _VLLMMessagesNamespace:
    def __init__(self, client):
        self._client = client

    def create(self, model, max_tokens, system, messages):
        from config import LLM_MODEL
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        for m in messages:
            msgs.append({"role": m["role"], "content": m["content"]})

        response = self._client.chat.completions.create(
            model      = LLM_MODEL,
            messages   = msgs,
            max_tokens = max_tokens,
            temperature= 0.2,
            stream     = False,
        )
        return _VLLMMessagesResponse(response.choices[0].message.content)

    def stream(self, model, max_tokens, system, messages):
        return _VLLMStreamContextManager(system, messages, max_tokens, self._client)


class _VLLMClient:
    def __init__(self, client):
        self.messages = _VLLMMessagesNamespace(client)

'''

    # _GemmaClient 앞에 삽입
    if '_VLLMClient' not in src:
        src = src.replace('class _GemmaClient:', vllm_client_code + 'class _GemmaClient:')

    with open(path, "w") as f: f.write(src)
    print("✓ generation.py (vLLM 클라이언트 방식)")

# ── 3. retrieval.py - Reranker CPU ────────────────────────────
def update_retrieval_cpu():
    path = f"{MAIN_PY}/retrieval.py"
    with open(path) as f: src = f.read()

    src = src.replace(
        '    reranker = CrossEncoder(\n        RERANKER_ID,\n        device=DEVICE,\n    )',
        '    reranker = CrossEncoder(\n        RERANKER_ID,\n        device="cpu",\n    )'
    )
    with open(path, "w") as f: f.write(src)
    print("✓ retrieval.py (Reranker → CPU)")

# ── 4. web_rag_admin.py 서브캡션 ──────────────────────────────
def update_admin_caption():
    path = f"{MAIN_PY}/web_rag_admin.py"
    with open(path) as f: src = f.read()
    for old in ['Gemma4 E4B · LoRA · ChromaDB · BM25 · KURE-v1',
                'Phi-4-mini · LoRA · ChromaDB · BM25 · KURE-v1']:
        src = src.replace(old, 'Gemma4 E4B · vLLM · ChromaDB · BM25 · KURE-v1')
    with open(path, "w") as f: f.write(src)
    print("✓ web_rag_admin.py (캡션)")

if __name__ == "__main__":
    update_config()
    update_generation()
    update_retrieval_cpu()
    update_admin_caption()
    print("\n✅ vLLM 전환 완료")
    print("   다음 단계: bash /mnt/gukrul/start_vllm.sh")
