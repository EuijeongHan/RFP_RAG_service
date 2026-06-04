#!/usr/bin/env python3
"""
Phi-4-mini + LoRA 전환 스크립트
- config.py: BASE_MODEL_ID, LLM_MODEL, ADAPTER_PATH 변경
- generation.py: _build_chat_input Phi-4 방식, init_generator 수정
- retrieval.py: Reranker GPU로 복구
"""
import os, re, sys

MAIN_PY = "/mnt/gukrul/main_py"

def update_config():
    path = f"{MAIN_PY}/config.py"
    with open(path) as f: src = f.read()

    replacements = [
        ('google/gemma-4-E4B-it', 'microsoft/Phi-4-mini-instruct'),
        ('"peft_output" / "gemma4-E4B_v2" / "lora_adapter_v2"',
         '"peft_output" / "phi4-mini" / "lora_adapter"'),
        ('MAX_TOKENS_GENERATE = 800', 'MAX_TOKENS_GENERATE = 800'),
    ]
    for old, new in replacements:
        src = src.replace(old, new)

    with open(path, "w") as f: f.write(src)
    print("✓ config.py")

def update_generation():
    path = f"{MAIN_PY}/generation.py"
    with open(path) as f: src = f.read()

    # import 정리
    src = re.sub(r'from peft import PeftModel\n', '', src)
    if 'from peft import PeftModel' not in src:
        src = src.replace(
            'from transformers import AutoTokenizer',
            'from peft import PeftModel\nfrom transformers import AutoTokenizer'
        )

    # _build_chat_input - Phi-4 방식 (system role 직접 지원)
    old_build = re.search(r'def _build_chat_input.*?return chat\n', src, re.DOTALL)
    if old_build:
        src = src.replace(old_build.group(), '''def _build_chat_input(system: str, messages: list) -> list:
    chat = []
    if system:
        chat.append({"role": "system", "content": system})
    for m in messages:
        chat.append({"role": m["role"], "content": m["content"]})
    return chat
''')

    # init_generator - Phi-4 bfloat16 + LoRA
    old_init = re.search(r'def init_generator.*?^# 모듈', src, re.DOTALL | re.MULTILINE)
    if old_init:
        src = src.replace(old_init.group(), '''def init_generator(get_context_fn) -> BidMateGenerator:
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
''')

    with open(path, "w") as f: f.write(src)
    print("✓ generation.py")

def update_retrieval_gpu():
    path = f"{MAIN_PY}/retrieval.py"
    with open(path) as f: src = f.read()

    src = src.replace(
        '    reranker = CrossEncoder(\n        RERANKER_ID,\n        device="cpu",\n    )',
        '    reranker = CrossEncoder(\n        RERANKER_ID,\n        device=DEVICE,\n    )'
    )
    with open(path, "w") as f: f.write(src)
    print("✓ retrieval.py (Reranker → GPU)")

def update_admin_caption():
    path = f"{MAIN_PY}/web_rag_admin.py"
    with open(path) as f: src = f.read()
    src = src.replace(
        'Gemma4 E4B · LoRA · ChromaDB · BM25 · KURE-v1',
        'Phi-4-mini · LoRA · ChromaDB · BM25 · KURE-v1'
    )
    with open(path, "w") as f: f.write(src)
    print("✓ web_rag_admin.py (캡션)")

if __name__ == "__main__":
    update_config()
    update_generation()
    update_retrieval_gpu()
    update_admin_caption()
    print("\n✅ Phi-4 전환 완료")
