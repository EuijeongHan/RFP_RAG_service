from pathlib import Path
"""
입찰메이트 RAG — Generation 모듈
======================================================
담당 : Generation 파트 (한의정)
용도 : LLM 텍스트 생성 전용 (Retrieval과 완전 분리)

[파일 관계]
    retrieval_interface_F.py    ← Retrieval (Hybrid Search + Reranker)
    generation_interface.py     ← 현재 파일 (LLM 생성만 담당)
    serving_main.py             ← Retrieval + Generation 조합 + 히스토리 관리
    eval_e2e_abde.py            ← A/B/D/E타입 End-to-End 비교 평가
    eval_e2e_ctype.py           ← C타입 End-to-End 비교 평가 (히스토리 포함)
    eval_quant_judge_dual.py    ← LLM-as-a-Judge 정량 평가 (듀얼 Judge)
    check_release_gate.py       ← Release Gate 통과 여부 체크

[구조]
    BidMateGenerator (추상 기반)
    ├── LocalHFGenerator : GEMMA / QWEN / PHI (GCP L4 로컬 모델)
    └── APIGenerator     : OPENAI (gpt-5-mini, OpenAI API)

    get_generator(scenario)  : 팩토리 함수 — serving_main.py 에서 이 함수만 호출

[시나리오 구성 — 임베딩_LLM 조합 12개]
    KURE_GEMMA  / KURE_QWEN  / KURE_PHI  / KURE_OPENAI
    KOE5_GEMMA  / KOE5_QWEN  / KOE5_PHI  / KOE5_OPENAI
    SMALL_GEMMA / SMALL_QWEN / SMALL_PHI / SMALL_OPENAI

=======================================================================
[모델 선정 근거]

▶ GEMMA — google/gemma-4-E4B-it  ← 채택
  선택 이유:
    1) Gemma 4 최신 아키텍처 (AutoModelForImageTextToText 멀티모달)
    2) GCP L4(24GB) 기준 bfloat16 ~8GB → 양자화 없이 안정 운영 가능
    3) 파인튜닝: Colab L4에서 4-bit NF4 LoRA 가능 (BATCH=2, GRAD_ACCUM=8)
    4) 채팅 포맷: <start_of_turn>user/model (retrieval_interface_F.py 템플릿과 일치)

▶ QWEN — Qwen/Qwen2.5-7B-Instruct  ← 채택
  선택 이유:
    1) 7B 파라미터 → 입찰 공고 도메인 긴 문서 이해 유리
    2) 파인튜닝: Colab A100(40GB) bfloat16 풀정밀도 (팀장 노트북 기준)
    3) 파인튜닝 후 merge_and_unload() → 단일 병합 모델로 GCP 이전
    4) 채팅 포맷: ChatML (<|im_start|>role)

▶ PHI — microsoft/Phi-4-mini-instruct  ← 채택
  선택 이유:
    1) 4B 경량 모델, 범용 instruction following 특화
    2) GCP L4(24GB) bfloat16 ~8GB → 안정 운영 가능
    3) 채팅 포맷: <|system|>/<|user|>/<|assistant|>
    4) AutoModelForCausalLM 사용 (순수 텍스트 모델, GEMMA와 다름)

  ⚠️ 모델 클래스 주의:
    GEMMA → AutoModelForImageTextToText (멀티모달 아키텍처)
    QWEN  → AutoModelForCausalLM       (순수 텍스트)
    PHI   → AutoModelForCausalLM       (순수 텍스트)
    LocalHFGenerator 내부에서 scenario 기준으로 자동 분기 처리됨.

▶ OPENAI — gpt-5-mini (기본값)
  선택 방법:
    - 환경변수 OPENAI_LLM_MODEL 로 지정 (기본값: gpt-5-mini)
      예) OPENAI_LLM_MODEL=gpt-4.1-mini python serving_main.py
    - gpt-5-mini: max_completion_tokens 파라미터 사용
    - gpt-4.1-mini: max_tokens 파라미터 사용

[LoRA 어댑터 적용 방법]
  파인튜닝 완료 후 환경변수로 어댑터 경로 지정:
    export GEMMA_ADAPTER_PATH=/mnt/gukrul/.../peft_output/gemma4-E4B/lora_adapter
    export QWEN_ADAPTER_PATH=/mnt/gukrul/.../peft_output/qwen25-7B/merged_model
  경로가 없거나 존재하지 않으면 베이스 모델 그대로 사용.
  Gemma: LoRA 어댑터 로드 (PeftModel)
  Qwen : merge_and_unload() 병합 모델 직접 로드

=======================================================================
[텍스트 생성 파라미터 설계 근거]
    temperature=0.1  : 입찰 공고 답변은 수치/날짜 정확도가 최우선 → 창의성 최소화
                       답변 다양성이 필요한 경우 0.2~0.3으로 조정
    top_p=0.9        : 하위 10% 확률 토큰 제거 → 뜬금없는 단어 방지
    max_new_tokens=512 : 입찰 답변은 길어야 3~4단락, 512토큰이면 충분
                       장문 필요 시 768~1024로 늘릴 것 (비용/레이턴시 증가)
    repetition_penalty=1.1 : 로컬 모델 반복 답변 방지 (API 모델엔 미적용)
"""

import os
import gc
import logging
from typing import List, Dict, Iterator, Optional

import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
_logger = logging.getLogger(__name__)

# ======================================================================
# 환경 감지
# ======================================================================
try:
    import google.colab; _ENV = 'colab'
except ImportError:
    _ENV = os.environ.get('BIDMATE_ENV', 'gcp')

# GPU 디바이스 선택 (retrieval_interface_F.py 와 동일 패턴)
if torch.cuda.is_available():            _DEVICE = 'cuda'
elif torch.backends.mps.is_available(): _DEVICE = 'mps'
else:                                    _DEVICE = 'cpu'

_logger.info(f'[Generation] ENV={_ENV} | DEVICE={_DEVICE}')


# ======================================================================
# 모델 설정 상수
# ======================================================================

# ⚠️ [교체 포인트 A] 시나리오별 로컬 모델 변경 시 이 값만 수정
_A1_MODEL_ID = 'google/gemma-4-E4B-it'           # GEMMA: Gemma 4 E4B
_A2_MODEL_ID = 'Qwen/Qwen2.5-7B-Instruct'        # QWEN : Qwen2.5 7B
_A3_MODEL_ID = 'microsoft/Phi-4-mini-instruct'    # PHI  : Phi-4-mini

_LOCAL_MODEL_MAP = {
    'GEMMA': _A1_MODEL_ID,
    'QWEN' : _A2_MODEL_ID,
    'PHI'  : _A3_MODEL_ID,
}

# LoRA 어댑터 경로 (파인튜닝 완료 후 환경변수로 지정)
# 경로가 None이거나 존재하지 않으면 베이스 모델만 로드.
# 예) export GEMMA_ADAPTER_PATH=/home/euijeong/.../lora_adapter
_GEMMA_ADAPTER_PATH = os.environ.get('GEMMA_ADAPTER_PATH', None)
_QWEN_ADAPTER_PATH  = os.environ.get('QWEN_ADAPTER_PATH',  None)
_PHI_ADAPTER_PATH   = os.environ.get('PHI_ADAPTER_PATH',   None)

_ADAPTER_PATH_MAP = {
    'GEMMA': _GEMMA_ADAPTER_PATH,
    'QWEN' : _QWEN_ADAPTER_PATH,
    'PHI'  : _PHI_ADAPTER_PATH,
}

_B_MODEL_ID  = os.environ.get('OPENAI_LLM_MODEL', 'gpt-5-mini')
# gpt-5-mini 는 max_completion_tokens, gpt-4.1-mini 는 max_tokens
_B_USE_COMPLETION_TOKENS = _B_MODEL_ID in ('gpt-5-mini', 'gpt-5.4-mini')

LLM_CONFIG = {
    'GEMMA' : {'type': 'local'},
    'QWEN'  : {'type': 'local'},
    'PHI'   : {'type': 'local'},
    'OPENAI': {'type': 'api'},
}


# T4(16GB) 기준 4-bit NF4 양자화 사용 여부
# ⚠️ A100(40GB+) 환경에서는 False로 변경하여 전체 bfloat16 정밀도 사용 가능
_QUANTIZE = False

# 텍스트 생성 파라미터
# ⚠️ 정확도 우선: temperature=0.05 / 다양성 필요: temperature=0.2~0.3
_LOCAL_GEN_PARAMS = {
    'max_new_tokens'    : 512,
    'temperature'       : 0.1,
    'top_p'             : 0.9,
    'do_sample'         : True,   # temperature > 0 이면 True 필수
    'repetition_penalty': 1.1,    # 로컬 모델 반복 방지
}

# API 모델 파라미터 (repetition_penalty 없음 — 대부분 API에서 미지원)
_API_GEN_PARAMS = {
    'max_tokens' : 512,
    'temperature': 1,
    'top_p'      : 1.0,
    
    
}

# 히스토리 최대 유지 턴 수 (user + assistant 쌍 기준)
# ⚠️ 128k 모델로 교체 시 5~10으로 늘릴 수 있음 (현재 4k 컨텍스트 가정)
_MAX_HISTORY_TURNS = 3  # 최근 3턴 (메시지 6개) 유지


# ======================================================================
# 프롬프트 포맷 (retrieval_interface_F.py 의 _PROMPT_TEMPLATES 와 동기화)
# ======================================================================
# generate() 에서 이미 완성된 context+query 프롬프트를 받는 구조이므로,
# 여기서는 히스토리를 프롬프트에 삽입하는 로직만 담당한다.
#
# ⚠️ 프롬프트 포맷 교체 시:
#   - Gemma 계열 → <start_of_turn>user ... <start_of_turn>model 유지
#   - LLaMA 계열 → <|begin_of_text|><|start_header_id|>... 유지 (A-2 포맷)
#   - API 모델   → messages 배열 포맷 유지 (변경 불필요)
def _inject_history_into_prompt(prompt: str, history: list, scenario: str) -> str:
    """
    로컬 모델용 완성 프롬프트에 히스토리 블록을 삽입.

    retrieval_interface_F.py 의 build_prompt() 가 반환한 프롬프트 문자열에서
    [질문] 태그 앞에 [이전 대화] 블록을 끼워 넣는 방식.

    이 방식을 선택한 이유:
      - apply_chat_template() 재호출 시 포맷이 이중 래핑되는 문제 방지
        (retrieval_interface_F.py 가 이미 Gemma/LLaMA 포맷으로 완성한 문자열이므로
         tokenizer.apply_chat_template을 다시 쓰면 <start_of_turn>이 두 번 들어감)
      - 히스토리를 [이전 대화] 인라인 삽입하면 컨텍스트 윈도우 내 위치 제어 가능

    ⚠️ 모델 교체 시: [질문] 태그가 바뀌면 이 함수의 split 기준도 수정 필요
    """
    if not history:
        return prompt  # 히스토리 없으면 원본 그대로 반환

    # 히스토리 → 텍스트 직렬화
    trimmed = history[-(2 * _MAX_HISTORY_TURNS):]
    lines = []
    for h in trimmed:
        role = '사용자' if h['role'] == 'user' else 'AI'
        lines.append(f'{role}: {h["content"]}')
    history_block = '[이전 대화]\n' + '\n'.join(lines) + '\n\n'

    # [질문] 태그 앞에 히스토리 삽입
    # Gemma(A-1): "[질문]\n{query}\n" 패턴
    # LLaMA(A-2): "[질문]\n{query}\n" 패턴 (동일)
    if '[질문]' in prompt:
        return prompt.replace('[질문]', history_block + '[질문]', 1)

    # fallback: 프롬프트 마지막 라인 직전에 삽입
    _logger.warning('[Generation] [질문] 태그를 찾을 수 없어 히스토리를 프롬프트 끝에 삽입')
    return prompt.rstrip('\n') + '\n\n' + history_block


# ======================================================================
# 응답 후처리
# ======================================================================
def _postprocess(raw_text: str) -> str:
    """
    로컬 모델 출력에서 특수 토큰 제거 및 앞뒤 공백 정리.
    """
    if not raw_text:
        return ''

    # Gemma 4 계열 special token
    for tok in ['<end_of_turn>', '<eos>', '<bos>', '<pad>', '<start_of_turn>model']:
        raw_text = raw_text.replace(tok, '')

    # Qwen3.5 계열 special token
    for tok in ['<|im_end|>', '<|im_start|>', '<|endoftext|>']:
        raw_text = raw_text.replace(tok, '')

    # Phi-4-mini 계열 special token
    for tok in ['<|end|>', '<|system|>', '<|user|>', '<|assistant|>']:
        raw_text = raw_text.replace(tok, '')

    # LLaMA 계열 special token (하위호환)
    for tok in ['<|eot_id|>', '<|end_of_text|>', '<|begin_of_text|>',
                '<|start_header_id|>', '<|end_header_id|>']:
        raw_text = raw_text.replace(tok, '')

    return raw_text.strip()


# ======================================================================
# 추상 기반 클래스
# ======================================================================
class BidMateGenerator:
    """
    모든 Generator의 추상 기반.
    generate() 는 문자열을 반환하고,
    generate_stream() 은 스트리밍 Iterator를 반환한다.

    serving_main.py 에서는 generate() 를 사용한다.
    FastAPI SSE 서빙 시 generate_stream() 으로 교체 가능.
    """

    def generate(self, prompt_dict: dict, history: List[Dict]) -> str:
        """
        Parameters
        ----------
        prompt_dict : retrieval_interface_F.build_prompt() 반환값
            {
              'scenario': str,
              'prompt'  : str | None,   # 로컬 모델용 완성 프롬프트
              'system'  : str | None,   # API 모델용 시스템 메시지
              'user'    : str | None,   # API 모델용 유저 메시지
            }
        history : [{'role': 'user'|'assistant', 'content': str}, ...]
                  serving_main.py 에서 관리하는 세션 히스토리
        Returns : 답변 문자열
        """
        raise NotImplementedError

    def generate_stream(self, prompt_dict: dict, history: List[Dict]) -> Iterator[str]:
        """
        스트리밍 버전. 기본 구현은 generate() 결과를 단일 청크로 yield.
        실시간 스트리밍이 필요하면 하위 클래스에서 오버라이드.
        ⚠️ FastAPI StreamingResponse / SSE 연동 시 이 메서드 사용
        """
        yield self.generate(prompt_dict, history)

    def release(self):
        """GPU 메모리 해제. 시나리오 전환 전 호출. 하위 클래스에서 오버라이드."""
        pass


# ======================================================================
# 시나리오 A: 로컬 HuggingFace 모델 (GCP GPU 서버용)
# ======================================================================
class LocalHFGenerator(BidMateGenerator):
    """
    HuggingFace 로컬 모델 기반 Generator.

    초기화 비용이 크므로 (모델 로드 수십 초) get_generator() 로 생성한
    인스턴스를 serving_main.py 에서 싱글턴으로 재사용해야 한다.
    """

    def __init__(self, scenario: str):
        """
        Parameters
        ----------
        scenario : 'GEMMA' | 'QWEN' | 'PHI'
        """
        from transformers import AutoTokenizer, AutoModelForImageTextToText, BitsAndBytesConfig

        self.scenario = scenario

        # scenario별 모델 ID 선택
        if scenario not in _LOCAL_MODEL_MAP:
            raise ValueError(
                f"LocalHFGenerator: 지원하지 않는 scenario='{scenario}'. "
                f"가능한 값: {list(_LOCAL_MODEL_MAP.keys())}"
            )
        model_id = _LOCAL_MODEL_MAP[scenario]

        _logger.info(f'[Generation] 로컬 모델 로드 시작 | {model_id} | 양자화={_QUANTIZE}')

        # 토크나이저 로드
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
        self._tokenizer.padding_side = 'left'
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # 모델 로드 (양자화 여부 분기)
        if _QUANTIZE and _DEVICE == 'cuda':
            # 4-bit NF4 양자화: T4(16GB) 기준 Gemma-3-4b ~5GB
            # ⚠️ A100(40GB+) 에서는 _QUANTIZE=False 전환 권장 (정확도↑)
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit              = True,
                bnb_4bit_compute_dtype    = torch.bfloat16,  # 연산 dtype (Gemma 공식 권장)
                bnb_4bit_quant_type       = 'nf4',           # NF4 양자화 (QLoRA 논문 기준)
                bnb_4bit_use_double_quant = True,            # 이중 양자화 → 추가 메모리 절약
            )
            # GEMMA만 멀티모달 아키텍처, QWEN/PHI는 CausalLM
            if scenario == 'GEMMA':
                self._model = AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    quantization_config = bnb_cfg,
                    device_map          = 'auto',
                    trust_remote_code   = False,
                )
            else:
                from transformers import AutoModelForCausalLM
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config = bnb_cfg,
                    device_map          = 'auto',
                    trust_remote_code   = False,
                )
        else:
            # 양자화 없이 bfloat16 로드
            if scenario == 'GEMMA':
                self._model = AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    torch_dtype       = torch.bfloat16,
                    device_map        = 'auto',
                    trust_remote_code = False,
                )
            else:
                from transformers import AutoModelForCausalLM
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype       = torch.bfloat16,
                    device_map        = 'auto',
                    trust_remote_code = False,
                )

        # LoRA 어댑터 로드 (파인튜닝 가중치 적용)
        # 환경변수로 지정: GEMMA_ADAPTER_PATH / QWEN_ADAPTER_PATH / PHI_ADAPTER_PATH
        # 경로가 없거나 존재하지 않으면 베이스 모델 그대로 사용.
        adapter_path = _ADAPTER_PATH_MAP.get(scenario)
        if adapter_path and Path(adapter_path).exists():
            try:
                from peft import PeftModel
                _logger.info(f'[Generation] LoRA 어댑터 로드 | {adapter_path}')
                self._model = PeftModel.from_pretrained(
                    self._model,
                    adapter_path,
                    torch_dtype = torch.bfloat16,
                )
                _logger.info(f'[Generation] ✅ LoRA 어댑터 적용 완료')
            except Exception as e:
                _logger.warning(f'[Generation] 어댑터 로드 실패 → 베이스 모델 사용: {e}')
        elif adapter_path:
            _logger.warning(f'[Generation] 어댑터 경로 없음 → 베이스 모델 사용: {adapter_path}')

        self._model.eval()  # 추론 전용 모드 (Dropout 비활성화, 성능 고정)
        _logger.info(f'[Generation] ✅ 로컬 모델 로드 완료 | {model_id}')

    def generate(self, prompt_dict: dict, history: List[Dict]) -> str:
        """
        prompt_dict['prompt'] (완성된 문자열)을 받아 히스토리 삽입 후 생성.

        ⚠️ 버그 방지 포인트:
          - prompt_dict['prompt'] 는 retrieval_interface_F.build_prompt() 가
            반환한 Gemma/LLaMA 포맷의 완성 문자열이다.
          - 여기서 apply_chat_template() 을 다시 쓰면 포맷이 이중 래핑됨.
            → _inject_history_into_prompt() 로 직접 텍스트 삽입 방식 사용.
        """
        # 완성 프롬프트에 히스토리 삽입
        prompt = _inject_history_into_prompt(
            prompt   = prompt_dict['prompt'],
            history  = history,
            scenario = self.scenario,
        )

        # 토크나이징
        # ⚠️ max_length: 컨텍스트(~1,500) + 히스토리(~300) + 질문(~100) = 약 2,000
        #    여유 있게 3,072로 설정. 긴 문서 청크 사용 시 4,096으로 늘릴 것
        inputs = self._tokenizer(
            prompt,
            return_tensors = 'pt',
            truncation     = True,
            max_length     = 3072,
            padding        = False,
        ).to(_DEVICE)

        # stop token 설정 — 모델별 분기
        eos_ids = [self._tokenizer.eos_token_id]
        if self.scenario == 'GEMMA':
            # Gemma 4: <end_of_turn> 토큰 추가
            eot = self._tokenizer.convert_tokens_to_ids('<end_of_turn>')
            if eot and eot != self._tokenizer.unk_token_id:
                eos_ids.append(eot)
        elif self.scenario == 'QWEN':
            # Qwen3.5: <|im_end|> 토큰 추가
            im_end = self._tokenizer.convert_tokens_to_ids('<|im_end|>')
            if im_end and im_end != self._tokenizer.unk_token_id:
                eos_ids.append(im_end)
        elif self.scenario == 'PHI':
            # Phi-4-mini: <|end|> 토큰 추가
            end = self._tokenizer.convert_tokens_to_ids('<|end|>')
            if end and end != self._tokenizer.unk_token_id:
                eos_ids.append(end)

        with torch.no_grad():  # 그래디언트 계산 비활성화 → 메모리 절약, 속도↑
            output_ids = self._model.generate(
                input_ids          = inputs['input_ids'],
                attention_mask     = inputs['attention_mask'],
                # ⚠️ _LOCAL_GEN_PARAMS 딕셔너리에서 일괄 관리
                max_new_tokens     = _LOCAL_GEN_PARAMS['max_new_tokens'],
                temperature        = _LOCAL_GEN_PARAMS['temperature'],
                top_p              = _LOCAL_GEN_PARAMS['top_p'],
                do_sample          = _LOCAL_GEN_PARAMS['do_sample'],
                repetition_penalty = _LOCAL_GEN_PARAMS['repetition_penalty'],
                eos_token_id       = eos_ids,
                pad_token_id       = self._tokenizer.pad_token_id,
            )

        # 입력 토큰 제거 → 생성 부분만 디코딩
        input_len = inputs['input_ids'].shape[1]
        gen_ids   = output_ids[0][input_len:]
        raw_text  = self._tokenizer.decode(gen_ids, skip_special_tokens=True)

        return _postprocess(raw_text)

    def generate_stream(self, prompt_dict: dict, history: List[Dict]) -> Iterator[str]:
        """
        TextIteratorStreamer 기반 실시간 스트리밍.
        ⚠️ FastAPI StreamingResponse 연동 시 이 메서드를 사용.
        """
        from transformers import TextIteratorStreamer
        from threading import Thread

        prompt = _inject_history_into_prompt(
            prompt   = prompt_dict['prompt'],
            history  = history,
            scenario = self.scenario,
        )
        inputs = self._tokenizer(
            prompt, return_tensors='pt', truncation=True,
            max_length=3072, padding=False,
        ).to(_DEVICE)

        # stop token 설정 — 모델별 분기
        eos_ids = [self._tokenizer.eos_token_id]
        if self.scenario == 'GEMMA':
            eot = self._tokenizer.convert_tokens_to_ids('<end_of_turn>')
            if eot and eot != self._tokenizer.unk_token_id:
                eos_ids.append(eot)
        elif self.scenario == 'QWEN':
            im_end = self._tokenizer.convert_tokens_to_ids('<|im_end|>')
            if im_end and im_end != self._tokenizer.unk_token_id:
                eos_ids.append(im_end)
        elif self.scenario == 'PHI':
            end = self._tokenizer.convert_tokens_to_ids('<|end|>')
            if end and end != self._tokenizer.unk_token_id:
                eos_ids.append(end)

        # skip_special_tokens=True: 스트리밍 도중 stop token 노출 방지
        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            input_ids          = inputs['input_ids'],
            attention_mask     = inputs['attention_mask'],
            max_new_tokens     = _LOCAL_GEN_PARAMS['max_new_tokens'],
            temperature        = _LOCAL_GEN_PARAMS['temperature'],
            top_p              = _LOCAL_GEN_PARAMS['top_p'],
            do_sample          = _LOCAL_GEN_PARAMS['do_sample'],
            repetition_penalty = _LOCAL_GEN_PARAMS['repetition_penalty'],
            eos_token_id       = eos_ids,
            pad_token_id       = self._tokenizer.pad_token_id,
            streamer           = streamer,
        )
        # 별도 스레드에서 generate (블로킹 방지)
        thread = Thread(target=self._model.generate, kwargs=gen_kwargs)
        thread.start()

        for token in streamer:
            yield token

    def release(self):
        """GPU 메모리 해제. 시나리오 전환 전 호출."""
        if hasattr(self, '_model') and self._model is not None:
            del self._model
        if hasattr(self, '_tokenizer') and self._tokenizer is not None:
            del self._tokenizer
        self._model, self._tokenizer = None, None
        gc.collect()
        torch.cuda.empty_cache()
        free_gb = torch.cuda.mem_get_info()[0] / 1024**3 if torch.cuda.is_available() else 0
        _logger.info(f'[Generation] GPU 해제 완료 | 여유: {free_gb:.1f}GB')


# ======================================================================
# 시나리오 B: API 기반 LLM
# ======================================================================
class APIGenerator(BidMateGenerator):
    """
    OpenAI 호환 API 기반 Generator.

    ⚠️ Claude/Gemini 전환 시: _generate() 내 분기 블록 주석 해제
    """

    def __init__(self, scenario: str = 'B'):
        from openai import OpenAI

        self.scenario = scenario
        self._model_id = _B_MODEL_ID
        # OPENAI_API_KEY 환경변수 자동 참조
        # ⚠️ Azure OpenAI 사용 시: OpenAI(base_url=..., api_key=...) 로 변경
        self._client = OpenAI()
        _logger.info(f'[Generation] OpenAI 클라이언트 초기화 | 모델={self._model_id} | completion_tokens={_B_USE_COMPLETION_TOKENS}')

    def _build_api_messages(self, prompt_dict: dict, history: List[Dict]) -> list:
        """
        prompt_dict 의 system/user 필드와 히스토리를 OpenAI messages 배열로 조합.

        ⚠️ 버그 방지 포인트 (Gemini 버전의 핵심 버그):
          시나리오 B 의 build_prompt() 는 {'system': ..., 'user': ...} 를 반환한다.
          Gemini 버전은 prompt_dict.get('user') 만 메시지에 넣고 'system' 을 누락시켰음.
          → 반드시 system 메시지를 messages[0] 에 추가해야 함.
        """
        system_msg = prompt_dict.get('system') or ''
        user_msg   = prompt_dict.get('user')   or ''

        if not user_msg:
            # fallback: 완성 프롬프트가 'prompt' 필드에 있는 경우 (A 시나리오 prompt_dict 혼용 방지)
            _logger.warning('[Generation] prompt_dict에 user 필드 없음. prompt 필드로 fallback.')
            user_msg = prompt_dict.get('prompt', '')

        # 히스토리 자르기
        trimmed = history[-(2 * _MAX_HISTORY_TURNS):]

        messages = []
        if system_msg:
            messages.append({'role': 'system', 'content': system_msg})
        # 히스토리 (system 다음, 현재 user 이전)
        messages.extend(trimmed)
        messages.append({'role': 'user', 'content': user_msg})

        return messages

    def _token_param(self) -> dict:
        """
        gpt-5-mini 는 max_completion_tokens 사용, 구버전은 max_tokens 사용.
        _B_USE_COMPLETION_TOKENS 플래그로 자동 분기.
        """
        if _B_USE_COMPLETION_TOKENS:
            return {'max_completion_tokens': _API_GEN_PARAMS['max_tokens']}
        return {'max_tokens': _API_GEN_PARAMS['max_tokens']}

    def generate(self, prompt_dict: dict, history: List[Dict]) -> str:
        messages = self._build_api_messages(prompt_dict, history)

        response = self._client.chat.completions.create(
            model       = self._model_id,
            messages    = messages,
            # ⚠️ gpt-5-mini: max_completion_tokens / gpt-4.1-mini: max_tokens
            # _token_param() 이 자동 분기
            temperature = _API_GEN_PARAMS['temperature'],
            top_p       = _API_GEN_PARAMS['top_p'],
            **self._token_param(),
        )
        return response.choices[0].message.content.strip()

    def generate_stream(self, prompt_dict: dict, history: List[Dict]) -> Iterator[str]:
        """
        OpenAI streaming API 사용.
        ⚠️ FastAPI StreamingResponse 연동 시 이 메서드 사용.
        """
        messages = self._build_api_messages(prompt_dict, history)

        stream = self._client.chat.completions.create(
            model       = self._model_id,
            messages    = messages,
            temperature = _API_GEN_PARAMS['temperature'],
            top_p       = _API_GEN_PARAMS['top_p'],
            **self._token_param(),
            stream      = True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ⚠️ Claude(Anthropic) API 전환 시 아래 블록 주석 해제 후 generate() 에서 호출
    # def _generate_anthropic(self, messages: list) -> str:
    #     import anthropic
    #     client = anthropic.Anthropic()   # ANTHROPIC_API_KEY 환경변수 참조
    #     system = next((m['content'] for m in messages if m['role'] == 'system'), '')
    #     user_msgs = [m for m in messages if m['role'] != 'system']
    #     res = client.messages.create(
    #         model      = 'claude-haiku-4-5-20251001',
    #         max_tokens = _API_GEN_PARAMS['max_tokens'],
    #         system     = system,
    #         messages   = user_msgs,
    #         temperature= _API_GEN_PARAMS['temperature'],
    #     )
    #     return res.content[0].text.strip()

    # ⚠️ Gemini API 전환 시 아래 블록 주석 해제
    # def _generate_gemini(self, messages: list) -> str:
    #     import google.generativeai as genai
    #     genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
    #     model = genai.GenerativeModel('gemini-2.0-flash')
    #     # OpenAI messages → Gemini contents 변환 필요
    #     contents = [{'role': m['role'], 'parts': [m['content']]}
    #                 for m in messages if m['role'] != 'system']
    #     res = model.generate_content(contents)
    #     return res.text.strip()


# ======================================================================
# OpenRouter Generator (팀장 요청 반영)
# ======================================================================
# OpenRouter: HuggingFace 오픈소스 모델을 API 형태로 제공하는 허브.
# 로컬 GPU 없이도 Gemma·Qwen·Phi 등을 테스트할 수 있어
# "최적 모델 탐색 → 이후 로컬 LoRA 파인튜닝" 전략의 1단계로 활용.
#
# 무료 모델은 model_id 끝에 :free 를 붙여야 과금 방지됨.
# ⚠️ 무료 모델은 RPM(분당 요청 수)과 context 길이에 제한이 있음.
#    프로덕션 트래픽에는 사용 불가, 탐색/평가 전용.


# # ⚠️ [교체 포인트 OR] OpenRouter 모델 변경 시 아래 딕셔너리만 수정
# _OR_MODEL_MAP = {
#     # Gemma 3 4B — 팀 조건 "Gemma 4 이상" 충족, 무료 제공 확인됨
#     # ⚠️ gemma-4-9b-it 가 OpenRouter에 무료로 올라오면 해당 ID로 교체
#     'OR-GEMMA' : 'google/gemma-3-4b-it:free',
#     # Qwen 3 14B — 한국어 포함 다국어 우수, 논리 추론 강점
#     # 팀 대화에서 "Qwen은 학습 필요" 언급 → 파인튜닝 후보 1순위
#     'OR-QWEN'  : 'qwen/qwen3-235b-a22b:free',
#     # Phi-4 Mini — 강사님 추천, 128k 컨텍스트, M2 16GB 로컬 구동 가능 체급
#     'OR-PHI'   : 'microsoft/phi-4-mini-instruct:free',
# }
# 수정
_OR_MODEL_MAP = {
    'OR-GEMMA': 'google/gemma-3-4b-it:free',
    'OR-QWEN' : 'qwen/qwen3-235b-a22b:free',
    'OR-PHI'  : 'microsoft/phi-4-mini-instruct:free',
    'QWEN'    : 'qwen/qwen3-235b-a22b:free',
    'PHI'     : 'microsoft/phi-4-mini-instruct:free',
}


# OpenRouter 공식 권장 헤더 (로깅 및 무료 티어 집계에 사용됨)
_OR_HEADERS = {
    'HTTP-Referer': 'https://github.com/BidMate',
    'X-Title'     : 'BidMate RAG System',
}


class OpenRouterGenerator(BidMateGenerator):
    """
    OpenRouter API 기반 Generator.
    APIGenerator 와 동일한 OpenAI 호환 클라이언트를 사용하되,
    base_url 과 헤더만 다르다.

    사용 예:
        export OPENROUTER_API_KEY="sk-or-v1-..."
        export EMBED_SCENARIO="OR-GEMMA"
        python serving_main.py
    """

    def __init__(self, scenario: str):
        from openai import OpenAI

        # ⚠️ OPENROUTER_API_KEY 환경변수 필수
        api_key = os.environ.get('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError(
                'OPENROUTER_API_KEY 환경변수가 설정되지 않았습니다.\n'
                '  export OPENROUTER_API_KEY="sk-or-v1-..."'
            )

        self.scenario  = scenario
        self._model_id = _OR_MODEL_MAP.get(scenario)
        if not self._model_id:
            raise ValueError(
                f"OR 시나리오 '{scenario}'가 _OR_MODEL_MAP에 없음. "
                f"가능한 값: {list(_OR_MODEL_MAP.keys())}"
            )

        # OpenRouter 엔드포인트 — OpenAI 패키지 그대로 사용 가능
        self._client = OpenAI(
            base_url = 'https://openrouter.ai/api/v1',
            api_key  = api_key,
        )
        _logger.info(f'[Generation] OpenRouter 초기화 | scenario={scenario} | model={self._model_id}')

    def _build_messages(self, prompt_dict: dict, history: List[Dict]) -> list:
        """
        APIGenerator._build_api_messages() 와 동일 로직.
        시나리오 B 와 OR 모두 build_prompt() 의 B 포맷(system/user)을 사용.
        """
        system_msg = prompt_dict.get('system') or ''
        user_msg   = prompt_dict.get('user')   or prompt_dict.get('prompt', '')

        trimmed  = history[-(2 * _MAX_HISTORY_TURNS):]
        messages = []
        if system_msg:
            messages.append({'role': 'system', 'content': system_msg})
        messages.extend(trimmed)
        messages.append({'role': 'user', 'content': user_msg})
        return messages

    def generate(self, prompt_dict: dict, history: List[Dict]) -> str:
        """
        문자열 반환 (serving_main.py 의 chat() 인터페이스와 일치).
        ⚠️ 무료 모델은 rate limit 에 걸릴 수 있음 → except 로 폴백 처리.
        """
        messages = self._build_messages(prompt_dict, history)
        try:
            response = self._client.chat.completions.create(
                model           = self._model_id,
                messages        = messages,
                temperature     = _API_GEN_PARAMS['temperature'],
                top_p           = _API_GEN_PARAMS['top_p'],
                max_tokens      = _API_GEN_PARAMS['max_tokens'],
                extra_headers   = _OR_HEADERS,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # ⚠️ 무료 모델 rate limit 또는 일시 장애 시 빈 문자열 반환
            #    프로덕션이라면 재시도(retry) 로직 추가 권장
            _logger.error(f'[OpenRouter] 호출 실패: {e}')
            return f'[OpenRouter 오류: {e}]'

    def generate_stream(self, prompt_dict: dict, history: List[Dict]) -> Iterator[str]:
        """
        FastAPI StreamingResponse 연동용.
        OpenRouter 도 stream=True 지원.
        """
        messages = self._build_messages(prompt_dict, history)
        try:
            stream = self._client.chat.completions.create(
                model         = self._model_id,
                messages      = messages,
                temperature   = _API_GEN_PARAMS['temperature'],
                top_p         = _API_GEN_PARAMS['top_p'],
                max_tokens    = _API_GEN_PARAMS['max_tokens'],
                stream        = True,
                extra_headers = _OR_HEADERS,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            _logger.error(f'[OpenRouter] 스트리밍 실패: {e}')
            yield f'[오류: {e}]'


# ======================================================================
# 팩토리 함수
# ======================================================================
# def get_generator(scenario: str) -> BidMateGenerator:
#     """
#     시나리오에 맞는 Generator 인스턴스를 반환하는 팩토리 함수.
#     serving_main.py / eval 파일에서 이 함수만 호출한다.

#     Parameters
#     ----------
#     scenario : 'A-1' | 'A-2' | 'B' | 'OR-GEMMA' | 'OR-QWEN' | 'OR-PHI'

#     ⚠️ 새 시나리오 추가 방법:
#         - OpenRouter 모델: _OR_MODEL_MAP 에 항목 추가 후 'OR-XXX' 로 호출
#         - 로컬 모델 교체: _A1_MODEL_ID / _A2_MODEL_ID 상수만 변경
#         - API 모델 교체: _B_MODEL_ID 상수만 변경
#     """
#     _logger.info(f'[Generation] get_generator 호출 | scenario={scenario}')

#     if scenario in ('A-1', 'A-2'):
#         # GCP L4 로컬 모델 (HuggingFace 직접 로드)
#         return LocalHFGenerator(scenario=scenario)

#     elif scenario == 'B':
#         # OpenAI 공식 API
#         # ⚠️ Ollama 로컬 서빙 시: OPENAI_BASE_URL=http://localhost:11434/v1 설정
#         #    → APIGenerator 는 OPENAI_BASE_URL 을 자동 참조하도록 확장 가능
#         return APIGenerator(scenario=scenario)

#     elif scenario.startswith('OR-'):
#         # OpenRouter 오픈소스 모델 (팀장 요청 반영)
#         # Mac M2 16GB 로컬 환경에서 GPU 없이 테스트 가능
#         return OpenRouterGenerator(scenario=scenario)

#     else:
#         raise ValueError(
#             f"정의되지 않은 시나리오: '{scenario}'.\n"
#             f"가능한 값: 'A-1', 'A-2', 'B', {list(_OR_MODEL_MAP.keys())}"
#         )
    
# ======================================================================
# Self-Correction (생성 후 검토 루프)
# ======================================================================
# 모델이 생성한 답변이 컨텍스트에 근거하는지 스스로 검토하여 재생성.
# 로컬 모델 기준 호출 2배 → 속도 저하 있음. 필요 시 활성화.
_SELF_CORRECTION_ENABLED = os.environ.get('SELF_CORRECTION', 'false').lower() == 'true'

_SELF_CORRECTION_PROMPT = (
    "당신은 AI 답변 검토자입니다.\n"
    "아래 [답변]이 [참고 문서]에만 근거하는지 검토하고,\n"
    "근거 없는 내용이 있으면 제거하거나 수정하여 최종 답변을 작성하세요.\n"
    "근거 없는 내용이 없으면 [답변]을 그대로 출력하세요.\n\n"
    "[참고 문서]\n{context}\n\n"
    "[답변]\n{answer}\n\n"
    "최종 답변:"
)


def apply_self_correction(answer: str, context: str, generator) -> str:
    """
    생성된 답변을 컨텍스트 기준으로 자기 검토 후 재생성.

    Parameters
    ----------
    answer    : 1차 생성 답변
    context   : retriever 컨텍스트
    generator : BidMateGenerator 인스턴스 (generate 메서드 사용)

    Returns
    -------
    str : 검토 후 최종 답변
    """
    if not _SELF_CORRECTION_ENABLED:
        return answer

    correction_prompt = _SELF_CORRECTION_PROMPT.format(
        context=context, answer=answer
    )
    prompt_dict = {
        'scenario': generator.scenario if hasattr(generator, 'scenario') else 'OPENAI',
        'prompt'  : correction_prompt,
        'system'  : None,
        'user'    : None,
    }
    try:
        corrected = generator.generate(prompt_dict=prompt_dict, history=[])
        _logger.info('[Self-Correction] 검토 완료')
        return corrected if corrected else answer
    except Exception as e:
        _logger.warning(f'[Self-Correction] 실패 → 원본 답변 사용: {e}')
        return answer


# 수정
def get_generator(scenario: str) -> BidMateGenerator:
    """
    시나리오에 맞는 Generator 인스턴스 반환.

    Parameters
    ----------
    scenario : 'KURE_GEMMA' | 'KURE_QWEN' | 'KURE_PHI' | 'KURE_OPENAI'
               'KOE5_GEMMA' | 'KOE5_QWEN' | ... (임베딩_LLM 조합 12개)

    OpenAI 모델 선택:
        OPENAI_LLM_MODEL=gpt-5-mini (기본값) → max_completion_tokens 사용
        OPENAI_LLM_MODEL=gpt-4.1-mini          → max_tokens 사용
    """
    _logger.info(f'[Generation] get_generator 호출 | scenario={scenario}')
    llm_key = scenario.split('_')[1] if '_' in scenario else scenario

    if llm_key in ('GEMMA', 'QWEN', 'PHI'):
        return LocalHFGenerator(scenario=llm_key)
    elif llm_key == 'OPENAI':
        return APIGenerator(scenario='OPENAI')
    else:
        raise ValueError(
            f"정의되지 않은 시나리오: '{scenario}'.\n"
            f"가능한 값: KURE/KOE5/SMALL + GEMMA/QWEN/PHI/OPENAI 조합\n"
            f"예) 'KURE_GEMMA', 'SMALL_OPENAI'"
        )
