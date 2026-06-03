"""
입찰메이트 RAG — End-to-End Generation 비교 평가 (A/B/D/E타입 통합)
=================================================================
담당 : Generation / Retrieval 파트 (한의정)

[설명]
- 579개 eval 셋 중 A/B/D/E타입 (총 518개) 추출.
- C타입(히스토리 복합 쿼리)은 eval_e2e_ctype.py 에서 별도 처리.
- Retrieval Context는 extract_contexts.py 로 사전 추출된 CSV 재사용.
  → get_context() 재호출 없음, 시간 대폭 절약.
- GEMMA 모델 로드를 1회만 수행하여 시간 절약.
- 결과를 CSV로 저장 → eval_quant_judge_dual.py 에서 사용.

[사전 조건]
  extract_contexts.py 를 먼저 실행하여 eval_contexts_abde.csv 생성 필요.

[시나리오 지정 방법]
  환경변수로 비교할 두 시나리오를 지정 (임베딩_LLM 형태):
    SCENARIO_A=KURE_GEMMA   (기본값)
    SCENARIO_B=SMALL_OPENAI (기본값)
  예) SCENARIO_A=KURE_GEMMA SCENARIO_B=SMALL_OPENAI python eval_e2e_abde.py

[OpenAI 모델 선택]
  OPENAI_LLM_MODEL=gpt-5-mini (기본값) — max_completion_tokens 사용
"""

import os
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from retrieval_interface_F import build_prompt
from generation_interface import get_generator

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
_logger = logging.getLogger(__name__)

# ======================================================================
# 경로 설정
# ======================================================================
_RESULT_DIR = Path('/mnt/gukrul/dataset/eval_results/generation')
_RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 임베딩 모델 환경변수 (extract_contexts.py 와 일치해야 함)
_EMBED         = os.environ.get('EMBED', 'KURE').lower()
_CONTEXTS_PATH = _RESULT_DIR / f'eval_contexts_abde_{_EMBED}.csv'
_SAVE_PATH_TPL = str(_RESULT_DIR / 'e2e_abde_{embed}_{scenario_a}_comparison.csv')
_FT_SUFFIX = '_FT' if (os.environ.get('PHI_ADAPTER_PATH') or os.environ.get('GEMMA_ADAPTER_PATH') or os.environ.get('QWEN_ADAPTER_PATH')) else ''

_RUN_SEQUENTIAL = True


# ======================================================================
# 단일 시나리오 생성 루프
# ======================================================================
def _generate_answers(generator, scenario: str, contexts: pd.DataFrame) -> list:
    """
    contexts CSV 의 각 행에 대해 Generation 수행.
    Retrieval은 이미 완료된 context 텍스트를 재사용.

    Returns: [{'question', 'type', 'difficulty', 'history', 'retrieved_context', 'answer'}, ...]
    """
    records = []
    for _, row in tqdm(contexts.iterrows(), total=len(contexts),
                       desc=f'[{scenario}] 생성 중'):
        query        = row['question']
        context_text = row['retrieved_context']
        history      = []  # A/B/D/E타입은 히스토리 없음

        prompt_dict = build_prompt(
            query    = query,
            context  = context_text,
            scenario = scenario,
        )

        answer = None
        for _retry in range(3):
            try:
                answer = generator.generate(prompt_dict=prompt_dict, history=history)
                if answer and '생성 오류' not in answer:
                    break
            except Exception as e:
                _logger.warning(f'Generation 재시도 {_retry+1}/3: {e}')
                import time; time.sleep(2 ** _retry)
        if not answer:
            answer = f'[생성 오류: 최대 재시도 초과]'

        records.append({
            'question'         : query,
            'type'             : row['type'],
            'difficulty'       : row.get('difficulty', ''),
            'history'          : '[]',
            'retrieved_context': context_text,
            'answer'           : answer,
        })

    return records


# ======================================================================
# 메인 비교 평가 실행
# ======================================================================
def run_e2e_comparison():
    SCENARIO_A = os.environ.get('SCENARIO_A', 'KURE_GEMMA')
    SCENARIO_B = os.environ.get('SCENARIO_B', 'SMALL_OPENAI')
    col_a = 'ans_' + SCENARIO_A.lower().replace('-', '_')
    col_b = 'ans_' + SCENARIO_B.lower().replace('-', '_')

    _SAVE_PATH = Path(_SAVE_PATH_TPL.format(embed=_EMBED, scenario_a=SCENARIO_A.lower()))
    _logger.info(f'[E2E-ABDE] SCENARIO_A={SCENARIO_A} | SCENARIO_B={SCENARIO_B}')

    # ── Step 1. 사전 추출된 Context 로드 ────────────────────────────
    if not _CONTEXTS_PATH.exists():
        raise FileNotFoundError(
            f'{_CONTEXTS_PATH} 없음. extract_contexts.py 먼저 실행하세요.'
        )
    contexts = pd.read_csv(_CONTEXTS_PATH)
    _logger.info(
        f'[E2E-ABDE] Context 로드 완료: {len(contexts)}개 '
        f'(A={len(contexts[contexts["type"]=="A"])}, '
        f'B={len(contexts[contexts["type"]=="B"])}, '
        f'D={len(contexts[contexts["type"]=="D"])}, '
        f'E={len(contexts[contexts["type"]=="E"])})'
    )

    # ── Step 2. 모델별 생성 ─────────────────────────────────────────
    if _RUN_SEQUENTIAL:
        _logger.info(f'[E2E-ABDE] 순차 실행 모드 | A={SCENARIO_A}')

        _ft = '_FT' if (os.environ.get('PHI_ADAPTER_PATH') or os.environ.get('GEMMA_ADAPTER_PATH') or os.environ.get('QWEN_ADAPTER_PATH')) else ''
        _mid_a = _RESULT_DIR / f'e2e_abde_{_EMBED}_mid_{SCENARIO_A}{_ft}.csv'
        if _mid_a.exists():
            _logger.info(f'[E2E-ABDE] 중간 파일 재사용: {_mid_a}')
            records_a = pd.read_csv(_mid_a).to_dict('records')
        else:
            gen_a     = get_generator(SCENARIO_A)
            records_a = _generate_answers(gen_a, scenario=SCENARIO_A, contexts=contexts)
            gen_a.release()
            pd.DataFrame(records_a).rename(columns={'answer': col_a}).to_csv(
                _mid_a, index=False, encoding='utf-8-sig')
            _logger.info(f'[E2E-ABDE] 중간 저장 완료: {_mid_a}')

        _mid_b = _RESULT_DIR / f'e2e_abde_{_EMBED}_mid_{SCENARIO_B}.csv'
        if _mid_b.exists():
            _logger.info(f'[E2E-ABDE] 중간 파일 재사용: {_mid_b}')
            records_b = pd.read_csv(_mid_b).to_dict('records')
        else:
            _logger.info(f'[E2E-ABDE] {SCENARIO_B} 생성 시작')
            gen_b     = get_generator(SCENARIO_B)
            records_b = _generate_answers(gen_b, scenario=SCENARIO_B, contexts=contexts)
            gen_b.release()
            pd.DataFrame(records_b).rename(columns={'answer': col_b}).to_csv(
                _mid_b, index=False, encoding='utf-8-sig')
            _logger.info(f'[E2E-ABDE] 중간 저장 완료: {_mid_b}')

    else:
        gen_a     = get_generator(SCENARIO_A)
        gen_b     = get_generator(SCENARIO_B)
        records_a = _generate_answers(gen_a, scenario=SCENARIO_A, contexts=contexts)
        records_b = _generate_answers(gen_b, scenario=SCENARIO_B, contexts=contexts)

    # ── Step 3. 결과 병합 및 저장 ───────────────────────────────────
    df_a = pd.DataFrame(records_a)
    if 'answer' in df_a.columns:
        df_a = df_a.rename(columns={'answer': col_a})

    df_b = pd.DataFrame(records_b)
    if 'answer' in df_b.columns:
        df_b = df_b.rename(columns={'answer': col_b})

    result_df = df_a.merge(df_b[['question', col_b]], on='question', how='left')

    result_df.to_csv(_SAVE_PATH, index=False, encoding='utf-8-sig')
    _logger.info(f'[E2E-ABDE] ✅ 저장 완료: {_SAVE_PATH}')
    _logger.info(f'[E2E-ABDE] 답변 컬럼: {col_a}, {col_b}')

    # ── Step 4. 타입별 요약 출력 ─────────────────────────────────────
    print('\n📊 타입별 생성 완료')
    for t in ['A', 'B', 'D', 'E']:
        n = len(result_df[result_df['type'] == t])
        print(f'  {t}타입: {n}개')


if __name__ == '__main__':
    run_e2e_comparison()
