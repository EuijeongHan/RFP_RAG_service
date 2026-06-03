"""
입찰메이트 RAG — LLM-as-a-Judge 정량 평가 (12개 시나리오 전체)
=================================================================
담당 : Generation 파트 (한의정)

[Judge]
  gpt-5.4-mini (OpenAI) 단독 사용 — 비동기 병렬 호출 + 재시도 로직

[입력]  e2e_all_comparison.csv
[출력]  quantitative_scores_all.csv

[실행]
  export OPENAI_API_KEY=...
  python eval_quant_judge1_all_asynch.py
"""

import os
import re
import asyncio
import logging
import pandas as pd
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
_logger = logging.getLogger(__name__)

# ======================================================================
# Judge 설정
# ======================================================================
_JUDGE_1_MODEL = 'gpt-5.4-mini'
_async_client  = AsyncOpenAI()
_SEMAPHORE     = asyncio.Semaphore(15)
_MAX_RETRIES   = 3

# ======================================================================
# 경로
# ======================================================================
_RESULT_DIR  = Path('/mnt/gukrul/hej/eval_results')
_INPUT_PATH  = Path(os.environ.get('E2E_INPUT', str(_RESULT_DIR / 'generation' / 'e2e_all_comparison.csv')))
_OUTPUT_PATH = Path(os.environ.get('SCORE_OUTPUT', str(_RESULT_DIR / 'generation' / 'quant' / 'BL' / 'quantitative_scores_all.csv')))

# ======================================================================
# 채점 대상 시나리오 (컬럼명)
# ======================================================================
_SCENARIOS = [
    'ans_kure_gemma', 'ans_kure_phi', 'ans_kure_qwen', 'ans_kure_openai',
    'ans_koe5_gemma', 'ans_koe5_phi', 'ans_koe5_qwen', 'ans_koe5_openai',
    'ans_small_gemma','ans_small_phi','ans_small_qwen','ans_small_openai',
]

# ======================================================================
# Judge 프롬프트
# ======================================================================
_JUDGE_PROMPTS = {
    'faithfulness': (
        "당신은 AI 답변의 환각(Hallucination)을 탐지하는 엄격한 평가자입니다.\n"
        "[Context]에 제시된 정보만으로 [Answer]가 작성되었는지 평가하세요.\n\n"
        "채점 기준:\n"
        "  5점: 모든 내용이 Context에 근거함\n"
        "  4점: 대부분 근거 있으나 사소한 추론 포함\n"
        "  3점: Context에 없는 내용이 일부 포함\n"
        "  2점: 상당 부분 Context 외 정보\n"
        "  1점: Context와 무관하게 지어냄\n\n"
        "[Context]:\n{context}\n\n[Answer]:\n{answer}\n\n"
        "반드시 '점수: X' 형식으로만 출력하세요. (X는 1~5 정수)"
    ),
    'relevance': (
        "당신은 AI 답변의 관련성을 평가하는 평가자입니다.\n"
        "[Question]의 의도를 [Answer]가 명확히 해결하고 있는지 평가하세요.\n\n"
        "채점 기준:\n"
        "  5점: 질문 핵심을 정확하고 간결하게 답변\n"
        "  4점: 핵심 답변이 있으나 불필요한 내용 포함\n"
        "  3점: 부분적으로만 답변\n"
        "  2점: 질문과 관련성이 낮음\n"
        "  1점: 동문서답 또는 완전 무관\n\n"
        "[Question]:\n{query}\n\n[Answer]:\n{answer}\n\n"
        "반드시 '점수: X' 형식으로만 출력하세요. (X는 1~5 정수)"
    ),
    'rejection': (
        "당신은 AI 시스템의 거절 적절성을 평가하는 평가자입니다.\n"
        "[Context]에 [Question]에 대한 답이 없을 때,\n"
        "[Answer]가 억지로 답을 만들지 않고 '해당 정보를 찾을 수 없습니다'라고 답했는지 평가하세요.\n\n"
        "채점 기준:\n"
        "  5점: Context에 답이 없으면 정직하게 거절했거나, Context에 답이 있어 정상 답변함\n"
        "  3점: 거절했으나 불완전하거나 모호한 답변\n"
        "  1점: Context에 없는 내용을 억지로 지어내어 답변함\n\n"
        "[Context]:\n{context}\n\n[Question]:\n{query}\n\n[Answer]:\n{answer}\n\n"
        "반드시 '점수: X' 형식으로만 출력하세요. (X는 1~5 정수)"
    ),
    'correctness': (
        "당신은 AI 답변의 팩트 정확도를 평가하는 평가자입니다.\n"
        "[Ground Truth]의 핵심 사실(수치, 날짜, 기관명)과 [Answer]가 일치하는지 평가하세요.\n\n"
        "채점 기준:\n"
        "  5점: 핵심 사실과 수치가 완벽히 일치\n"
        "  4점: 대부분 일치하나 사소한 차이\n"
        "  3점: 핵심 사실 일부 누락 또는 부정확\n"
        "  2점: 주요 오류 다수\n"
        "  1점: 핵심 사실이 틀림\n\n"
        "[Ground Truth]:\n{ground_truth}\n\n[Answer]:\n{answer}\n\n"
        "반드시 '점수: X' 형식으로만 출력하세요. (X는 1~5 정수)"
    ),
    'context_precision': (
        "당신은 검색 시스템의 정밀도를 평가하는 평가자입니다.\n"
        "[Context]에 포함된 각 문서 조각이 [Question]을 답변하는 데 실제로 필요한 정보를 담고 있는지 평가하세요.\n\n"
        "채점 기준:\n"
        "  5점: 검색된 모든 내용이 질문과 직접 관련 있음\n"
        "  4점: 대부분 관련 있으나 일부 불필요한 내용 포함\n"
        "  3점: 절반 정도만 질문과 관련 있음\n"
        "  2점: 관련 없는 내용이 대부분\n"
        "  1점: 질문과 전혀 무관한 내용만 검색됨\n\n"
        "[Question]:\n{query}\n\n[Context]:\n{context}\n\n"
        "반드시 '점수: X' 형식으로만 출력하세요. (X는 1~5 정수)"
    ),
    'context_recall': (
        "당신은 검색 시스템의 재현율을 평가하는 평가자입니다.\n"
        "[Ground Truth]에 담긴 핵심 정보가 [Context]에 충분히 포함되어 있는지 평가하세요.\n\n"
        "채점 기준:\n"
        "  5점: Ground Truth의 모든 핵심 정보가 Context에 존재함\n"
        "  4점: 대부분 존재하나 사소한 정보 일부 누락\n"
        "  3점: 핵심 정보의 절반 정도만 Context에 있음\n"
        "  2점: 핵심 정보 대부분 누락\n"
        "  1점: Ground Truth 정보가 Context에 전혀 없음\n\n"
        "[Ground Truth]:\n{ground_truth}\n\n[Context]:\n{context}\n\n"
        "반드시 '점수: X' 형식으로만 출력하세요. (X는 1~5 정수)"
    ),
}


def _parse_score(raw: str) -> Optional[int]:
    if not raw:
        return None
    match = re.search(r'점수\s*:\s*(\d)', raw)
    if match:
        return int(match.group(1))
    raw_strip = raw.strip()
    if raw_strip.isdigit() and 1 <= int(raw_strip) <= 5:
        return int(raw_strip)
    digits = re.findall(r'\b[1-5]\b', raw)
    return int(digits[0]) if digits else None


async def _call_judge_async(prompt: str) -> Optional[int]:
    """지수 백오프 재시도 포함 API 호출"""
    for attempt in range(_MAX_RETRIES):
        async with _SEMAPHORE:
            try:
                resp = await _async_client.chat.completions.create(
                    model=_JUDGE_1_MODEL,
                    messages=[{'role': 'user', 'content': prompt}],
                    max_completion_tokens=20,
                    timeout=15,
                )
                return _parse_score(resp.choices[0].message.content)
            except Exception as e:
                if attempt == _MAX_RETRIES - 1:
                    _logger.error(f'[Judge] 최종 실패: {e}')
                    return None
                wait = 2 ** attempt
                _logger.warning(f'[Judge] 재시도 {attempt + 1}/{_MAX_RETRIES} ({wait}초 대기): {e}')
                await asyncio.sleep(wait)


async def score_one_async(query, context, answer, ground_truth=None):
    async_tasks = {}
    none_keys   = []

    for metric in ('faithfulness', 'relevance', 'rejection'):
        prompt = _JUDGE_PROMPTS[metric].format(context=context, query=query, answer=answer)
        async_tasks[f'avg_{metric}'] = _call_judge_async(prompt)

    for metric in ('correctness', 'context_recall'):
        if ground_truth and pd.notna(ground_truth) and str(ground_truth).strip():
            prompt = _JUDGE_PROMPTS[metric].format(
                ground_truth=ground_truth, answer=answer, context=context,
            )
            async_tasks[f'avg_{metric}'] = _call_judge_async(prompt)
        else:
            none_keys.append(f'avg_{metric}')

    prompt = _JUDGE_PROMPTS['context_precision'].format(query=query, context=context)
    async_tasks['avg_context_precision'] = _call_judge_async(prompt)

    results_list = await asyncio.gather(*async_tasks.values())
    result = dict(zip(async_tasks.keys(), results_list))
    for k in none_keys:
        result[k] = None

    return result


def _is_row_complete(row: pd.Series) -> bool:
    """relevance 컬럼 기준으로 행이 정상 채점됐는지 확인"""
    for col in _SCENARIOS:
        prefix = col.replace('ans_', '')
        if pd.isna(row.get(f'{prefix}_avg_relevance')):
            return False
    return True


async def run_all_async():
    _logger.info(f'[Judge-All] 시작 | Judge={_JUDGE_1_MODEL} | 비동기 병렬 + 재시도')
    df = pd.read_csv(_INPUT_PATH)
    _logger.info(f'[Judge-All] 입력: {len(df)}행 × {len(_SCENARIOS)}개 시나리오')

    # 체크포인트: 행 단위 NaN 검증으로 정상 채점된 행만 재사용
    completed_questions = set()
    checkpoint_rows     = []

    if _OUTPUT_PATH.exists():
        try:
            _ckpt = pd.read_csv(_OUTPUT_PATH)
            for _, row in _ckpt.iterrows():
                if _is_row_complete(row):
                    completed_questions.add(row['question'])
                    checkpoint_rows.append(row.to_dict())
            n_invalid = len(_ckpt) - len(checkpoint_rows)
            _logger.info(
                f'[Judge-All] 체크포인트: 정상 {len(checkpoint_rows)}행 재사용'
                + (f' | NaN {n_invalid}행 재채점' if n_invalid > 0 else '')
            )
        except Exception as e:
            _logger.warning(f'[Judge-All] 체크포인트 파싱 실패 → 전체 재실행: {e}')

    # 아직 채점 안 된 행만 필터
    pending = [(idx, row) for idx, row in df.iterrows()
               if row['question'] not in completed_questions]
    _logger.info(f'[Judge-All] 신규 채점 대상: {len(pending)}행')

    if not pending:
        _logger.info('[Judge-All] 모든 행 채점 완료 상태.')
        return

    results = checkpoint_rows.copy()

    for idx, row in tqdm(pending, total=len(pending), desc='[Judge] 채점 중'):
        query   = row['question']
        context = row['retrieved_context']
        gt      = row.get('ground_truth_answer', None)

        row_result = {
            'question': query,
            'type'    : row['type'],
            'history' : row.get('history', ''),
        }

        scenario_tasks = {}
        for col in _SCENARIOS:
            answer = row.get(col, '')
            prefix = col.replace('ans_', '')
            if not isinstance(answer, str) or '생성 오류' in answer:
                for metric in ('faithfulness', 'relevance', 'rejection',
                               'correctness', 'context_precision', 'context_recall'):
                    row_result[f'{prefix}_avg_{metric}'] = None
            else:
                scenario_tasks[prefix] = score_one_async(query, context, answer, gt)

        if scenario_tasks:
            scores_list = await asyncio.gather(*scenario_tasks.values())
            for prefix, scores in zip(scenario_tasks.keys(), scores_list):
                for k, v in scores.items():
                    row_result[f'{prefix}_{k}'] = v

        results.append(row_result)

        if len(results) % 50 == 0:
            pd.DataFrame(results).to_csv(_OUTPUT_PATH, index=False, encoding='utf-8-sig')
            _logger.info(f'[Judge-All] 체크포인트 저장: {len(results)}행')

    # 최종 저장
    result_df = pd.DataFrame(results)
    result_df.to_csv(_OUTPUT_PATH, index=False, encoding='utf-8-sig')
    _logger.info(f'[Judge-All] ✅ 저장 완료: {_OUTPUT_PATH}')

    # 요약 출력
    print('\n📊 정량 평가 요약 (avg 기준)')
    metrics = ['faithfulness', 'relevance', 'rejection', 'context_precision']
    header = f'{"시나리오":<20}' + ''.join(f'{m[:8]:>10}' for m in metrics)
    print(header)
    print('-' * (20 + 10 * len(metrics)))
    for col in _SCENARIOS:
        prefix = col.replace('ans_', '')
        row_str = f'{prefix:<20}'
        for metric in metrics:
            val = result_df[f'{prefix}_avg_{metric}'].dropna().mean()
            row_str += f'{val:>10.3f}' if not pd.isna(val) else f'{"N/A":>10}'
        print(row_str)


def run_all():
    asyncio.run(run_all_async())


if __name__ == '__main__':
    run_all()