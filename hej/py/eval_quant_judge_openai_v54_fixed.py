"""
입찰메이트 RAG — OPENAI 재채점 v2 (mid 파일 직접 입력)
=================================================================
[배경]
  e2e_all_comparison.csv / v12 의 retrieved_context 컬럼이 460/579행
  비어 있었다(다운로드 사본에서 긴 context 텍스트 손실 추정).
  → 단일 context 컬럼으로 3개 임베딩 답변을 채점하던 구조 자체도
     임베딩-context 불일치 문제가 있었음.

[해결]
  임베딩별 mid 파일(context 100% 정상, 빈행 0)을 직접 입력으로 사용.
  각 임베딩의 retrieved_context로 그 임베딩의 OPENAI 답변만 채점 →
  context-답변 정합 완전 보장.

[입력] (임베딩별 abde + ctype 합쳐 579행)
  e2e_abde_{kure,koe5,small}_mid_{KURE,KOE5,SMALL}_OPENAI.csv  (516)
  e2e_ctype_{kure,koe5,small}_mid_{KURE,KOE5,SMALL}_OPENAI.csv (63)

[출력] quantitative_scores_openai_v54_fixed.csv
       (kure_openai / koe5_openai / small_openai × 6지표)

[Judge] gpt-5.4-mini — 기존 프롬프트/파싱/재시도 동일

[실행 - Colab]
  import importlib.util, sys
  spec = importlib.util.spec_from_file_location('J2',
      '/content/drive/MyDrive/data/bidmate/code/eval_quant_judge_openai_v54_fixed.py')
  J2 = importlib.util.module_from_spec(spec); sys.modules['J2']=J2
  spec.loader.exec_module(J2)
  import os; from openai import AsyncOpenAI
  J2._async_client = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])
  J2._BASE = Path('/content/drive/MyDrive/data/bidmate/eval_results/generation/')
  J2._OUTPUT_PATH = J2._BASE / 'quant' / 'quantitative_scores_openai_v54_fixed.csv'
  import nest_asyncio; nest_asyncio.apply()
  await J2.run_all_async()
"""

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

_JUDGE_1_MODEL = 'gpt-5.4-mini'
_async_client  = AsyncOpenAI()
_SEMAPHORE     = asyncio.Semaphore(15)
_MAX_RETRIES   = 3

# ======================================================================
# 경로 — Colab에서 _BASE / _OUTPUT_PATH 덮어쓰기
# ======================================================================
_BASE = Path('/mnt/gukrul/dataset/eval_results/generation')
_OUTPUT_PATH = _BASE / 'quant' / 'quantitative_scores_openai_v54_fixed.csv'

# 임베딩별 (abde mid, ctype mid, 답변컬럼, 출력 prefix)
_EMB_FILES = {
    'kure_openai': (
        'e2e_abde_kure_mid_KURE_OPENAI.csv',
        'e2e_ctype_kure_mid_KURE_OPENAI.csv',
        'ans_kure_openai',
    ),
    'koe5_openai': (
        'e2e_abde_koe5_mid_KOE5_OPENAI.csv',
        'e2e_ctype_koe5_mid_KOE5_OPENAI.csv',
        'ans_koe5_openai',
    ),
    'small_openai': (
        'e2e_abde_small_mid_SMALL_OPENAI.csv',
        'e2e_ctype_small_mid_SMALL_OPENAI.csv',
        'ans_small_openai',
    ),
}

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
    m = re.search(r'점수\s*:\s*(\d)', raw)
    if m:
        return int(m.group(1))
    s = raw.strip()
    if s.isdigit() and 1 <= int(s) <= 5:
        return int(s)
    d = re.findall(r'\b[1-5]\b', raw)
    return int(d[0]) if d else None


async def _call_judge_async(prompt: str) -> Optional[int]:
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
                _logger.warning(f'[Judge] 재시도 {attempt+1}/{_MAX_RETRIES} ({wait}s): {e}')
                await asyncio.sleep(wait)


async def score_one_async(query, context, answer, ground_truth=None):
    tasks = {}
    none_keys = []
    for metric in ('faithfulness', 'relevance', 'rejection'):
        p = _JUDGE_PROMPTS[metric].format(context=context, query=query, answer=answer)
        tasks[f'avg_{metric}'] = _call_judge_async(p)
    for metric in ('correctness', 'context_recall'):
        if ground_truth and pd.notna(ground_truth) and str(ground_truth).strip():
            p = _JUDGE_PROMPTS[metric].format(ground_truth=ground_truth, answer=answer, context=context)
            tasks[f'avg_{metric}'] = _call_judge_async(p)
        else:
            none_keys.append(f'avg_{metric}')
    p = _JUDGE_PROMPTS['context_precision'].format(query=query, context=context)
    tasks['avg_context_precision'] = _call_judge_async(p)

    vals = await asyncio.gather(*tasks.values())
    res = dict(zip(tasks.keys(), vals))
    for k in none_keys:
        res[k] = None
    return res


def _load_emb(abde_name, ctype_name):
    """abde(516) + ctype(63) → 579행. context/답변 정합 보장."""
    a = pd.read_csv(_BASE / abde_name)
    c = pd.read_csv(_BASE / ctype_name)
    df = pd.concat([a, c], ignore_index=True)
    return df


async def run_all_async():
    _logger.info(f'[Judge-FIX] 시작 | Judge={_JUDGE_1_MODEL} | mid 직접 입력 (context 정합)')
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 임베딩별 df 로드 후 question 기준으로 병합 (한 행에 3 임베딩 묶기)
    # 단 context는 임베딩마다 다르므로 채점은 각자 자기 context로 한다.
    emb_dfs = {}
    for prefix, (abde, ctype, anscol) in _EMB_FILES.items():
        d = _load_emb(abde, ctype)
        e = d['retrieved_context'].isna() | (d['retrieved_context'].astype(str).str.strip() == '')
        _logger.info(f'[Judge-FIX] {prefix}: {len(d)}행, 빈context={e.sum()}')
        emb_dfs[prefix] = d

    # 기준 질문 목록 (kure 기준 579)
    base_df = emb_dfs['kure_openai'][['question', 'type', 'history']].copy()

    # 체크포인트
    completed = set()
    rows = []
    if _OUTPUT_PATH.exists():
        try:
            ck = pd.read_csv(_OUTPUT_PATH)
            for _, r in ck.iterrows():
                # 3 임베딩 모두 relevance 있으면 완료로 간주
                ok = all(pd.notna(r.get(f'{p}_avg_relevance')) for p in _EMB_FILES)
                if ok:
                    completed.add(r['question'])
                    rows.append(r.to_dict())
            _logger.info(f'[Judge-FIX] 체크포인트 재사용 {len(rows)}행')
        except Exception as e:
            _logger.warning(f'[Judge-FIX] 체크포인트 파싱 실패: {e}')

    pending_idx = [i for i in range(len(base_df))
                   if base_df.iloc[i]['question'] not in completed]
    _logger.info(f'[Judge-FIX] 신규 채점: {len(pending_idx)}행')
    if not pending_idx:
        _logger.info('완료 상태.')
        return

    for i in tqdm(pending_idx, desc='[Judge-FIX] 채점'):
        q = base_df.iloc[i]['question']
        row_result = {
            'question': q,
            'type': base_df.iloc[i]['type'],
            'history': base_df.iloc[i].get('history', ''),
        }
        scen_tasks = {}
        for prefix, (_, _, anscol) in _EMB_FILES.items():
            d = emb_dfs[prefix]
            # 같은 위치(i) — abde/ctype 순서가 임베딩 간 동일하므로 행 정렬 일치
            answer  = d.iloc[i][anscol]
            context = d.iloc[i]['retrieved_context']
            if not isinstance(answer, str) or '생성 오류' in answer:
                for m in ('faithfulness','relevance','rejection',
                          'correctness','context_precision','context_recall'):
                    row_result[f'{prefix}_avg_{m}'] = None
            else:
                scen_tasks[prefix] = score_one_async(q, context, answer, None)

        if scen_tasks:
            res_list = await asyncio.gather(*scen_tasks.values())
            for prefix, scores in zip(scen_tasks.keys(), res_list):
                for k, v in scores.items():
                    row_result[f'{prefix}_{k}'] = v

        rows.append(row_result)
        if len(rows) % 50 == 0:
            pd.DataFrame(rows).to_csv(_OUTPUT_PATH, index=False, encoding='utf-8-sig')
            _logger.info(f'[Judge-FIX] 체크포인트 저장 {len(rows)}행')

    out = pd.DataFrame(rows)
    out.to_csv(_OUTPUT_PATH, index=False, encoding='utf-8-sig')
    _logger.info(f'[Judge-FIX] ✅ 저장: {_OUTPUT_PATH}')

    print('\n📊 OPENAI 재채점 (mid 직접, context 정합) — gpt-5.4-mini')
    metrics = ['faithfulness', 'relevance', 'rejection', 'context_precision']
    print(f'{"시나리오":<16}' + ''.join(f'{m[:8]:>10}' for m in metrics))
    print('-'*(16+10*len(metrics)))
    for prefix in _EMB_FILES:
        line = f'{prefix:<16}'
        for m in metrics:
            v = out[f'{prefix}_avg_{m}'].dropna().mean()
            line += f'{v:>10.3f}' if pd.notna(v) else f'{"N/A":>10}'
        print(line)


def run_all():
    asyncio.run(run_all_async())


if __name__ == '__main__':
    run_all()
