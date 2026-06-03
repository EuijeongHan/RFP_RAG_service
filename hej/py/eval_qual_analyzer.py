"""
입찰메이트 RAG — 정성 평가 리포트 추출기 (12개 시나리오 버전)
=================================================================
담당 : Generation 파트 (한의정)

[설명]
1. 오류 역추적 (Error Analysis): 12개 시나리오 중 하나라도 지표 ≤ 3점인 행 추출
2. Side-by-Side 블라인드 테스트: 모든 시나리오 답변 랜덤 레이블로 배치
3. C타입 맥락 추적: 대명사 포함 질문 + type=='C' 필터링
4. 시나리오별 요약 통계

[입력]
    e2e_all_comparison.csv         (579개 × 12컬럼 합본)
    quantitative_scores_all.csv   (eval_quant_judge1_all.py 출력)

[컬럼 형식]
    답변 컬럼: ans_kure_gemma, ans_kure_phi, ... (12개)
    점수 컬럼: kure_gemma_avg_faithfulness, ... (12 × 6 = 72개)

[실행]
    python eval_qual_analyzer.py
"""

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
_logger = logging.getLogger(__name__)

# ======================================================================
# 경로 설정
# ======================================================================
_RESULT_DIR  = Path('/mnt/gukrul/dataset/eval_results')
_E2E_PATH    = _RESULT_DIR / 'generation' / 'e2e_all_comparison.csv'
_SCORE_PATH  = _RESULT_DIR / 'generation' / 'quant' / 'BL' / 'quantitative_scores_all.csv'
_QUAL_DIR    = _RESULT_DIR / 'generation' / 'qual'
_QUAL_DIR.mkdir(parents=True, exist_ok=True)

_ERROR_PATH  = _QUAL_DIR / 'qual_error_analysis.csv'
_SBS_PATH    = _QUAL_DIR / 'qual_side_by_side.csv'
_SBS_KEY     = _QUAL_DIR / 'qual_side_by_side_key.csv'
_CTYPE_PATH  = _QUAL_DIR / 'qual_ctype_tracking.csv'
_SUMMARY_PATH= _QUAL_DIR / 'qual_summary.csv'

# ======================================================================
# 12개 시나리오 정의
# ======================================================================
_SCENARIOS = [
    'kure_gemma', 'kure_phi', 'kure_qwen', 'kure_openai',
    'koe5_gemma', 'koe5_phi', 'koe5_qwen', 'koe5_openai',
    'small_gemma', 'small_phi', 'small_qwen', 'small_openai',
]

_METRICS = [
    'faithfulness', 'relevance', 'rejection',
    'correctness', 'context_precision', 'context_recall',
]

# 점수 컬럼명: {scenario}_avg_{metric}
def _score_col(scenario, metric):
    return f'{scenario}_avg_{metric}'

# 답변 컬럼명: ans_{scenario}
def _ans_col(scenario):
    return f'ans_{scenario}'

# 오류 판정 기준
_ERROR_THRESHOLD = 3.0


def generate_qualitative_report():

    # ── 데이터 로드 ────────────────────────────────────────────────
    _logger.info(f'[정성평가] E2E 로드: {_E2E_PATH}')
    e2e_df = pd.read_csv(_E2E_PATH)

    _logger.info(f'[정성평가] 점수 로드: {_SCORE_PATH}')
    score_df = pd.read_csv(_SCORE_PATH)

    # 점수 파일에 question 컬럼이 있으면 merge, 없으면 index 기준 concat
    if 'question' in score_df.columns:
        merge_keys = ['question']
        if 'history' in score_df.columns and 'history' in e2e_df.columns:
            merge_keys.append('history')
        # type/difficulty 등 중복 컬럼은 score_df에서 제거 후 merge
        drop_cols = [c for c in score_df.columns if c in e2e_df.columns and c not in merge_keys]
        merged = pd.merge(e2e_df, score_df.drop(columns=drop_cols), on=merge_keys, how='left')
    else:
        # 행 순서 동일 가정
        merged = pd.concat(
            [e2e_df.reset_index(drop=True), score_df.reset_index(drop=True)],
            axis=1
        )

    _logger.info(f'[정성평가] 병합 완료: {len(merged)}행')

    # 실제 존재하는 시나리오 필터링
    valid_scenarios = [s for s in _SCENARIOS if _ans_col(s) in merged.columns]
    _logger.info(f'[정성평가] 유효 시나리오: {len(valid_scenarios)}개')

    # ──────────────────────────────────────────────────────────────────
    # 1. 오류 역추적 (Error Analysis)
    # ──────────────────────────────────────────────────────────────────
    error_mask = pd.Series(False, index=merged.index)

    # 각 시나리오의 핵심 지표(faithfulness, relevance) 평균이 모두 ≤ threshold인 행
    for scenario in valid_scenarios:
        for metric in ['faithfulness', 'relevance']:
            col = _score_col(scenario, metric)
            if col in merged.columns:
                # NaN은 제외 (채점 안 된 행)
                non_null = merged[col].notna()
                error_mask |= (non_null & (merged[col] <= _ERROR_THRESHOLD))

    # 생성 오류 행도 포함
    for scenario in valid_scenarios:
        ans_col = _ans_col(scenario)
        if ans_col in merged.columns:
            error_mask |= merged[ans_col].str.contains('생성 오류', na=False)

    # 오류 행 추출 컬럼
    base_cols = [c for c in ['question', 'type', 'history', 'retrieved_context']
                 if c in merged.columns]

    ans_cols   = [_ans_col(s) for s in valid_scenarios if _ans_col(s) in merged.columns]
    score_cols = []
    for scenario in valid_scenarios:
        for metric in _METRICS:
            col = _score_col(scenario, metric)
            if col in merged.columns:
                score_cols.append(col)

    error_df = merged[error_mask][base_cols + ans_cols + score_cols].copy()
    error_df.to_csv(_ERROR_PATH, index=False, encoding='utf-8-sig')
    _logger.info(f'[정성평가] 오류 케이스: {len(error_df)}개 → {_ERROR_PATH.name}')

    # ──────────────────────────────────────────────────────────────────
    # 2. Side-by-Side 블라인드 테스트 (12개 시나리오 전부)
    # ──────────────────────────────────────────────────────────────────
    np.random.seed(42)

    sbs_base = merged[base_cols].copy()

    # 시나리오 순서 랜덤 셔플
    shuffled_orders = []
    for _ in range(len(merged)):
        order = valid_scenarios.copy()
        np.random.shuffle(order)
        shuffled_orders.append(order)

    # 공개용: Model_1 ~ Model_N (시나리오명 숨김)
    for i in range(len(valid_scenarios)):
        col_name = f'Model_{chr(65+i)}_Answer'  # Model_A, Model_B, ...
        sbs_base[col_name] = [
            merged.at[idx, _ans_col(shuffled_orders[j][i])]
            if _ans_col(shuffled_orders[j][i]) in merged.columns else ''
            for j, idx in enumerate(merged.index)
        ]

    # 정답 키 (블라인드 평가 완료 후 열람)
    key_df = merged[['question']].copy()
    for i in range(len(valid_scenarios)):
        key_df[f'Secret_Model_{chr(65+i)}'] = [
            shuffled_orders[j][i] for j in range(len(merged))
        ]

    sbs_base.to_csv(_SBS_PATH, index=False, encoding='utf-8-sig')
    key_df.to_csv(_SBS_KEY, index=False, encoding='utf-8-sig')
    _logger.info(f'[정성평가] S-b-S: {len(sbs_base)}개 → {_SBS_PATH.name} / 키: {_SBS_KEY.name}')

    # ──────────────────────────────────────────────────────────────────
    # 3. C타입 맥락 추적
    # ──────────────────────────────────────────────────────────────────
    context_keywords = ['그 ', '저 ', '위에서 ', '앞서 ', '아까 ', '해당 ', '그것 ', '거기']

    ctype_mask = pd.Series(False, index=merged.index)

    # type 컬럼으로 C타입 필터
    if 'type' in merged.columns:
        ctype_mask |= (merged['type'] == 'C')

    # 키워드 기반 추가 필터
    ctype_mask |= merged['question'].str.contains(
        '|'.join(context_keywords), na=False, regex=False
    )

    ctype_df = merged[ctype_mask][base_cols + ans_cols].copy()
    ctype_df.to_csv(_CTYPE_PATH, index=False, encoding='utf-8-sig')
    _logger.info(f'[정성평가] C타입 추적: {len(ctype_df)}개 → {_CTYPE_PATH.name}')

    # ──────────────────────────────────────────────────────────────────
    # 4. 시나리오별 요약 통계
    # ──────────────────────────────────────────────────────────────────
    summary_rows = []
    for scenario in valid_scenarios:
        row = {'scenario': scenario}
        for metric in _METRICS:
            col = _score_col(scenario, metric)
            if col in merged.columns:
                row[metric] = merged[col].dropna().mean()
            else:
                row[metric] = None
        # 생성 오류 건수
        ans_col = _ans_col(scenario)
        if ans_col in merged.columns:
            row['gen_errors'] = merged[ans_col].str.contains('생성 오류', na=False).sum()
        else:
            row['gen_errors'] = None
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(_SUMMARY_PATH, index=False, encoding='utf-8-sig')

    # ── 결과 요약 출력 ────────────────────────────────────────────
    print('\n✅ 정성 평가 리포트 생성 완료')
    print(f'  1. 오류 역추적     : {_ERROR_PATH.name}  ({len(error_df)}건)')
    print(f'  2. 블라인드 S-b-S  : {_SBS_PATH.name}  ({len(sbs_base)}행)')
    print(f'     └ 정답 키        : {_SBS_KEY.name}')
    print(f'  3. C타입 맥락 추적 : {_CTYPE_PATH.name}  ({len(ctype_df)}건)')
    print(f'  4. 시나리오 요약   : {_SUMMARY_PATH.name}')

    print('\n📊 시나리오별 평균 점수')
    print(f'  {"시나리오":<20} {"Faith":>7} {"Relev":>7} {"Rejct":>7} {"CP":>7}')
    print('  ' + '─' * 50)
    for _, r in summary_df.iterrows():
        f  = f'{r["faithfulness"]:.3f}' if r['faithfulness'] is not None else '  N/A'
        rv = f'{r["relevance"]:.3f}'    if r['relevance']    is not None else '  N/A'
        rj = f'{r["rejection"]:.3f}'   if r['rejection']    is not None else '  N/A'
        cp = f'{r["context_precision"]:.3f}' if r['context_precision'] is not None else '  N/A'
        print(f'  {r["scenario"]:<20} {f:>7} {rv:>7} {rj:>7} {cp:>7}')


if __name__ == '__main__':
    generate_qualitative_report()