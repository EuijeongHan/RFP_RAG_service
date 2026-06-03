"""
입찰메이트 RAG — Release Gate 통과 여부 체크
==============================================
담당 : Retrieval + Generation 파트 (한의정)

[설명]
- Retrieval / Generation 두 Gate 모두 통과해야 배포 가능.
- 각 지표는 PASS / GOOD / FAIL 3단계로 판정.
- Generation은 12개 시나리오 각각 판정 후 종합.

[판정 기준]

  Retrieval Gate (전체):
    PASS: Hit@5 ≥ 0.90, MRR ≥ 0.82, nDCG ≥ 0.78
    GOOD: Hit@5 ≥ 0.95, MRR ≥ 0.87, nDCG ≥ 0.83

  Retrieval Gate (타입별 MRR):
    A타입 (추출 정밀도) : PASS ≥ 0.92 / GOOD ≥ 0.95
    B타입 (종합 능력)   : PASS ≥ 0.77 / GOOD ≥ 0.82
    C타입 (맥락 유지)   : PASS ≥ 0.88 / GOOD ≥ 0.93
    D타입 (환각 방지)   : PASS ≥ 0.81 / GOOD ≥ 0.86
    E타입 (오타 질문)   : PASS ≥ 0.82 / GOOD ≥ 0.87

  Generation Gate (12개 시나리오 × 6지표):
    PASS: Faithfulness / Relevance / Rejection / Context Precision ≥ 3.5
    GOOD: 위 전부 ≥ 4.0

[입력 파일]
  Retrieval : eval_results/eval_results_final_A-1_579_v1.csv
  Generation: eval_results/generation/quantitative_scores_all.csv

[실행]
  python check_release_gate.py
  python check_release_gate.py \
    --retrieval_csv eval_results/eval_results_chunks_all_KURE_579_v1.csv \
    --gen_csv eval_results/generation/quantitative_scores_all.csv
"""

import os
import glob
import argparse
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
_logger = logging.getLogger(__name__)

# ======================================================================
# 경로
# ======================================================================
_RESULT_DIR = Path('/mnt/gukrul/hej/eval_results')
_GEN_CSV    = _RESULT_DIR / 'generation' / 'quantitative_scores_all.csv'

# ======================================================================
# Gate 기준값
# ======================================================================
_RET_PASS = {'hit': 0.90, 'mrr': 0.82, 'ndcg': 0.78}
_RET_GOOD = {'hit': 0.95, 'mrr': 0.87, 'ndcg': 0.83}

_TYPE_MRR_PASS = {'A': 0.92, 'B': 0.77, 'C': 0.88, 'D': 0.81, 'E': 0.82}
_TYPE_MRR_GOOD = {'A': 0.95, 'B': 0.82, 'C': 0.93, 'D': 0.86, 'E': 0.87}
_TYPE_LABEL    = {
    'A': '추출 정밀도 (Extraction Accuracy)',
    'B': '종합 능력   (Synthesis)',
    'C': '맥락 유지   (Contextual Awareness)',
    'D': '환각 방지   (Grounding & Refusal)',
    'E': '오타 질문   (Noise Robustness)',
}

_GEN_METRICS  = ['faithfulness', 'relevance', 'rejection', 'context_precision']
_GEN_PASS     = 3.5
_GEN_GOOD     = 4.0

_SCENARIOS = [
    'kure_gemma', 'kure_phi', 'kure_qwen', 'kure_openai',
    'koe5_gemma', 'koe5_phi', 'koe5_qwen', 'koe5_openai',
    'small_gemma', 'small_phi', 'small_qwen', 'small_openai',
]

# ======================================================================
# 유틸
# ======================================================================
def _grade(val, pass_th, good_th):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 'N/A', '⬜'
    if val >= good_th:
        return 'GOOD', '🟢'
    if val >= pass_th:
        return 'PASS', '🟡'
    return 'FAIL', '🔴'

def _sep(width=64):
    print('─' * width)


# ======================================================================
# Retrieval Gate
# ======================================================================
def check_retrieval_gate(csv_path: Path) -> bool:
    _logger.info(f'[Retrieval Gate] {csv_path.name}')
    df = pd.read_csv(csv_path)

    for col in ('hit', 'mrr', 'ndcg', 'type'):
        if col not in df.columns:
            _logger.error(f'컬럼 없음: {col}')
            return False

    subset = df[df['hit'].notna()]
    hit_mean  = subset['hit'].mean()
    mrr_mean  = subset['mrr'].mean()
    ndcg_mean = subset['ndcg'].mean()

    print('\n' + '═' * 64)
    print('  📋 Retrieval Gate')
    print('═' * 64)
    print(f'\n  {"지표":<8}  {"현재값":>8}  {"PASS기준":>9}  {"GOOD기준":>9}  {"판정":>6}')
    _sep()

    all_pass = True
    for label, val, pk, gk in [
        ('Hit@5', hit_mean,  _RET_PASS['hit'],  _RET_GOOD['hit']),
        ('MRR',   mrr_mean,  _RET_PASS['mrr'],  _RET_GOOD['mrr']),
        ('nDCG',  ndcg_mean, _RET_PASS['ndcg'], _RET_GOOD['ndcg']),
    ]:
        grade, icon = _grade(val, pk, gk)
        if grade == 'FAIL':
            all_pass = False
        print(f'  {label:<8}  {val:>8.4f}  {pk:>9.2f}  {gk:>9.2f}  {icon} {grade}')

    print(f'\n  [타입별 MRR]')
    _sep()
    for t in ['A', 'B', 'C', 'D', 'E']:
        tdf = subset[subset['type'] == t]
        if tdf.empty:
            print(f'  {t}타입  데이터 없음')
            continue
        mrr_t = tdf['mrr'].mean()
        grade, icon = _grade(mrr_t, _TYPE_MRR_PASS[t], _TYPE_MRR_GOOD[t])
        if grade == 'FAIL':
            all_pass = False
        label = _TYPE_LABEL.get(t, t)
        print(f'  {t}타입  {label}')
        print(f'         현재={mrr_t:.4f}  PASS≥{_TYPE_MRR_PASS[t]}  GOOD≥{_TYPE_MRR_GOOD[t]}  {icon} {grade}')

    _sep()
    icon = '✅' if all_pass else '❌'
    text = '통과' if all_pass else '미달 — 추가 실험 필요'
    print(f'\n  Retrieval Gate 최종: {icon} {text}\n')
    return all_pass


# ======================================================================
# Generation Gate
# ======================================================================
def check_generation_gate(csv_path: Path) -> bool:
    _logger.info(f'[Generation Gate] {csv_path.name}')
    df = pd.read_csv(csv_path)

    print('\n' + '═' * 64)
    print('  📋 Generation Gate (12개 시나리오)')
    print('═' * 64)

    # 헤더
    print(f'\n  {"시나리오":<20}', end='')
    for m in _GEN_METRICS:
        print(f'  {m[:7]:>7}', end='')
    print(f'  {"종합":>6}')
    _sep()

    all_pass    = True
    scenario_results = {}

    for scenario in _SCENARIOS:
        vals = {}
        for metric in _GEN_METRICS:
            col = f'{scenario}_avg_{metric}'
            if col in df.columns:
                vals[metric] = df[col].dropna().mean()
            else:
                vals[metric] = None

        # 시나리오 종합 판정 (존재하는 지표만)
        existing = [v for v in vals.values() if v is not None]
        if not existing:
            scenario_results[scenario] = 'N/A'
            continue

        sc_pass = all(v >= _GEN_PASS for v in existing)
        sc_good = all(v >= _GEN_GOOD for v in existing)
        sc_grade = 'GOOD' if sc_good else ('PASS' if sc_pass else 'FAIL')
        sc_icon  = '🟢' if sc_good else ('🟡' if sc_pass else '🔴')
        scenario_results[scenario] = sc_grade

        if sc_grade == 'FAIL':
            all_pass = False

        print(f'  {scenario:<20}', end='')
        for metric in _GEN_METRICS:
            v = vals.get(metric)
            if v is None:
                print(f'  {"N/A":>7}', end='')
            else:
                _, icon = _grade(v, _GEN_PASS, _GEN_GOOD)
                print(f'  {v:>6.3f}{icon[0] if icon != "⬜" else " "}', end='')
        print(f'  {sc_icon} {sc_grade}')

    _sep()

    # PASS 기준 기준 충족 시나리오 수
    pass_count = sum(1 for g in scenario_results.values() if g in ('PASS', 'GOOD'))
    good_count = sum(1 for g in scenario_results.values() if g == 'GOOD')
    total      = len([g for g in scenario_results.values() if g != 'N/A'])

    print(f'\n  통과 시나리오: {pass_count}/{total}개  (GOOD: {good_count}개)')

    icon = '✅' if all_pass else '❌'
    text = '전체 통과' if all_pass else f'일부 미달 — 파인튜닝 후 재평가 권장'
    print(f'  Generation Gate 최종: {icon} {text}\n')
    return all_pass


# ======================================================================
# 최종 판정
# ======================================================================
def final_verdict(ret_pass: bool, gen_pass: bool):
    print('═' * 64)
    print('  🏁 Release Gate 최종 판정')
    print('═' * 64)
    print(f'  Retrieval Gate  : {"✅ 통과" if ret_pass else "❌ 미달"}')
    print(f'  Generation Gate : {"✅ 통과" if gen_pass else "❌ 미달 / ⬜ 미실행"}')
    print()
    if ret_pass and gen_pass:
        print('  🚀 배포 가능 (PASS)')
    elif ret_pass and not gen_pass:
        print('  ⚠️  Retrieval 통과. Generation 파인튜닝 후 재평가 필요.')
    elif not ret_pass and gen_pass:
        print('  ⚠️  Generation 통과. Retrieval 추가 실험 필요.')
    else:
        print('  🔴 전 항목 미달. 추가 실험 필요.')
    print('═' * 64)


# ======================================================================
# 메인
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description='입찰메이트 Release Gate 체크')
    parser.add_argument('--retrieval_csv', default=None)
    parser.add_argument('--gen_csv', default=str(_GEN_CSV))
    args = parser.parse_args()

    # Retrieval CSV 자동 선택
    if args.retrieval_csv:
        ret_csv = Path(args.retrieval_csv)
    else:
        ret_csv = _RESULT_DIR / 'eval_results_final_A-1_579_v1.csv'
        _logger.info(f'Retrieval CSV 기본값: {ret_csv.name}')

    gen_csv = Path(args.gen_csv)

    ret_pass = False
    gen_pass = False

    if ret_csv and ret_csv.exists():
        ret_pass = check_retrieval_gate(ret_csv)
    else:
        print('\n⬜ Retrieval Gate: CSV 없음 → 건너뜀\n')

    if gen_csv.exists():
        gen_pass = check_generation_gate(gen_csv)
    else:
        print(f'\n⬜ Generation Gate: {gen_csv} 없음 → 건너뜀\n')

    final_verdict(ret_pass, gen_pass)


if __name__ == '__main__':
    import glob
    main()