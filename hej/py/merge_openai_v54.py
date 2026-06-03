"""
입찰메이트 RAG — OPENAI 재채점 결과 머지 + 발표용 숫자 재계산
=================================================================
[입력]
  quantitative_scores_all.csv          (기존 12개 시나리오 점수, OPENAI는 옛 judge)
  quantitative_scores_openai_v54.csv   (gpt-5.4-mini 재채점, OPENAI 3개만)
[출력]
  quantitative_scores_all_v54.csv      (OPENAI 3컬럼만 교체, 9개 보존)
  + 콘솔에 슬라이드 12 표 / 슬라이드 10·12 Faithfulness 평균 출력
"""

import pandas as pd
from pathlib import Path

_DIR = Path('/mnt/gukrul/dataset/eval_results/generation/quant')
_OLD = _DIR / 'quantitative_scores_all.csv'
_NEW = _DIR / 'quantitative_scores_openai_v54.csv'
_OUT = _DIR / 'quantitative_scores_all_v54.csv'

_METRICS = ['faithfulness', 'relevance', 'rejection',
            'context_precision', 'correctness', 'context_recall']
_OPENAI_SCEN = ['kure_openai', 'koe5_openai', 'small_openai']

old = pd.read_csv(_OLD)
new = pd.read_csv(_NEW)

# question 기준 정렬 정합
old = old.set_index('question')
new = new.set_index('question')

# OPENAI 컬럼만 교체
replaced = []
for scen in _OPENAI_SCEN:
    for m in _METRICS:
        col = f'{scen}_avg_{m}'
        if col in old.columns and col in new.columns:
            old[col] = new[col]
            replaced.append(col)
print(f'교체된 컬럼 {len(replaced)}개: {replaced}')

old = old.reset_index()
old.to_csv(_OUT, index=False, encoding='utf-8-sig')
print(f'저장: {_OUT}')

# ---- 슬라이드 12 표 (4지표) ----
df = old.set_index('question')
all_scen = ['kure_gemma','kure_phi','kure_qwen','kure_openai',
            'koe5_gemma','koe5_phi','koe5_qwen','koe5_openai',
            'small_gemma','small_phi','small_qwen','small_openai']
slide12_metrics = ['faithfulness','relevance','rejection','context_precision']

print('\n=== 슬라이드 12 표 (gpt-5.4-mini judge = OPENAI 3개만) ===')
hdr = f'{"시나리오":<16}' + ''.join(f'{m[:7]:>9}' for m in slide12_metrics)
print(hdr)
for s in all_scen:
    line = f'{s:<16}'
    for m in slide12_metrics:
        v = df[f'{s}_avg_{m}'].dropna().mean()
        line += f'{v:>9.2f}' if pd.notna(v) else f'{"N/A":>9}'
    print(line)

# ---- 슬라이드 10·12 Faithfulness 임베딩별 평균 (OPENAI 포함 4개) ----
print('\n=== 임베딩별 Faithfulness 평균 (LLM 4종 평균, OPENAI 포함) ===')
for emb in ['kure', 'koe5', 'small']:
    cols = [f'{emb}_{llm}_avg_faithfulness' for llm in ['gemma','phi','qwen','openai']]
    # 시나리오 평균을 먼저 낸 뒤 4개 평균 (= 슬라이드 산식)
    scen_means = [df[c].dropna().mean() for c in cols]
    print(f'{emb.upper():<6} 4개 평균 = {sum(scen_means)/len(scen_means):.3f}'
          f'  (gemma {scen_means[0]:.2f}, phi {scen_means[1]:.2f}, '
          f'qwen {scen_means[2]:.2f}, openai {scen_means[3]:.2f})')
