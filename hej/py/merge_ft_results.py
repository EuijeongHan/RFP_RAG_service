"""
입찰메이트 RAG — FT E2E 결과 합본
====================================
중간파일들을 읽어 e2e_all_comparison_ft.csv 생성
"""

import pandas as pd
from pathlib import Path

_GEN_DIR = Path('/mnt/gukrul/dataset/eval_results/generation')

# 중간파일 → 컬럼명 매핑
_ABDE_FILES = {
    'e2e_abde_kure_mid_KURE_GEMMA.csv'   : 'ans_kure_gemma',
    'e2e_abde_kure_mid_KURE_PHI_FT.csv'  : 'ans_kure_phi',
    'e2e_abde_kure_mid_SMALL_OPENAI.csv' : 'ans_small_openai',
    'e2e_abde_small_mid_SMALL_GEMMA.csv' : 'ans_small_gemma',
    'e2e_abde_small_mid_SMALL_PHI_FT.csv': 'ans_small_phi',
}

_CTYPE_FILES = {
    'e2e_ctype_kure_mid_SMALL_OPENAI.csv' : 'ans_small_openai',
    'e2e_ctype_kure_mid_KURE_GEMMA.csv'   : 'ans_kure_gemma',
    'e2e_ctype_kure_mid_KURE_PHI_FT.csv'  : 'ans_kure_phi',
    'e2e_ctype_small_mid_SMALL_OPENAI.csv': 'ans_small_openai',
    'e2e_ctype_small_mid_SMALL_GEMMA.csv' : 'ans_small_gemma',
    'e2e_ctype_small_mid_SMALL_PHI_FT.csv': 'ans_small_phi',
}

_KEY_COLS = ['question', 'type', 'difficulty', 'history', 'retrieved_context']

def merge_files(file_map: dict) -> pd.DataFrame:
    base_df = None
    for fname, col in file_map.items():
        fpath = _GEN_DIR / fname
        if not fpath.exists():
            print(f'  ⚠️  없음: {fname}')
            continue
        df = pd.read_csv(fpath)
        ans_col = [c for c in df.columns if c.startswith('ans_')]
        if not ans_col:
            print(f'  ⚠️  ans_ 컬럼 없음: {fname}')
            continue
        df = df.rename(columns={ans_col[0]: col})
        if base_df is None:
            base_df = df[_KEY_COLS + [col]]
        else:
            base_df = base_df.merge(df[['question', col]], on='question', how='left')
        print(f'  ✅ {fname} → {col} ({len(df)}개)')
    return base_df

print('=== ABDE 합본 ===')
abde_df = merge_files(_ABDE_FILES)

print('\n=== CTYPE 합본 ===')
ctype_df = merge_files(_CTYPE_FILES)

# ABDE + CTYPE 합치기
if abde_df is not None and ctype_df is not None:
    # 공통 컬럼 맞추기
    for col in abde_df.columns:
        if col not in ctype_df.columns:
            ctype_df[col] = None
    for col in ctype_df.columns:
        if col not in abde_df.columns:
            abde_df[col] = None
    merged = pd.concat([abde_df, ctype_df], ignore_index=True)
elif abde_df is not None:
    merged = abde_df
else:
    merged = ctype_df

out = _GEN_DIR / 'e2e_all_comparison_ft.csv'
merged.to_csv(out, index=False, encoding='utf-8-sig')
print(f'\n✅ 저장: {out} ({len(merged)}개)')
print(f'   컬럼: {list(merged.columns)}')
