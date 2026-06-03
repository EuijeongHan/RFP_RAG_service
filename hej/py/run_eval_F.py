"""
입찰메이트 RAG — 통합 배치 평가 스크립트
==========================================
담당 : Retrieval 파트 (한의정)

기존 배치 파일 통합 (16개 → 1개):
    start_batch_(1).py
    start_batch_kh_fixed_v1.py
    start_batch_kh_v3.py
    start_batch_kh_v3_hybrid_ck.py
    chunks_all_batch_(1).py
    chunks_all_batch_hybrid_ck.py
    c_type_batch_(1).py
    c_type_batch_kh_fixed_v1.py
    c_type_batch_kh_v3.py
    c_type_batch_kh_v3_hybrid_ck.py
    c_type_batch_chunks_all.py
    c_type_batch_chunks_all_hybrid_ck.py

[사용법]
    # 전체 eval (579개)
    python run_eval.py --chunks kh_fixed_v2          # 기본값
    python run_eval.py --chunks kh_fixed_v1
    python run_eval.py --chunks kh_v3
    python run_eval.py --chunks kh_v3    --hybrid
    python run_eval.py --chunks chunks_all
    python run_eval.py --chunks chunks_all --hybrid

    # C타입 히스토리 eval
    python run_eval.py --chunks kh_v3    --ctype
    python run_eval.py --chunks kh_v3    --hybrid --ctype
    python run_eval.py --chunks chunks_all --ctype
    python run_eval.py --chunks chunks_all --hybrid --ctype

    # 시나리오 지정 (기본: A-1 A-2 B 전체)
    python run_eval.py --chunks kh_v3 --scenarios A-1 A-2
"""

import sys
import ast
import json as _json
import argparse
from pathlib import Path

_hej = Path('/mnt/gukrul/hej')
if str(_hej) not in sys.path: sys.path.insert(0, str(_hej))

import pandas as pd
from tqdm import tqdm
from retrieval_eval_F import BidMateEvaluator, normalize_fn, calc_hit, calc_mrr, calc_ndcg

# ======================================================================
# 청크 파일 매핑
# ======================================================================
CHUNKS_MAP = {
    'kh_fixed_v2' : '/mnt/gukrul/dataset/chunks/kh_fixed_1200_200_v2.json',
    'kh_fixed_v1' : '/mnt/gukrul/dataset/chunks/kh_fixed_1000_150_v1.json',
    'kh_v3'       : '/mnt/gukrul/dataset/chunks/kh_v3.json',
    'chunks_all'  : '/mnt/gukrul/dataset/chunks/chunks_all.json',
}

EVAL_FILE  = Path('/mnt/gukrul/dataset/eval/eval_retrieval_579.csv')
RESULT_DIR = Path('/mnt/gukrul/dataset/eval_results')
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================
# 히스토리 파싱 (C타입용)
# ======================================================================
def parse_history(raw):
    if not raw or isinstance(raw, float): return []
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw or raw in ['[]', 'null']: return []
        try: parsed = _json.loads(raw)
        except: return []
    else: parsed = raw
    result = []
    for h in parsed:
        if not isinstance(h, dict): continue
        if 'role' in h and 'content' in h:
            result.append({'role': h['role'], 'content': h['content']})
        elif 'user' in h:
            result.append({'role': 'user', 'content': h['user']})
    return result


# ======================================================================
# 결과 저장 (버전 자동 증가)
# ======================================================================
def save_result(df, pattern, save_path_fn):
    existing = list(RESULT_DIR.glob(pattern))
    nums     = [int(f.stem.split('_v')[-1]) for f in existing if f.stem.split('_v')[-1].isdigit()]
    next_v   = max(nums) + 1 if nums else 1
    path     = save_path_fn(next_v)
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f'✅ 저장 : {path}')


# ======================================================================
# 전체 eval 루프
# ======================================================================
def run_full_eval(eval_df, chunks_tag, chunks_path, scenarios, use_hybrid, eval_tag):
    hybrid_tag  = 'hybrid_ck_' if use_hybrid else ''
    all_results = {}

    for scenario in scenarios:
        print(f'\n{"="*60}')
        print(f'🚀 SCENARIO {scenario} [{chunks_tag}] {"(hybrid)" if use_hybrid else ""}')
        print(f'{"="*60}')

        col_name = f'bidmate_{chunks_tag}_{scenario}'
        ev = BidMateEvaluator(
            scenario        = scenario,
            chunks_path     = chunks_path,
            collection_name = col_name,
            use_hybrid      = use_hybrid,
        )

        results_log = []
        for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc=f'{scenario} Eval'):
            query  = row['question']
            gt_raw = row.get('ground_truth_docs', '')
            try:    gt_docs = ast.literal_eval(gt_raw) if isinstance(gt_raw, str) else []
            except: gt_docs = [gt_raw] if isinstance(gt_raw, str) and gt_raw else []
            mf_raw = row.get('metadata_filter', {})
            try:    meta_filter = ast.literal_eval(mf_raw) if isinstance(mf_raw, str) else {}
            except: meta_filter = {}

            result    = ev.retrieve(query, meta_filter=meta_filter or None)
            ret_stems = [normalize_fn(c['metadata'].get('source_file', '')) for c in result['top_chunks']]
            gt_stems  = [normalize_fn(g) for g in gt_docs]

            results_log.append({
                'id': row['id'], 'type': row['type'], 'difficulty': row['difficulty'],
                'hit' : calc_hit(gt_stems, ret_stems),
                'mrr' : calc_mrr(gt_stems, ret_stems),
                'ndcg': calc_ndcg(gt_stems, ret_stems),
                'retrieved': ret_stems, 'gt_docs': gt_stems, 'sub_queries': result['sub_queries'],
            })

        ev.release()

        result_df   = pd.DataFrame(results_log)
        eval_subset = result_df[result_df['hit'].notna()]
        oh, om, on_ = eval_subset['hit'].mean(), eval_subset['mrr'].mean(), eval_subset['ndcg'].mean()
        all_results[scenario] = {'hit': oh, 'mrr': om, 'ndcg': on_}

        print(f'\n📊 {scenario} [{chunks_tag}]  Hit@5={oh:.4f}  MRR={om:.4f}  nDCG={on_:.4f}')
        print('[타입별]')
        for t, g in eval_subset.groupby('type'):
            print(f'  {t}  Hit={g["hit"].mean():.4f} MRR={g["mrr"].mean():.4f} nDCG={g["ndcg"].mean():.4f}  n={len(g)}')
        print('[난이도별]')
        for d, g in eval_subset.groupby('difficulty'):
            print(f'  {d}  Hit={g["hit"].mean():.4f} MRR={g["mrr"].mean():.4f} nDCG={g["ndcg"].mean():.4f}  n={len(g)}')

        save_result(
            result_df,
            pattern      = f'eval_results_{hybrid_tag}{chunks_tag}_{scenario}_{eval_tag}_v*.csv',
            save_path_fn = lambda v: RESULT_DIR / f'eval_results_{hybrid_tag}{chunks_tag}_{scenario}_{eval_tag}_v{v}.csv',
        )

    print(f'\n{"="*60}\n📊 [{chunks_tag}{"(hybrid)" if use_hybrid else ""}] 임베딩 모델 비교 요약 ({eval_tag})\n{"="*60}')
    print(f'{"시나리오":>8}  {"Hit@5":>7} {"MRR":>7} {"nDCG":>7}')
    for sc, res in all_results.items():
        print(f'  {sc:>6}  {res["hit"]:>7.4f} {res["mrr"]:>7.4f} {res["ndcg"]:>7.4f}')


# ======================================================================
# C타입 히스토리 eval 루프
# ======================================================================
def run_ctype_eval(eval_df, chunks_tag, chunks_path, scenarios, use_hybrid, eval_tag):
    hybrid_tag = 'hybrid_ck_' if use_hybrid else ''

    c_df = eval_df[eval_df['type'] == 'C'].copy()
    c_df['parsed_history'] = c_df['history'].apply(parse_history)
    c_hist = c_df[c_df['parsed_history'].apply(lambda h: len(h) > 0)].copy()
    print(f'✅ C타입 히스토리 있음 : {len(c_hist)}개')

    for scenario in scenarios:
        print(f'\n{"="*60}')
        print(f'🚀 C타입 히스토리 — {scenario} [{chunks_tag}] {"(hybrid)" if use_hybrid else ""}')
        print(f'{"="*60}')

        col_name = f'bidmate_{chunks_tag}_{scenario}'
        ev = BidMateEvaluator(
            scenario        = scenario,
            chunks_path     = chunks_path,
            collection_name = col_name,
            use_hybrid      = use_hybrid,
        )

        results_c = []
        for _, row in tqdm(c_hist.iterrows(), total=len(c_hist), desc=f'C타입 {scenario}'):
            query   = row['question']
            history = row['parsed_history']
            gt_raw  = row.get('ground_truth_docs', '')
            try:    gt_docs = ast.literal_eval(gt_raw) if isinstance(gt_raw, str) else []
            except: gt_docs = [gt_raw] if isinstance(gt_raw, str) and gt_raw else []
            mf_raw = row.get('metadata_filter', {})
            try:    meta_filter = ast.literal_eval(mf_raw) if isinstance(mf_raw, str) else {}
            except: meta_filter = {}

            prev_user = [h['content'] for h in history if h['role'] == 'user']
            effective = f'{prev_user[-1]} {query}' if prev_user else query
            result    = ev.retrieve(effective, meta_filter=meta_filter or None)
            ret_stems = [normalize_fn(c['metadata'].get('source_file', '')) for c in result['top_chunks']]
            gt_stems  = [normalize_fn(g) for g in gt_docs]

            results_c.append({
                'id': row['id'], 'difficulty': row['difficulty'],
                'hit' : calc_hit(gt_stems, ret_stems),
                'mrr' : calc_mrr(gt_stems, ret_stems),
                'ndcg': calc_ndcg(gt_stems, ret_stems),
            })

        ev.release()

        c_df2 = pd.DataFrame(results_c)
        c_sub = c_df2[c_df2['hit'].notna()]
        c_hit, c_mrr, c_ndcg = c_sub['hit'].mean(), c_sub['mrr'].mean(), c_sub['ndcg'].mean()

        print(f'\n📊 {scenario} [{chunks_tag}]  Hit@5={c_hit:.4f}  MRR={c_mrr:.4f}  nDCG={c_ndcg:.4f}')
        print('[난이도별]')
        for d, g in c_sub.groupby('difficulty'):
            print(f'  {d}  Hit={g["hit"].mean():.4f} MRR={g["mrr"].mean():.4f} nDCG={g["ndcg"].mean():.4f}  n={len(g)}')

        save_result(
            c_df2,
            pattern      = f'eval_results_{hybrid_tag}C타입_{chunks_tag}_{scenario}_{eval_tag}_v*.csv',
            save_path_fn = lambda v: RESULT_DIR / f'eval_results_{hybrid_tag}C타입_{chunks_tag}_{scenario}_{eval_tag}_v{v}.csv',
        )


# ======================================================================
# main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description='입찰메이트 Retrieval 배치 평가')
    parser.add_argument('--chunks',    default='kh_fixed_v2', choices=list(CHUNKS_MAP.keys()),
                        help='청크 전략 선택 (기본: kh_fixed_v2)')
    parser.add_argument('--hybrid',    action='store_true',
                        help='Child-to-Parent Retrieval 사용')
    parser.add_argument('--ctype',     action='store_true',
                        help='C타입 히스토리 평가 실행')
    
    # parser.add_argument('--scenarios', nargs='+', default=['A-1', 'A-2', 'B'],
    #                     choices=['A-1', 'A-2', 'B'],
    #                     help='평가할 시나리오 (기본: A-1 A-2 B)')
    # 수정
    _ALL_SCENARIOS = [
        f'{e}_{l}' for e in ['KURE', 'KOE5', 'SMALL']
                for l in ['GEMMA', 'QWEN', 'PHI', 'OPENAI']
    ]
    parser.add_argument('--scenarios', nargs='+', default=_ALL_SCENARIOS,
                        choices=_ALL_SCENARIOS)
    
    args = parser.parse_args()

    chunks_tag  = args.chunks
    chunks_path = CHUNKS_MAP[chunks_tag]
    eval_tag    = EVAL_FILE.stem.split('_')[-1]  # '579'

    eval_df = pd.read_csv(EVAL_FILE)
    print(f'✅ Eval 데이터 : {len(eval_df):,}개 ({eval_tag})')
    print(f'✅ 청크        : {chunks_tag}')
    print(f'✅ Hybrid      : {args.hybrid}')
    print(f'✅ C타입       : {args.ctype}')
    print(f'✅ 시나리오    : {args.scenarios}')

    if args.ctype:
        run_ctype_eval(eval_df, chunks_tag, chunks_path, args.scenarios, args.hybrid, eval_tag)
    else:
        run_full_eval(eval_df, chunks_tag, chunks_path, args.scenarios, args.hybrid, eval_tag)


if __name__ == '__main__':
    main()
