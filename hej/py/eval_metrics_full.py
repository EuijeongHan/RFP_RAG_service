"""
입찰메이트 RAG — 통합 평가 메트릭
====================================
담당 : Generation + Retrieval 파트 (한의정)

[설명]
멘토님 평가 메트릭 가이드 기반 + 입찰메이트 자체 지표 통합:

  섹션 1 : Retrieval 지표       — Hit@5 / MRR / nDCG (타입별)
  섹션 2 : Generation 정량 지표  — Faithfulness / Relevance / Rejection /
                                    Context Precision / Context Recall (Judge CSV)
  섹션 3 : BLEU / ROUGE          — FT 답변 vs OPENAI 답변 (pseudo GT)
  섹션 4 : BERTScore             — 의미 유사도
  섹션 5 : Perplexity (PPL)      — PHI 모델로 생성 답변 PPL 계산
  섹션 6 : 토큰 측정             — tiktoken 입출력 토큰 수 / 비용 추정
  섹션 7 : 생성 속도             — Tokens/sec (로그 파싱)
  섹션 8 : 종합 요약 출력

[입력 파일]
  E2E_PATH     : e2e_all_comparison_ft.csv      (FT 생성 결과)
  SCORE_PATH   : quantitative_scores_ft.csv     (Judge 채점 결과)
  RETRIEVAL_CSV: eval_results_chunks_all_*.csv  (Retrieval 평가 결과)

[실행]
  python eval_metrics_full.py
  python eval_metrics_full.py --skip_ppl        # PPL 계산 스킵 (빠름)
  python eval_metrics_full.py --skip_bertscore  # BERTScore 스킵
  python eval_metrics_full.py --section 3       # 특정 섹션만 실행

[의존성]
  pip install nltk rouge-score bert-score tiktoken transformers torch pandas
"""

import os
import re
import gc
import math
import json
import time
import logging
import argparse
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
_logger = logging.getLogger(__name__)

# ======================================================================
# 경로 설정
# ======================================================================
_BASE       = Path('/mnt/gukrul/hej')
_RESULT_DIR = _BASE / 'eval_results'
_GEN_DIR    = _RESULT_DIR / 'generation'

E2E_PATH      = Path(os.environ.get('E2E_PATH',     str(_GEN_DIR / 'e2e_all_comparison_ft.csv')))
SCORE_PATH    = Path(os.environ.get('SCORE_PATH',   str(_GEN_DIR / 'quant/FT/quantitative_scores_ft.csv')))
RETRIEVAL_CSV = Path(os.environ.get('RETRIEVAL_CSV',str(_RESULT_DIR / 'eval_results_chunks_all_KURE_GEMMA_579_v1.csv')))
OUTPUT_DIR    = _GEN_DIR / 'metrics_report'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ======================================================================
# FT 시나리오 정의
# ======================================================================
_FT_SCENARIOS = [
    'ans_kure_phi_ft',
    'ans_kure_gemma_ft',
    'ans_small_phi_ft',
    'ans_small_gemma_ft',
]
_OPENAI_COL = 'ans_small_openai'  # pseudo Ground Truth

_JUDGE_METRICS = [
    'faithfulness', 'relevance', 'rejection',
    'context_precision', 'context_recall',
]

# ======================================================================
# 유틸
# ======================================================================
def _sep(char='─', width=64):
    print(char * width)

def _header(title: str):
    print()
    _sep('═')
    print(f'  {title}')
    _sep('═')

def _safe_mean(series) -> Optional[float]:
    v = series.dropna()
    return float(v.mean()) if len(v) > 0 else None

def _fmt(val, fmt='.3f') -> str:
    return f'{val:{fmt}}' if val is not None and not (isinstance(val, float) and math.isnan(val)) else 'N/A'


# ======================================================================
# 섹션 1: Retrieval 지표 (Hit@5 / MRR / nDCG)
# ======================================================================
def section1_retrieval():
    _header('섹션 1 │ Retrieval 지표 (Hit@5 / MRR / nDCG)')

    if not RETRIEVAL_CSV.exists():
        print(f'  ⚠️  파일 없음: {RETRIEVAL_CSV}')
        print(f'  → eval_results/ 디렉토리에서 chunks_all CSV를 지정하세요.')
        return None

    df = pd.read_csv(RETRIEVAL_CSV)
    required = {'hit', 'mrr', 'ndcg', 'type'}
    if not required.issubset(df.columns):
        print(f'  ⚠️  필수 컬럼 없음: {required - set(df.columns)}')
        return None

    sub = df[df['hit'].notna()]
    overall = {
        'Hit@5': _safe_mean(sub['hit']),
        'MRR'  : _safe_mean(sub['mrr']),
        'nDCG' : _safe_mean(sub['ndcg']),
    }

    print(f'\n  [전체]')
    _sep(width=48)
    for k, v in overall.items():
        bar = '█' * int((v or 0) * 30)
        print(f'  {k:<8} {_fmt(v)}  {bar}')

    print(f'\n  [타입별 MRR]')
    _sep(width=48)
    type_label = {'A': '추출정밀도', 'B': '종합능력', 'C': '맥락유지',
                  'D': '환각방지', 'E': '오타질문'}
    results = {}
    for t in ['A', 'B', 'C', 'D', 'E']:
        tdf = sub[sub['type'] == t]
        if tdf.empty:
            continue
        mrr = _safe_mean(tdf['mrr'])
        results[t] = mrr
        bar = '█' * int((mrr or 0) * 30)
        print(f'  {t}타입 ({type_label.get(t,""):<6}) MRR={_fmt(mrr)}  {bar}  n={len(tdf)}')

    # CSV 저장
    out = pd.DataFrame([{'metric': k, 'value': v} for k, v in overall.items()])
    out.to_csv(OUTPUT_DIR / 'retrieval_metrics.csv', index=False, encoding='utf-8-sig')
    _logger.info(f'저장: {OUTPUT_DIR}/retrieval_metrics.csv')
    return overall


# ======================================================================
# 섹션 2: Generation 정량 지표 (Judge CSV)
# ======================================================================
def section2_judge_scores():
    _header('섹션 2 │ Generation 정량 지표 (LLM-as-Judge)')

    if not SCORE_PATH.exists():
        print(f'  ⚠️  파일 없음: {SCORE_PATH}')
        print(f'  → Judge 채점 완료 후 실행하세요.')
        return None

    df = pd.read_csv(SCORE_PATH)
    rows = []

    print(f'\n  {"시나리오":<22}', end='')
    for m in _JUDGE_METRICS:
        print(f'  {m[:8]:>8}', end='')
    print()
    _sep(width=70)

    for col in _FT_SCENARIOS:
        prefix = col.replace('ans_', '')
        row = {'scenario': prefix}
        print(f'  {prefix:<22}', end='')
        for metric in _JUDGE_METRICS:
            score_col = f'{prefix}_avg_{metric}'
            val = _safe_mean(df[score_col]) if score_col in df.columns else None
            row[metric] = val
            print(f'  {_fmt(val):>8}', end='')
        print()
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / 'judge_scores.csv', index=False, encoding='utf-8-sig')
    _logger.info(f'저장: {OUTPUT_DIR}/judge_scores.csv')
    return out


# ======================================================================
# 섹션 3: BLEU / ROUGE
# ======================================================================
def section3_bleu_rouge():
    _header('섹션 3 │ BLEU / ROUGE (FT 답변 vs OPENAI pseudo-GT)')

    if not E2E_PATH.exists():
        print(f'  ⚠️  파일 없음: {E2E_PATH}')
        return None

    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from rouge_score import rouge_scorer
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except ImportError as e:
        print(f'  ⚠️  라이브러리 없음: {e}')
        print('  → pip install nltk rouge-score')
        return None

    df = pd.read_csv(E2E_PATH)
    if _OPENAI_COL not in df.columns:
        print(f'  ⚠️  {_OPENAI_COL} 컬럼 없음')
        return None

    smooth  = SmoothingFunction().method1
    r_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

    rows = []
    for col in _FT_SCENARIOS:
        if col not in df.columns:
            continue
        prefix = col.replace('ans_', '')
        bleu1_list, bleu4_list = [], []
        r1_list, r2_list, rL_list = [], [], []

        for _, row in df.iterrows():
            hyp = str(row.get(col, '') or '')
            ref = str(row.get(_OPENAI_COL, '') or '')
            if not hyp or not ref or '생성 오류' in hyp:
                continue

            # BLEU
            ref_tok = ref.split()
            hyp_tok = hyp.split()
            if ref_tok and hyp_tok:
                bleu1_list.append(sentence_bleu([ref_tok], hyp_tok, weights=(1,0,0,0), smoothing_function=smooth))
                bleu4_list.append(sentence_bleu([ref_tok], hyp_tok, weights=(.25,.25,.25,.25), smoothing_function=smooth))

            # ROUGE
            scores = r_scorer.score(ref, hyp)
            r1_list.append(scores['rouge1'].fmeasure)
            r2_list.append(scores['rouge2'].fmeasure)
            rL_list.append(scores['rougeL'].fmeasure)

        row_result = {
            'scenario': prefix,
            'BLEU-1'  : float(np.mean(bleu1_list)) if bleu1_list else None,
            'BLEU-4'  : float(np.mean(bleu4_list)) if bleu4_list else None,
            'ROUGE-1' : float(np.mean(r1_list))     if r1_list   else None,
            'ROUGE-2' : float(np.mean(r2_list))     if r2_list   else None,
            'ROUGE-L' : float(np.mean(rL_list))     if rL_list   else None,
        }
        rows.append(row_result)

    out = pd.DataFrame(rows)
    print(f'\n  {"시나리오":<22}  {"BLEU-1":>7}  {"BLEU-4":>7}  {"ROUGE-1":>8}  {"ROUGE-2":>8}  {"ROUGE-L":>8}')
    _sep(width=72)
    for _, r in out.iterrows():
        print(f'  {r["scenario"]:<22}  {_fmt(r["BLEU-1"]):>7}  {_fmt(r["BLEU-4"]):>7}'
              f'  {_fmt(r["ROUGE-1"]):>8}  {_fmt(r["ROUGE-2"]):>8}  {_fmt(r["ROUGE-L"]):>8}')

    out.to_csv(OUTPUT_DIR / 'bleu_rouge.csv', index=False, encoding='utf-8-sig')
    _logger.info(f'저장: {OUTPUT_DIR}/bleu_rouge.csv')
    return out


# ======================================================================
# 섹션 4: BERTScore
# ======================================================================
def section4_bertscore():
    _header('섹션 4 │ BERTScore (의미적 유사도)')

    if not E2E_PATH.exists():
        print(f'  ⚠️  파일 없음: {E2E_PATH}')
        return None

    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        print('  ⚠️  라이브러리 없음: bert-score')
        print('  → pip install bert-score')
        return None

    df = pd.read_csv(E2E_PATH)
    if _OPENAI_COL not in df.columns:
        print(f'  ⚠️  {_OPENAI_COL} 컬럼 없음')
        return None

    rows = []
    for col in _FT_SCENARIOS:
        if col not in df.columns:
            continue
        prefix = col.replace('ans_', '')

        valid = df[[col, _OPENAI_COL]].dropna()
        valid = valid[~valid[col].astype(str).str.contains('생성 오류', na=False)]
        if valid.empty:
            continue

        hyps = valid[col].astype(str).tolist()
        refs = valid[_OPENAI_COL].astype(str).tolist()

        _logger.info(f'BERTScore 계산 중: {prefix} ({len(hyps)}개)')
        P, R, F1 = bert_score_fn(hyps, refs, lang='ko', verbose=False)

        row_result = {
            'scenario'      : prefix,
            'BERTScore_P'   : float(P.mean()),
            'BERTScore_R'   : float(R.mean()),
            'BERTScore_F1'  : float(F1.mean()),
        }
        rows.append(row_result)
        print(f'  {prefix:<22}  P={_fmt(row_result["BERTScore_P"])}  '
              f'R={_fmt(row_result["BERTScore_R"])}  F1={_fmt(row_result["BERTScore_F1"])}')

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / 'bertscore.csv', index=False, encoding='utf-8-sig')
    _logger.info(f'저장: {OUTPUT_DIR}/bertscore.csv')
    return out


# ======================================================================
# 섹션 5: Perplexity (PPL)
# ======================================================================
def section5_perplexity(n_samples: int = 50):
    _header(f'섹션 5 │ Perplexity (PHI 모델 기준, 샘플 {n_samples}개)')

    if not E2E_PATH.exists():
        print(f'  ⚠️  파일 없음: {E2E_PATH}')
        return None

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        print('  ⚠️  라이브러리 없음: transformers torch')
        return None

    model_id = 'microsoft/Phi-4-mini-instruct'
    hf_home  = os.environ.get('HF_HOME', '/mnt/gukrul/hf_cache')

    _logger.info(f'PPL 모델 로드: {model_id}')
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=hf_home)
    model     = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map='auto', cache_dir=hf_home
    )
    model.eval()

    df = pd.read_csv(E2E_PATH).head(n_samples)
    rows = []

    for col in _FT_SCENARIOS:
        if col not in df.columns:
            continue
        prefix  = col.replace('ans_', '')
        ppls    = []

        for text in df[col].dropna().astype(str):
            if '생성 오류' in text or len(text.strip()) < 5:
                continue
            try:
                enc  = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                ids  = enc['input_ids'].to(device)
                with torch.no_grad():
                    loss = model(ids, labels=ids).loss
                ppls.append(torch.exp(loss).item())
            except Exception as e:
                _logger.warning(f'PPL 계산 실패: {e}')

        avg_ppl = float(np.mean(ppls)) if ppls else None
        rows.append({'scenario': prefix, 'PPL': avg_ppl, 'n': len(ppls)})
        print(f'  {prefix:<22}  PPL={_fmt(avg_ppl, ".2f")}  (n={len(ppls)})')

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / 'perplexity.csv', index=False, encoding='utf-8-sig')
    _logger.info(f'저장: {OUTPUT_DIR}/perplexity.csv')
    return out


# ======================================================================
# 섹션 6: 토큰 측정 (tiktoken)
# ======================================================================
def section6_token_metrics():
    _header('섹션 6 │ 토큰 측정 (입출력 토큰 수 / 비용 추정)')

    if not E2E_PATH.exists():
        print(f'  ⚠️  파일 없음: {E2E_PATH}')
        return None

    try:
        import tiktoken
    except ImportError:
        print('  ⚠️  라이브러리 없음: tiktoken')
        print('  → pip install tiktoken')
        return None

    # gpt-5-mini 기준 가격 (2025 기준, 참고용)
    INPUT_PRICE  = 0.15 / 1_000_000   # $0.15 per 1M input tokens
    OUTPUT_PRICE = 0.60 / 1_000_000   # $0.60 per 1M output tokens

    enc = tiktoken.get_encoding('o200k_base')
    df  = pd.read_csv(E2E_PATH)

    rows = []
    print(f'\n  {"시나리오":<22}  {"avg_out_tok":>12}  {"avg_in_tok":>11}  {"est_cost_$":>11}  {"n":>5}')
    _sep(width=70)

    for col in _FT_SCENARIOS + [_OPENAI_COL]:
        if col not in df.columns:
            continue
        prefix = col.replace('ans_', '')

        in_tok_list, out_tok_list = [], []
        for _, row in df.iterrows():
            ctx = str(row.get('retrieved_context', '') or '')
            ans = str(row.get(col, '') or '')
            if not ans or '생성 오류' in ans:
                continue
            in_tok_list.append(len(enc.encode(ctx)))
            out_tok_list.append(len(enc.encode(ans)))

        if not out_tok_list:
            continue

        avg_in  = float(np.mean(in_tok_list))
        avg_out = float(np.mean(out_tok_list))
        # OPENAI만 비용 계산 (로컬 모델은 GPU 비용이므로 별도)
        est_cost = (avg_in * INPUT_PRICE + avg_out * OUTPUT_PRICE) * len(out_tok_list) \
                   if col == _OPENAI_COL else None

        rows.append({
            'scenario'   : prefix,
            'avg_in_tok' : avg_in,
            'avg_out_tok': avg_out,
            'est_cost_$' : est_cost,
            'n'          : len(out_tok_list),
        })
        cost_str = f'${est_cost:.4f}' if est_cost else 'N/A(로컬)'
        print(f'  {prefix:<22}  {avg_out:>12.1f}  {avg_in:>11.1f}  {cost_str:>11}  {len(out_tok_list):>5}')

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / 'token_metrics.csv', index=False, encoding='utf-8-sig')
    _logger.info(f'저장: {OUTPUT_DIR}/token_metrics.csv')
    return out


# ======================================================================
# 섹션 7: 생성 속도 (로그 파싱)
# ======================================================================
def section7_generation_speed():
    _header('섹션 7 │ 생성 속도 (Tokens/sec, 로그 파싱)')

    # v11 실험 히스토리 기반 고정값 (실측)
    speed_data = [
        {'model': 'gemma-4-E4B-it',       'scenario': 'GEMMA', 'sec_per_item': 44.5,  'n': 516},
        {'model': 'Qwen2.5-7B-Instruct',  'scenario': 'QWEN',  'sec_per_item': 6.5,   'n': 516},
        {'model': 'Phi-4-mini-instruct',   'scenario': 'PHI',   'sec_per_item': 2.25,  'n': 516},
        {'model': 'gpt-5-mini (API)',       'scenario': 'OPENAI','sec_per_item': 6.0,   'n': 516},
    ]

    # 로그 파일에서 실측값 파싱 시도
    log_patterns = {
        'PHI'  : _BASE / 'hej' / 'abde_kure_phi_ft_run.log',
        'GEMMA': _BASE / 'hej' / 'abde_kure_gemma_ft_run.log',
    }

    parsed = {}
    for scenario, log_path in log_patterns.items():
        if not log_path.exists():
            continue
        try:
            text = log_path.read_text(errors='ignore')
            # tqdm 형식: [HH:MM<MM:SS, X.XXs/it]
            matches = re.findall(r'(\d+\.\d+)s/it', text)
            if matches:
                vals = [float(m) for m in matches]
                parsed[scenario] = float(np.median(vals))
                _logger.info(f'{scenario} 실측 sec/item: {parsed[scenario]:.2f}s (로그 파싱)')
        except Exception as e:
            _logger.warning(f'로그 파싱 실패 ({scenario}): {e}')

    print(f'\n  {"시나리오":<10}  {"모델":<28}  {"sec/item":>9}  {"tok/sec(추정)":>14}  {"516개 소요":>10}')
    _sep(width=78)

    rows = []
    AVG_OUTPUT_TOKENS = 120  # 평균 출력 토큰 수 (섹션 6 결과 기반)
    for d in speed_data:
        sec = parsed.get(d['scenario'], d['sec_per_item'])
        tok_per_sec = AVG_OUTPUT_TOKENS / sec if sec > 0 else None
        total_min   = sec * d['n'] / 60
        print(f'  {d["scenario"]:<10}  {d["model"]:<28}  {sec:>9.2f}s  '
              f'{_fmt(tok_per_sec, ".1f"):>14}  {total_min:>8.0f}분')
        rows.append({
            'scenario'   : d['scenario'],
            'model'      : d['model'],
            'sec_per_item': sec,
            'tok_per_sec' : tok_per_sec,
            'total_min_516': total_min,
            'source'     : '실측(로그)' if d['scenario'] in parsed else '실험히스토리',
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / 'generation_speed.csv', index=False, encoding='utf-8-sig')
    _logger.info(f'저장: {OUTPUT_DIR}/generation_speed.csv')
    return out


# ======================================================================
# 섹션 8: 종합 요약
# ======================================================================
def section8_summary(results: dict):
    _header('섹션 8 │ 종합 요약')

    print('\n  ✅ 완료된 섹션:')
    for k, v in results.items():
        status = '완료' if v is not None else '스킵/오류'
        print(f'    {k}: {status}')

    print(f'\n  📁 결과 저장 위치: {OUTPUT_DIR}')
    saved = list(OUTPUT_DIR.glob('*.csv'))
    for f in sorted(saved):
        size = f.stat().st_size / 1024
        print(f'    {f.name:<40} {size:>8.1f} KB')

    print()
    _sep('═')
    print('  🏁 평가 완료')
    _sep('═')


# ======================================================================
# 메인
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description='입찰메이트 통합 평가 메트릭')
    parser.add_argument('--section',        type=int, default=0,
                        help='특정 섹션만 실행 (0=전체, 1~8)')
    parser.add_argument('--skip_ppl',       action='store_true',
                        help='Perplexity 계산 스킵 (GPU 메모리 절약)')
    parser.add_argument('--skip_bertscore', action='store_true',
                        help='BERTScore 계산 스킵')
    parser.add_argument('--ppl_samples',    type=int, default=50,
                        help='PPL 계산 샘플 수 (기본: 50)')
    args = parser.parse_args()

    only = args.section
    results = {}

    if only in (0, 1):
        results['섹션1_Retrieval']   = section1_retrieval()

    if only in (0, 2):
        results['섹션2_Judge']       = section2_judge_scores()

    if only in (0, 3):
        results['섹션3_BLEU_ROUGE']  = section3_bleu_rouge()

    if only in (0, 4):
        if args.skip_bertscore:
            print('\n  [섹션 4] BERTScore 스킵 (--skip_bertscore)')
            results['섹션4_BERTScore'] = None
        else:
            results['섹션4_BERTScore'] = section4_bertscore()

    if only in (0, 5):
        if args.skip_ppl:
            print('\n  [섹션 5] PPL 스킵 (--skip_ppl)')
            results['섹션5_PPL'] = None
        else:
            results['섹션5_PPL'] = section5_perplexity(n_samples=args.ppl_samples)

    if only in (0, 6):
        results['섹션6_토큰측정']    = section6_token_metrics()

    if only in (0, 7):
        results['섹션7_생성속도']    = section7_generation_speed()

    if only in (0, 8):
        section8_summary(results)


if __name__ == '__main__':
    main()


# ======================================================================
# [B] HTML 대시보드 생성 (주석 처리 — 필요 시 활성화)
# ======================================================================
# def generate_html_dashboard(output_dir: Path):
#     """
#     섹션 1~7 CSV 결과를 읽어 인터랙티브 HTML 대시보드 생성.
#     Chart.js 기반, 탭 구조.
#
#     활성화 방법:
#       1. 이 함수 주석 해제
#       2. main() 마지막에 generate_html_dashboard(OUTPUT_DIR) 추가
#
#     의존성: 추가 라이브러리 불필요 (Chart.js CDN 사용)
#     """
#     import json
#
#     # 각 섹션 CSV 로드
#     def _load(name):
#         p = output_dir / f'{name}.csv'
#         return pd.read_csv(p).to_dict('records') if p.exists() else []
#
#     retrieval  = _load('retrieval_metrics')
#     judge      = _load('judge_scores')
#     bleu_rouge = _load('bleu_rouge')
#     bertscore  = _load('bertscore')
#     token      = _load('token_metrics')
#     speed      = _load('generation_speed')
#
#     html = f"""<!DOCTYPE html>
# <html lang="ko">
# <head>
#   <meta charset="UTF-8">
#   <title>입찰메이트 RAG 평가 대시보드</title>
#   <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
#   <style>
#     body {{ font-family: 'Segoe UI', sans-serif; background:#0f1117; color:#e8eaf6; padding:24px; }}
#     h1 {{ color:#a89cf7; }} h2 {{ color:#4fc3f7; margin-top:32px; }}
#     table {{ border-collapse:collapse; width:100%; margin-bottom:24px; }}
#     th {{ background:#22263a; color:#9fa8c7; padding:10px; text-align:left; }}
#     td {{ border-bottom:1px solid #2e3357; padding:10px; color:#9fa8c7; }}
#     canvas {{ max-width:600px; margin:16px 0; }}
#   </style>
# </head>
# <body>
#   <h1>📊 입찰메이트 RAG 평가 대시보드</h1>
#   <p>BL(베이스라인) vs FT(파인튜닝) 비교 평가</p>
#
#   <h2>섹션 1: Retrieval 지표</h2>
#   <table>
#     <tr><th>지표</th><th>값</th></tr>
#     {''.join(f"<tr><td>{r['metric']}</td><td>{r['value']:.4f}</td></tr>" for r in retrieval)}
#   </table>
#
#   <h2>섹션 2: Judge 점수</h2>
#   <table>
#     <tr><th>시나리오</th><th>Faithfulness</th><th>Relevance</th><th>Rejection</th></tr>
#     {''.join(f"<tr><td>{r['scenario']}</td><td>{r.get('faithfulness','N/A')}</td><td>{r.get('relevance','N/A')}</td><td>{r.get('rejection','N/A')}</td></tr>" for r in judge)}
#   </table>
#
#   <script>
#     // Chart.js 예시 — Judge 점수 레이더 차트
#     const ctx = document.createElement('canvas');
#     document.body.appendChild(ctx);
#     new Chart(ctx, {{
#       type: 'radar',
#       data: {{
#         labels: ['Faithfulness','Relevance','Rejection','Context Precision','Context Recall'],
#         datasets: {json.dumps([
#             {{
#               'label': r['scenario'],
#               'data': [r.get('faithfulness'), r.get('relevance'), r.get('rejection'),
#                        r.get('context_precision'), r.get('context_recall')],
#             }} for r in judge
#         ])}
#       }},
#       options: {{ scales: {{ r: {{ min:1, max:5 }} }} }}
#     }});
#   </script>
# </body>
# </html>"""
#
#     out_path = output_dir / 'dashboard.html'
#     out_path.write_text(html, encoding='utf-8')
#     print(f'\n  🌐 HTML 대시보드 저장: {out_path}')
