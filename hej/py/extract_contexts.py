"""
입찰메이트 RAG — Retrieval Context 사전 추출
=============================================
담당 : Retrieval 파트 (한의정)

[설명]
- 579개 eval 쿼리에 대해 Retrieval Context를 한번만 추출하여 저장.
- 이후 E2E 평가 시 get_context() 재호출 없이 CSV에서 읽어 재사용.
- A/B/D/E타입 → chunks_all retriever
- C타입        → kh_v3_hybrid retriever
- 임베딩 모델별로 파일명 구분하여 저장.

[실행]
    BIDMATE_ENV=gcp python extract_contexts.py --embed KURE
    BIDMATE_ENV=gcp python extract_contexts.py --embed KOE5
    BIDMATE_ENV=gcp python extract_contexts.py --embed SMALL

[출력]
    eval_contexts_abde_{embed}.csv  ← A/B/D/E타입 context
    eval_contexts_c_{embed}.csv     ← C타입 context
"""

import os
import argparse
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from retrieval_interface_F import (
    BidMateRetriever, EMBED_CONFIG,
    _load_chunks, _build_embed_model,
    _build_or_load_chroma, _build_or_load_bm25,
    _parse_meta_filter_with, _RERANK_MODEL,
)
from sentence_transformers import CrossEncoder
import chromadb
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
_logger = logging.getLogger(__name__)

# ======================================================================
# 환경 경로
# ======================================================================
try:
    import google.colab; _ENV = 'colab'
except ImportError:
    _ENV = os.environ.get('BIDMATE_ENV', 'gcp')

if _ENV == 'gcp':
    _CHUNKS_ABDE  = Path('/mnt/gukrul/dataset/chunks/chunks_all.json')
    _CHUNKS_C     = Path('/mnt/gukrul/dataset/chunks/kh_v3.json')
    _CHROMA_PATH  = Path('/mnt/gukrul/dataset/hej/chroma_db')
    _BM25_DIR     = Path('/mnt/gukrul/dataset/bm25')
else:
    _CHUNKS_ABDE  = Path.home() / 'Desktop' / 'AI엔지니어_8기' / '프로젝트' / '중급' / 'data' / 'chunks' / 'chunks_all.json'
    _CHUNKS_C     = Path.home() / 'Desktop' / 'AI엔지니어_8기' / '프로젝트' / '중급' / 'data' / 'chunks' / 'kh_v3.json'
    _CHROMA_PATH  = Path.home() / 'Desktop' / 'AI엔지니어_8기' / '프로젝트' / '중급' / 'chroma_db'
    _BM25_DIR     = Path.home() / 'Desktop' / 'AI엔지니어_8기' / '프로젝트' / '중급' / 'data'

_EVAL_FILE   = Path('/mnt/gukrul/dataset/eval/eval_retrieval_579.csv')
_RESULT_DIR  = Path('/mnt/gukrul/dataset/eval_results/generation')
_RESULT_DIR.mkdir(parents=True, exist_ok=True)

if torch.cuda.is_available():           _DEVICE = 'cuda'
elif torch.backends.mps.is_available(): _DEVICE = 'mps'
else:                                   _DEVICE = 'cpu'


# ======================================================================
# Retriever 빌드
# ======================================================================
def build_retriever(chunks_path: Path, collection_name: str,
                    embed_scenario: str, use_hybrid: bool = False):
    _logger.info(f'Retriever 초기화 | collection={collection_name} | hybrid={use_hybrid}')
    client      = chromadb.PersistentClient(path=str(_CHROMA_PATH))
    chunks      = _load_chunks(chunks_path)
    embed_model = _build_embed_model(embed_scenario)
    col         = _build_or_load_chroma(client, collection_name, chunks, embed_model)
    bm25_path   = _BM25_DIR / f'bm25_index_{collection_name}.pkl'
    bm25_idx, bm25_cids, bm25_texts = _build_or_load_bm25(chunks, bm25_path)
    reranker    = CrossEncoder(_RERANK_MODEL, device=_DEVICE)

    retriever = BidMateRetriever(
        collection     = col,
        bm25_index     = bm25_idx,
        bm25_chunk_ids = bm25_cids,
        bm25_texts     = bm25_texts,
        embed_model    = embed_model,
        all_chunks     = chunks,
        reranker       = reranker,
        use_hybrid     = use_hybrid,
    )
    return retriever, embed_model, reranker


# ======================================================================
# Context 추출
# ======================================================================
def extract_contexts(retriever, rows: pd.DataFrame, history_col: bool = False) -> list:
    records = []
    for _, row in tqdm(rows.iterrows(), total=len(rows), desc='Context 추출'):
        query = row['question']

        if history_col:
            import json
            raw = row.get('history', '')
            try:
                history = json.loads(raw) if isinstance(raw, str) and raw not in ['', '[]', 'null'] else []
            except:
                history = []
            prev_user = [h['content'] for h in history if h.get('role') == 'user']
            effective_query = f'{prev_user[-1]} {query}' if prev_user else query
        else:
            effective_query = query
            history = []

        meta_filter = _parse_meta_filter_with(query, retriever._all_agencies)
        result      = retriever.retrieve(effective_query, meta_filter=meta_filter)
        context     = '\n\n'.join([
            c['text'] for c in result['top_chunks']
        ])

        records.append({
            'question'         : query,
            'type'             : row['type'],
            'difficulty'       : row.get('difficulty', ''),
            'retrieved_context': context,
            'sub_queries'      : str(result['sub_queries']),
        })

    return records


# ======================================================================
# 메인
# ======================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed', default='KURE', choices=['KURE', 'KOE5', 'SMALL'],
                        help='임베딩 모델 선택 (기본값: KURE)')
    args = parser.parse_args()

    embed    = args.embed.upper()
    embed_lc = embed.lower()

    # 임베딩별 컬렉션명 매핑
    _COLLECTION_MAP = {
        'KURE' : 'bidmate_kure',
        'KOE5' : 'bidmate_chunks_all_A-2',
        'SMALL': 'bidmate_chunks_all_B',
    }
    _COLLECTION_C_MAP = {
        'KURE' : 'bidmate_kh_v3_A-1',
        'KOE5' : 'bidmate_kh_v3_A-2',
        'SMALL': 'bidmate_kh_v3_B',
    }

    _logger.info(f'임베딩 모델: {embed}')
    _logger.info('eval 데이터 로드')
    df = pd.read_csv(_EVAL_FILE)

    abde_df = df[df['type'].isin(['A', 'B', 'D', 'E'])].copy()
    c_df    = df[df['type'] == 'C'].copy()
    _logger.info(f'A/B/D/E: {len(abde_df)}개 | C: {len(c_df)}개')

    # ── A/B/D/E: chunks_all retriever ───────────────────────────────
    out_abde = _RESULT_DIR / f'eval_contexts_abde_{embed_lc}.csv'
    if out_abde.exists():
        _logger.info(f'이미 존재: {out_abde} → 건너뜀')
    else:
        _logger.info(f'A/B/D/E context 추출 시작 (chunks_all, embed={embed})')
        retriever_abde, em, rr = build_retriever(
            chunks_path     = _CHUNKS_ABDE,
            collection_name = _COLLECTION_MAP[embed],
            embed_scenario  = embed,
            use_hybrid      = False,
        )
        records_abde = extract_contexts(retriever_abde, abde_df, history_col=False)
        pd.DataFrame(records_abde).to_csv(out_abde, index=False, encoding='utf-8-sig')
        _logger.info(f'✅ 저장: {out_abde}')
        del em, rr
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()

    # ── C타입: kh_v3_hybrid retriever ───────────────────────────────
    out_c = _RESULT_DIR / f'eval_contexts_c_{embed_lc}.csv'
    if out_c.exists():
        _logger.info(f'이미 존재: {out_c} → 건너뜀')
    else:
        _logger.info(f'C타입 context 추출 시작 (kh_v3_hybrid, embed={embed})')
        retriever_c, em, rr = build_retriever(
            chunks_path     = _CHUNKS_C,
            collection_name = _COLLECTION_C_MAP[embed],
            embed_scenario  = embed,
            use_hybrid      = True,
        )
        records_c = extract_contexts(retriever_c, c_df, history_col=True)
        pd.DataFrame(records_c).to_csv(out_c, index=False, encoding='utf-8-sig')
        _logger.info(f'✅ 저장: {out_c}')

    _logger.info('전체 완료')
    _logger.info(f'  A/B/D/E context: {out_abde}')
    _logger.info(f'  C타입 context  : {out_c}')


if __name__ == '__main__':
    main()
