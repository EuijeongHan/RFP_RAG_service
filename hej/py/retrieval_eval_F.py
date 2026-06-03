"""
입찰메이트 RAG — Retrieval 평가 모듈
======================================
담당 : Retrieval 파트 (한의정)
용도 : 평가 전용 (서빙과 완전 분리)

변경사항 (v6 통합)
- retrieval_eval_hybrid_ck.py 흡수
  - BidMateEvaluator에 use_hybrid 파라미터 추가
  - retrieval_eval_hybrid_ck.py 별도 파일 불필요

[사용법]
    from retrieval_eval import BidMateEvaluator

    # 일반
    ev = BidMateEvaluator(scenario='A-1')

    # hybrid (Child-to-Parent)
    ev = BidMateEvaluator(scenario='A-1', use_hybrid=True)

    # 청크/컬렉션 지정
    ev = BidMateEvaluator(scenario='A-1',
                          chunks_path='/path/to/chunks.json',
                          collection_name='bidmate_kh_v3_A-1',
                          use_hybrid=True)

    result = ev.retrieve(query="질문", meta_filter={...})
    ev.release()
"""

import os
import gc
import math
import logging
from pathlib import Path

import torch
import chromadb
from sentence_transformers import CrossEncoder

from retrieval_interface_F import (
    BidMateRetriever, EMBED_CONFIG,
    _tokenize_ko, _normalize_chunk, _load_chunks,
    _build_embed_model, _build_or_load_chroma, _build_or_load_bm25,
    _parse_meta_filter_with, _RERANK_MODEL,
)

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
    _DEFAULT_CHUNKS = Path('/mnt/gukrul/dataset/chunks/chunks_all.json')
    _CHROMA_PATH    = Path('/mnt/gukrul/dataset/hej/chroma_db')
    _BM25_DIR       = Path('/mnt/gukrul/dataset/bm25')
else:
    _DEFAULT_CHUNKS = Path.home() / 'Desktop' / 'AI엔지니어_8기' / '프로젝트' / '중급' / 'data' / 'chunks' / 'kh_fixed_1200_200_v2.json'
    _CHROMA_PATH    = Path.home() / 'Desktop' / 'AI엔지니어_8기' / '프로젝트' / '중급' / 'chroma_db'
    _BM25_DIR       = Path.home() / 'Desktop' / 'AI엔지니어_8기' / '프로젝트' / '중급' / 'data'

if torch.cuda.is_available():           _DEVICE = 'cuda'
elif torch.backends.mps.is_available(): _DEVICE = 'mps'
else:                                   _DEVICE = 'cpu'

_chroma_client = chromadb.PersistentClient(path=str(_CHROMA_PATH))


# ======================================================================
# BidMateEvaluator
# ======================================================================
class BidMateEvaluator:
    """평가 전용 Retriever. 시나리오별 독립 인스턴스 생성."""

    def __init__(self, scenario: str = 'A-1',
                 chunks_path: str = None,
                 collection_name: str = None,
                 use_hybrid: bool = False):
        """
        Parameters
        ----------
        scenario        : 'A-1' / 'A-2' / 'B'
        chunks_path     : 청크 JSON 경로. 
        collection_name : ChromaDB 컬렉션명. None이면 EMBED_CONFIG 기본값 사용
        use_hybrid      : True면 Child-to-Parent Retrieval 적용
        """
        self.scenario   = scenario
        self.use_hybrid = use_hybrid
        chunks_path     = Path(chunks_path) if chunks_path else _DEFAULT_CHUNKS
        # 수정 — scenario가 'KURE_GEMMA' 형태로 오면 embed_key만 추출
        embed_key = scenario.split('_')[0] if '_' in scenario else scenario
        col_name  = collection_name or EMBED_CONFIG[embed_key]['collection']

        # BM25 인덱스 파일명을 collection_name 기준으로 격리
        bm25_path = _BM25_DIR / f'bm25_index_{col_name}.pkl'

        _logger.info(
            f'[평가] BidMateEvaluator 초기화 | scenario={scenario} | '
            f'collection={col_name} | use_hybrid={use_hybrid} | bm25={bm25_path.name}'
        )

        chunks            = _load_chunks(chunks_path)
        self._embed_model = _build_embed_model(scenario)
        col               = _build_or_load_chroma(_chroma_client, col_name, chunks, self._embed_model)
        bm25_idx, bm25_cids, bm25_texts = _build_or_load_bm25(chunks, bm25_path)
        self._reranker    = CrossEncoder(_RERANK_MODEL, device=_DEVICE)

        self._retriever = BidMateRetriever(
            collection     = col,
            bm25_index     = bm25_idx,
            bm25_chunk_ids = bm25_cids,
            bm25_texts     = bm25_texts,
            embed_model    = self._embed_model,
            all_chunks     = chunks,
            reranker       = self._reranker,
            use_hybrid     = use_hybrid,
        )
        _logger.info(f'[평가] ✅ 초기화 완료 | scenario={scenario}')

    def retrieve(self, query: str, meta_filter: dict = None) -> dict:
        """단일 쿼리 검색."""
        if meta_filter is None:
            meta_filter = _parse_meta_filter_with(query, self._retriever._all_agencies)
        result = self._retriever.retrieve(query, meta_filter=meta_filter)
        result['embed_scenario'] = self.scenario
        return result

    def release(self):
        """GPU 메모리 해제. 시나리오 전환 전 호출."""
        del self._embed_model
        del self._reranker
        del self._retriever
        gc.collect()
        torch.cuda.empty_cache()
        free_gb = torch.cuda.mem_get_info()[0] / 1024**3 if torch.cuda.is_available() else 0
        _logger.info(f'[평가] GPU 해제 완료 | 여유: {free_gb:.1f}GB')


# ======================================================================
# 평가 지표 함수
# ======================================================================
def normalize_fn(fn): return Path(fn).stem.strip() if fn else ''

def _match_flags(gt, ret):
    rem = list(gt); flags = []
    for r in ret:
        matched = False
        for g in rem:
            if g in r or r in g:
                flags.append(1); rem.remove(g); matched = True; break
        if not matched: flags.append(0)
    return flags

def calc_hit(gt, ret):
    if not gt: return None
    return 1 in _match_flags(gt, ret)

def calc_mrr(gt, ret):
    if not gt: return None
    for rank, f in enumerate(_match_flags(gt, ret), 1):
        if f: return 1. / rank
    return 0.

def calc_ndcg(gt, ret):
    if not gt: return None
    flags = _match_flags(gt, ret)
    dcg   = sum(f / math.log2(r+1) for r, f in enumerate(flags, 1))
    idcg  = sum(1. / math.log2(i+2) for i in range(min(len(gt), len(flags))))
    return dcg / idcg if idcg > 0 else 0.
