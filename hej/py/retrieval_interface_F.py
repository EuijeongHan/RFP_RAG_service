"""
입찰메이트 RAG — Retrieval 서빙 모듈
======================================
담당 : Retrieval 파트 (한의정)
용도 : Generation 파트 서빙 전용

변경사항 (v6 통합)
- retrieval_interface_hybrid_ck.py 흡수
  - Child-to-Parent Retrieval: USE_HYBRID=true 시 parent_text 우선 반환
  - parent_text 위치 버그 수정: 최상위 또는 metadata 안 모두 읽음 (chunks_all 대응)
- retrieval_interface_hybrid_ck.py 별도 파일 불필요

[서빙 사용법]
    import os
    os.environ['BIDMATE_ENV']    = 'gcp'
    os.environ['EMBED_SCENARIO'] = 'A-1'
    os.environ['USE_HYBRID']     = 'true'   # Child-to-Parent 사용 시

    from retrieval_interface import get_context, build_prompt

    result = get_context(query="한국가스공사 ERP 예산은?")
    prompt = build_prompt(query, result['context'], scenario='A-1')
"""

import os
import re
import json
import pickle
import difflib
import logging
from pathlib import Path

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

# ======================================================================
# 환경
# ======================================================================
try:
    import google.colab
    _ENV = 'colab'
except ImportError:
    _ENV = os.environ.get('BIDMATE_ENV', 'gcp')

# EMBED_SCENARIO = os.environ.get('EMBED_SCENARIO', 'A-1')
# 수정
EMBED_SCENARIO = os.environ.get('EMBED_SCENARIO', 'KURE_GEMMA').split('_')[0]
USE_HYBRID     = os.environ.get('USE_HYBRID', 'false').lower() == 'true'

if _ENV == 'gcp':
    _PROJECT_ROOT = Path('/mnt/gukrul/hej')
    _CHUNKS_PATH  = _PROJECT_ROOT / 'chunks' / 'kh_fixed_1200_200_v2.json'
    _CHROMA_PATH  = Path('/mnt/gukrul/dataset/hej/chroma_db')
    _BM25_DIR     = _PROJECT_ROOT / 'bm25'
else:
    _PROJECT_ROOT = Path.home() / 'Desktop' / 'AI엔지니어_8기' / '프로젝트' / '중급'
    _CHUNKS_PATH  = _PROJECT_ROOT / 'data' / 'chunks' / 'kh_fixed_1200_200_v2.json'
    _CHROMA_PATH  = _PROJECT_ROOT / 'chroma_db'
    _BM25_DIR     = _PROJECT_ROOT / 'data'

_BM25_PATH = _BM25_DIR / f'bm25_index_{EMBED_SCENARIO}.pkl'

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
_logger = logging.getLogger(__name__)
_logger.info(f'[서빙] ENV={_ENV} | EMBED_SCENARIO={EMBED_SCENARIO} | USE_HYBRID={USE_HYBRID}')

# ======================================================================
# 하이퍼파라미터
# ======================================================================
_DENSE_K      = 15
_SPARSE_K     = 15
_RRF_K        = 60
_TOP_K        = 5
_MMR_LAMBDA   = 0.6
_MMR_TOP_N    = 20
_RERANK_TOP_N = 15
_RERANK_MODEL = 'BAAI/bge-reranker-v2-m3'

# 현재
# EMBED_CONFIG = {
#     'A-1': {'model': 'nlpai-lab/KURE-v1',     'collection': 'bidmate_retrieval_v1'},
#     'A-2': {'model': 'nlpai-lab/KoE5',         'collection': 'bidmate_retrieval_koe5_v1'},
#     'B'  : {'model': 'text-embedding-3-small', 'collection': 'bidmate_retrieval_openai_v1'},
# }
# 수정
EMBED_CONFIG = {
    'KURE' : {'model': 'nlpai-lab/KURE-v1',     'collection': 'bidmate_kure'},
    'KOE5' : {'model': 'nlpai-lab/KoE5',         'collection': 'bidmate_koe5'},
    'SMALL': {'model': 'text-embedding-3-small', 'collection': 'bidmate_small'},
}

if torch.cuda.is_available():           _DEVICE = 'cuda'
elif torch.backends.mps.is_available(): _DEVICE = 'mps'
else:                                   _DEVICE = 'cpu'

# ======================================================================
# 공통 유틸
# ======================================================================
_kiwi = Kiwi()

def _tokenize_ko(text: str) -> list:
    if not isinstance(text, str) or not text.strip(): return []
    keep = {'NNG','NNP','NNB','NR','VV','VA','SL','SN'}
    return [t.form for t in _kiwi.tokenize(text, normalize_coda=True)
            if t.tag in keep and len(t.form) > 1]

def _normalize_chunk(c: dict) -> dict:
    meta = dict(c.get('metadata', {}))

    # agency 정규화: kh_v3/chunks_all은 agency 있음, kh_fixed는 organization_cleaned/raw
    if 'agency' not in meta:
        meta['agency'] = meta.get('organization_cleaned', meta.get('organization_raw', '미지정'))

    # source_file 정규화: kh_v3/chunks_all은 source_file 있음, kh_fixed는 original_name
    if 'source_file' not in meta:
        meta['source_file'] = meta.get('original_name', '')

    # year 문자열 통일
    if 'year' in meta:
        meta['year'] = str(meta['year'])

    # has_table / has_number 기본값 (kh_fixed_v1/v2에 없음)
    meta.setdefault('has_table',  False)
    meta.setdefault('has_number', False)

    # 리스트 타입 → 문자열 변환 (ChromaDB는 리스트 메타데이터 불가)
    for k, v in list(meta.items()):
        if isinstance(v, list):
            meta[k] = ','.join(str(x) for x in v) if v else ''

    # chunk_id: kh_fixed/chunks_all은 chunk_id, kh_v3는 child_id
    chunk_id = c.get('chunk_id') or c.get('child_id', '')

    # parent_text 위치 버그 수정:
    #   kh_v3      → 최상위 c['parent_text']
    #   chunks_all → metadata 안 c['metadata']['parent_text']
    #   kh_fixed   → 없음 (빈 문자열)
    parent_text = c.get('parent_text', '') or meta.get('parent_text', '')

    # parent_text는 ChromaDB 메타에 넣지 않음 (대용량, 오염 방지)
    meta.pop('parent_text', None)

    return {
        'chunk_id'   : chunk_id,
        'text'       : c.get('chunk_text', c.get('text', '')),
        'parent_text': parent_text,
        'metadata'   : meta,
    }

def _load_chunks(chunks_path: Path) -> list:
    if not chunks_path.exists(): raise FileNotFoundError(f'청크 JSON 없음: {chunks_path}')
    with open(chunks_path, 'r', encoding='utf-8') as f: data = json.load(f)
    raw = data if isinstance(data, list) else [data]
    seen, chunks = set(), []
    for c in raw:
        nc = _normalize_chunk(c)
        if nc['chunk_id'] not in seen:
            seen.add(nc['chunk_id']); chunks.append(nc)
    return chunks

def _build_embed_model(scenario: str):
    cfg = EMBED_CONFIG[scenario]
    if scenario in ('B', 'SMALL'):
        from openai import OpenAI as _OAI
        client = _OAI()
        class _OAIEmb:
            def encode(self, texts, normalize_embeddings=True):
                if isinstance(texts, str): texts = [texts]
                res  = client.embeddings.create(model=cfg['model'], input=texts)
                vecs = np.array([d.embedding for d in res.data], dtype=np.float32)
                if normalize_embeddings:
                    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
                return vecs
        return _OAIEmb()
    return SentenceTransformer(cfg['model'], device=_DEVICE)

def _build_or_load_chroma(client, collection_name, chunks, embed_model):
    col = client.get_or_create_collection(name=collection_name, metadata={'hnsw:space': 'cosine'})
    if col.count() == 0:
        _logger.info(f'ChromaDB 인덱싱 : {len(chunks):,}개 → {collection_name}')
        for i in range(0, len(chunks), 64):
            b = chunks[i:i+64]
            col.upsert(
                ids        = [c['chunk_id'] for c in b],
                documents  = [c['text']     for c in b],
                metadatas  = [c['metadata'] for c in b],
                embeddings = embed_model.encode([c['text'] for c in b], normalize_embeddings=True).tolist(),
            )
            if (i // 64) % 10 == 0: _logger.info(f'   {min(i+64, len(chunks)):,}/{len(chunks):,}')
        _logger.info(f'ChromaDB 인덱싱 완료 : {col.count():,}개')
    else:
        _logger.info(f'ChromaDB 재사용 : {collection_name} ({col.count():,}개)')
    return col

def _build_or_load_bm25(chunks, bm25_path):
    if bm25_path.exists():
        _logger.info(f'BM25 로드 : {bm25_path.name}')
        with open(bm25_path, 'rb') as f: d = pickle.load(f)
        return d['index'], d['chunk_ids'], d['texts']
    _logger.info('BM25 구축 중...')
    texts = [c['text']     for c in chunks]
    cids  = [c['chunk_id'] for c in chunks]
    idx   = BM25Okapi([_tokenize_ko(t) for t in texts])
    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bm25_path, 'wb') as f:
        pickle.dump({'index': idx, 'chunk_ids': cids, 'texts': texts}, f)
    _logger.info(f'BM25 저장 완료 : {bm25_path.name}')
    return idx, cids, texts

def _extract_year(query):
    for pat in [r'(20\d{2})년?', r"'(\d{2})년?", r'(\d{2})년도']:
        m = re.search(pat, query)
        if m: y = m.group(1); return y if len(y) == 4 else f'20{y}'
    return None

def _parse_meta_filter_with(query, all_agencies):
    f = {}
    best, best_s = None, 0.
    for ag in all_agencies:
        s = difflib.SequenceMatcher(None, query, ag).ratio()
        for kw in ag.split():
            if len(kw) >= 2 and kw in query: s = max(s, 0.75)
        if s > best_s: best_s = s; best = ag
    if best and best_s >= 0.6: f['agency'] = best
    yr = _extract_year(query)
    if yr: f['year'] = yr
    return f


# ======================================================================
# BidMateRetriever
# ======================================================================
class BidMateRetriever:
    """
    범용 Retriever. 서빙/평가 공통 사용.
    use_hybrid=True 시 Child-to-Parent Retrieval 적용:
      - 검색(Dense/Sparse/Reranker)은 child text 기준
      - 컨텍스트 반환은 parent_text 우선 (없으면 child text fallback)
    """

    def __init__(self, collection, bm25_index, bm25_chunk_ids,
                 bm25_texts, embed_model, all_chunks, reranker=None,
                 use_hybrid: bool = False):
        self.collection       = collection
        self.bm25_index       = bm25_index
        self.bm25_chunk_ids   = bm25_chunk_ids
        self.embed_model      = embed_model
        self.chunk_meta_map   = {c['chunk_id']: c['metadata']    for c in all_chunks}
        self.chunk_text_map   = {c['chunk_id']: c['text']        for c in all_chunks}
        self.chunk_parent_map = {c['chunk_id']: c['parent_text'] for c in all_chunks}
        self._emb_cache: dict = {}
        self.reranker         = reranker
        self.use_hybrid       = use_hybrid
        self._all_agencies    = list({
            c['metadata'].get('agency', '') for c in all_chunks
            if c['metadata'].get('agency', '')
        })

    def _build_where(self, mf):
        if not mf: return None
        conds = []
        for k, v in mf.items():
            if not v: continue
            conds.append({k: {'$in': [str(x) for x in v]}} if isinstance(v, list) else {k: {'$eq': str(v)}})
        if not conds: return None
        return conds[0] if len(conds) == 1 else {'$and': conds}

    def _filter_bm25(self, mf):
        if not mf: return None
        allowed = {
            i for i, cid in enumerate(self.bm25_chunk_ids)
            if not ('agency' in mf and self.chunk_meta_map.get(cid, {}).get('agency') != mf['agency'])
            and not ('year'  in mf and self.chunk_meta_map.get(cid, {}).get('year')   != mf['year'])
        }
        return list(allowed) if allowed else None

    def _decompose(self, query):
        clean = query.replace("'", "").replace('"', "")
        found = []
        for ag in sorted(self._all_agencies, key=len, reverse=True):
            core = re.sub(r'^\(주\)|\(사\)|주식회사', '', ag).strip()
            if len(core) >= 2 and core in clean and not any(core in fa[1] for fa in found):
                found.append((ag, core))
        if len(found) < 2: return [query]
        found = found[:3]; found.sort(key=lambda x: clean.find(x[1]))
        segs = []
        for i, (af, ac) in enumerate(found):
            s = clean.find(ac) + len(ac)
            e = clean.find(found[i+1][1]) if i < len(found)-1 else len(clean)
            nouns = [t.form for t in _kiwi.tokenize(clean[s:e], normalize_coda=True)
                     if (t.tag.startswith('N') or t.tag in ['SL', 'SN']) and len(t.form) >= 2]
            segs.append({'agency': af, 'local_nouns': nouns})
        gn = segs[-1]['local_nouns']
        return [f"{seg['agency']} {' '.join(seg['local_nouns'] or gn)}".strip() for seg in segs]

    def _dense(self, query, where):
        qe  = self.embed_model.encode([query], normalize_embeddings=True).tolist()
        kw  = dict(query_embeddings=qe, n_results=_DENSE_K,
                   include=['documents', 'metadatas', 'distances', 'embeddings'])
        if where: kw['where'] = where
        res = self.collection.query(**kw)
        el  = res.get('embeddings')
        if el and len(el) > 0 and len(el[0]) > 0:
            for cid, emb in zip(res['ids'][0], el[0]):
                if cid not in self._emb_cache: self._emb_cache[cid] = emb
        return res['ids'][0]

    def _sparse(self, query, allowed):
        toks = _tokenize_ko(query)
        if not toks: return []
        scores = self.bm25_index.get_scores(toks)
        if allowed is not None:
            mask = np.zeros(len(scores)); mask[allowed] = 1.0; scores = scores * mask
        top = np.argsort(scores)[::-1][:_SPARSE_K]
        return [self.bm25_chunk_ids[i] for i in top if scores[i] > 0]

    def _multi(self, queries, where, allowed, orig=''):
        di, si, sd, ss = [], [], set(), set()
        for q in ([orig] if orig else []) + list(queries):
            for cid in self._dense(q, where):
                if cid not in sd: di.append(cid); sd.add(cid)
            for cid in self._sparse(q, allowed):
                if cid not in ss: si.append(cid); ss.add(cid)
        return di, si

    def _rrf(self, di, si):
        rrf = {}
        for r, cid in enumerate(di, 1): rrf[cid] = rrf.get(cid, 0.) + 1. / (_RRF_K + r)
        for r, cid in enumerate(si, 1): rrf[cid] = rrf.get(cid, 0.) + 1. / (_RRF_K + r)
        return sorted(rrf.items(), key=lambda x: x[1], reverse=True)

    def _boost(self, ranked):
        out = []
        for cid, score in ranked:
            meta = self.chunk_meta_map.get(cid, {}); mul = 1.0
            if meta.get('has_table',  False): mul += 0.10
            if meta.get('has_number', False): mul += 0.05
            out.append((cid, score * mul))
        return sorted(out, key=lambda x: x[1], reverse=True)

    def _mmr(self, boosted, query):
        cands = boosted[:_MMR_TOP_N]
        if len(cands) <= 1: return cands
        valid  = [(cid, s) for cid, s in cands if cid in self._emb_cache]
        if not valid: return cands
        cids   = [c for c, _ in valid]; scores = [s for _, s in valid]
        vecs   = np.array([self._emb_cache[c] for c in cids], dtype=np.float32)
        qv     = np.array(self.embed_model.encode([query], normalize_embeddings=True)[0], dtype=np.float32)
        vn     = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        qn     = qv / (np.linalg.norm(qv) + 1e-9); rs = vn @ qn
        def mm(a): mn, mx = a.min(), a.max(); return (a - mn) / (mx - mn + 1e-9)
        rel    = (mm(rs) + mm(np.array(scores))) / 2.
        sel, rem = [], list(range(len(cids)))
        while rem:
            if not sel: best = max(rem, key=lambda i: rel[i])
            else:
                sv = vn[sel]; best = -1; bm = -float('inf')
                for i in rem:
                    ms = _MMR_LAMBDA * rel[i] - (1 - _MMR_LAMBDA) * float(np.max(vn[i] @ sv.T))
                    if ms > bm: bm = ms; best = i
            sel.append(best); rem.remove(best)
        return [(cids[i], scores[i]) for i in sel]

    def _rerank(self, boosted, query):
        if self.reranker is None: return boosted
        cands = boosted[:_RERANK_TOP_N]
        if not cands: return boosted
        # use_hybrid=True면 parent_text 기준 rerank (없으면 child text fallback)
        if self.use_hybrid:
            pairs = [[query, self.chunk_parent_map.get(cid, '') or self.chunk_text_map.get(cid, '')]
                     for cid, _ in cands]
        else:
            pairs = [[query, self.chunk_text_map.get(cid, '')] for cid, _ in cands]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip([c for c, _ in cands], scores), key=lambda x: x[1], reverse=True)
        rids   = {c for c, _ in ranked}
        return [(c, float(s)) for c, s in ranked] + [(c, s) for c, s in boosted[_RERANK_TOP_N:] if c not in rids]

    def _context(self, top):
        lines = []
        for i, (cid, score) in enumerate(top[:_TOP_K], 1):
            meta = self.chunk_meta_map.get(cid, {})
            # use_hybrid=True면 parent_text 우선, 없으면 child text fallback
            if self.use_hybrid:
                text = self.chunk_parent_map.get(cid, '') or self.chunk_text_map.get(cid, '')
            else:
                text = self.chunk_text_map.get(cid, '')
            lines.append(
                f'[{i}] {text}\n'
                f'(출처: {meta.get("agency","미상")} {meta.get("year","")} | score: {score:.4f})'
            )
        return '\n\n'.join(lines)

    def retrieve(self, query, meta_filter=None):
        if meta_filter is None: meta_filter = _parse_meta_filter_with(query, self._all_agencies)
        where   = self._build_where(meta_filter)
        allowed = self._filter_bm25(meta_filter)
        subs    = self._decompose(query)
        if len(subs) > 1: di, si = self._multi(subs, where, allowed, orig=query)
        else:             di = self._dense(query, where); si = self._sparse(query, allowed)
        boosted = self._boost(self._rrf(di, si))
        boosted = self._mmr(boosted, query)
        boosted = self._rerank(boosted, query)
        top5    = boosted[:_TOP_K]
        return {
            'context'    : self._context(top5),
            'top_chunks' : [{'rank': i+1, 'chunk_id': cid, 'boosted_score': s,
                              'text': self.chunk_text_map.get(cid, ''),
                              'parent_text': self.chunk_parent_map.get(cid, ''),
                              'metadata': self.chunk_meta_map.get(cid, {})}
                             for i, (cid, s) in enumerate(top5)],
            'meta_filter': meta_filter,
            'sub_queries': subs,
        }


# ======================================================================
# 서빙용 싱글턴 — Lazy Initialization
# ======================================================================
_retriever = None

def _get_serving_retriever():
    global _retriever
    if _retriever is None:
        _logger.info('서빙 Retriever 초기화 중 (Lazy)...')
        all_chunks  = _load_chunks(_CHUNKS_PATH)
        embed_model = _build_embed_model(EMBED_SCENARIO)
        client      = chromadb.PersistentClient(path=str(_CHROMA_PATH))
        col         = _build_or_load_chroma(client, EMBED_CONFIG[EMBED_SCENARIO]['collection'], all_chunks, embed_model)
        bm25_idx, bm25_cids, bm25_texts = _build_or_load_bm25(all_chunks, _BM25_PATH)
        reranker    = CrossEncoder(_RERANK_MODEL, device=_DEVICE)
        _retriever  = BidMateRetriever(
            collection     = col,
            bm25_index     = bm25_idx,
            bm25_chunk_ids = bm25_cids,
            bm25_texts     = bm25_texts,
            embed_model    = embed_model,
            all_chunks     = all_chunks,
            reranker       = reranker,
            use_hybrid     = USE_HYBRID,
        )
        _logger.info('✅ 서빙 Retriever 초기화 완료')
    return _retriever


# ======================================================================
# 서빙 Public API
# ======================================================================
def get_context(query: str, history: list = None, meta_filter: dict = None) -> dict:
    """Generation 파트 메인 인터페이스."""
    effective = query
    if history:
        prev = [h['content'] for h in history if h.get('role') == 'user']
        if prev: effective = f'{prev[-1]} {query}'
    result = _get_serving_retriever().retrieve(effective, meta_filter=meta_filter)
    return {
        'context'    : result['context'],
        'sources'    : [{'rank': c['rank'], 'agency': c['metadata'].get('agency', ''),
                         'year': c['metadata'].get('year', ''),
                         'project': c['metadata'].get('project_name', ''),
                         'score': c['boosted_score']} for c in result['top_chunks']],
        'filter'     : result['meta_filter'],
        'sub_queries': result['sub_queries'],
    }


# ======================================================================
# 프롬프트 빌더
# ======================================================================
# ======================================================================
# Prompt Compression 설정
# ======================================================================
# 청크 텍스트를 LLM에 전달하기 전 최대 길이 제한 (토큰 절약)
# 1개 청크당 최대 문자 수. None이면 제한 없음.
# 입찰 공고 특성상 핵심 정보는 앞부분에 집중 → 앞부분 유지 방식 적용
_PROMPT_COMPRESS_CHARS = int(os.environ.get('PROMPT_COMPRESS_CHARS', '800'))


def _compress_context(context: str, max_chars_per_chunk: int = None) -> str:
    """
    Prompt Compression: 각 청크 텍스트를 max_chars_per_chunk 이하로 자름.
    청크 구분자는 빈 줄(\n\n)로 가정.

    Parameters
    ----------
    context           : retriever가 반환한 전체 컨텍스트 문자열
    max_chars_per_chunk : 청크당 최대 문자 수. None이면 원본 반환.
    """
    if not max_chars_per_chunk or not context or not isinstance(context, str):
        return context
    chunks = context.split('\n\n')
    compressed = [c[:max_chars_per_chunk] for c in chunks]
    return '\n\n'.join(compressed)


# ======================================================================
# 프롬프트 템플릿 (LLM별)
# ======================================================================
# 기존 A-1/A-2/B → GEMMA/QWEN/PHI/OPENAI 로 전환
# 하위호환을 위해 A-1/A-2/B 키도 유지

_SYSTEM_PROMPT = (
    "당신은 대한민국 공공 입찰 공고 전문 AI 어시스턴트입니다.\n"
    "반드시 제공된 [참고 문서]만을 근거로 답변하세요.\n"
    "- 문서에 없는 내용은 '해당 정보를 찾을 수 없습니다'라고 답하세요.\n"
    "- 예산/금액은 원 단위까지 정확하게 표기하세요.\n"
    "- 답변 말미에 출처 문서명을 반드시 명시하세요.\n"
    "- 추측하거나 일반 지식을 사용하지 마세요."
)

_PROMPT_TEMPLATES = {
    # ── Gemma 4 계열 (gemma-4-E4B-it) ─────────────────────────────────
    # <start_of_turn>user ... <end_of_turn><start_of_turn>model 포맷
    'GEMMA': (
        "<start_of_turn>user\n"
        + _SYSTEM_PROMPT + "\n\n"
        "[참고 문서]\n{context}\n\n[질문]\n{query}\n"
        "<end_of_turn>\n<start_of_turn>model\n"
    ),
    # ── Qwen3.5 계열 (Qwen/Qwen3.5-4B) ────────────────────────────────
    # <|im_start|>system ... <|im_end|> 포맷
    'QWEN': (
        "<|im_start|>system\n"
        + _SYSTEM_PROMPT + "\n<|im_end|>\n"
        "<|im_start|>user\n"
        "[참고 문서]\n{context}\n\n[질문]\n{query}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    # ── Phi-4-mini-instruct (microsoft/Phi-4-mini-instruct) ────────────
    # <|system|>...<|end|><|user|>...<|end|><|assistant|> 포맷
    'PHI': (
        "<|system|>\n"
        + _SYSTEM_PROMPT + "\n<|end|>\n"
        "<|user|>\n"
        "[참고 문서]\n{context}\n\n[질문]\n{query}\n<|end|>\n"
        "<|assistant|>\n"
    ),
    # ── OpenAI API 계열 (messages 배열 방식) ───────────────────────────
    'OPENAI': {
        'system': _SYSTEM_PROMPT,
        'user'  : "[참고 문서]\n{context}\n\n[질문]\n{query}",
    },
    # ── 하위호환 키 (기존 A-1/A-2/B 방식) ─────────────────────────────
    'A-1': None,  # GEMMA로 위임
    'A-2': None,  # QWEN으로 위임
    'B'  : None,  # OPENAI로 위임
}

# 하위호환 매핑
_LEGACY_MAP = {'A-1': 'GEMMA', 'A-2': 'QWEN', 'B': 'OPENAI'}


def build_prompt(query: str, context: str, scenario: str = 'GEMMA',
                 compress: bool = True) -> dict:
    """
    쿼리 + 컨텍스트 → LLM 입력 프롬프트 딕셔너리 생성.

    Parameters
    ----------
    query    : 사용자 질문
    context  : retriever가 반환한 컨텍스트 문자열
    scenario : 'GEMMA' | 'QWEN' | 'PHI' | 'OPENAI'
               또는 하위호환 'A-1' | 'A-2' | 'B'
               또는 임베딩_LLM 조합 'KURE_GEMMA' 등 (LLM 키만 추출)
    compress : True면 Prompt Compression 적용 (기본값 True)

    Returns
    -------
    {
      'scenario': str,
      'prompt'  : str | None,   # 로컬 모델용 완성 프롬프트
      'system'  : str | None,   # API 모델용 시스템 메시지
      'user'    : str | None,   # API 모델용 유저 메시지
    }
    """
    # 임베딩_LLM 조합에서 LLM 키 추출 (예: 'KURE_GEMMA' → 'GEMMA')
    if '_' in scenario:
        scenario = scenario.split('_')[1]

    # 하위호환 매핑
    scenario = _LEGACY_MAP.get(scenario, scenario)

    if scenario not in _PROMPT_TEMPLATES or _PROMPT_TEMPLATES[scenario] is None:
        raise ValueError(
            f"정의되지 않은 scenario: '{scenario}'. "
            f"가능한 값: GEMMA / QWEN / PHI / OPENAI"
        )

    # Prompt Compression 적용
    if compress:
        context = _compress_context(context, _PROMPT_COMPRESS_CHARS)

    tmpl = _PROMPT_TEMPLATES[scenario]
    if isinstance(tmpl, str):
        return {
            'scenario': scenario,
            'prompt'  : tmpl.format(context=context, query=query),
            'system'  : None,
            'user'    : None,
        }
    return {
        'scenario': scenario,
        'prompt'  : None,
        'system'  : tmpl['system'],
        'user'    : tmpl['user'].format(context=context, query=query),
    }


# ======================================================================
# HyDE (Hypothetical Document Embeddings)
# ======================================================================
def generate_hyde_query(query: str, scenario: str = 'OPENAI') -> str:
    """
    HyDE: 질문에 대해 가상의 답변을 생성하고, 그 답변을 검색 쿼리로 사용.

    실제 답변과 의미적으로 유사한 문서를 검색하여 의미적 간극을 메움.
    API 모델(OPENAI)에서만 활성화. 로컬 모델은 속도 문제로 생략.

    Parameters
    ----------
    query    : 원본 사용자 질문
    scenario : 'OPENAI' (기본값) — API 모델만 사용 권장

    Returns
    -------
    str : 가상 답변 문자열 (검색 쿼리로 사용)
    """
    llm_key = scenario.split('_')[1] if '_' in scenario else scenario
    if llm_key != 'OPENAI':
        # 로컬 모델은 HyDE 생략 → 원본 쿼리 그대로 반환
        return query

    try:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model  = os.environ.get('OPENAI_LLM_MODEL', 'gpt-5-mini'),
            messages = [
                {
                    'role'   : 'system',
                    'content': (
                        "당신은 대한민국 공공 입찰 공고 전문가입니다.\n"
                        "사용자의 질문에 대해 실제 입찰 공고 문서에 나올 법한 "
                        "2~3문장 분량의 가상 답변을 작성하세요. "
                        "없는 정보는 일반적인 입찰 공고 형식으로 작성하세요."
                    ),
                },
                {'role': 'user', 'content': query},
            ],
            max_completion_tokens = 200,
            temperature           = 1,
        )
        hyde_text = response.choices[0].message.content.strip()
        _logger.info(f'[HyDE] 가상 답변 생성 완료 ({len(hyde_text)}자)')
        return f"{query} {hyde_text}"
    except Exception as e:
        _logger.warning(f'[HyDE] 가상 답변 생성 실패 → 원본 쿼리 사용: {e}')
        return query
