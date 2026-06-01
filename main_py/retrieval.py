import re
import json
import math
import pickle
import difflib
import logging
import numpy as np
from pathlib import Path
from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
import torch

from config import (
    CHUNKS_PATH, CHROMA_PATH, BM25_PATH, EMBED_MODEL_ID, RERANKER_ID,
    COLLECTION_NAME, DENSE_K, SPARSE_K, RRF_K, TOP_K,
    MMR_LAMBDA, MMR_TOP_N, RERANK_TOP_N, BATCH_SIZE,
)

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

kiwi = Kiwi()
ALL_AGENCIES: list = []


def _normalize_chunk(c: dict) -> dict:
    meta = dict(c.get("metadata", {}))
    if "agency" not in meta:
        meta["agency"] = meta.get("organization_cleaned",
                         meta.get("organization_raw", "미지정"))
    if "source_file" not in meta:
        meta["source_file"] = meta.get("original_name", "")
    if "year" in meta:
        meta["year"] = str(meta["year"])
    meta.setdefault("has_table",  False)
    meta.setdefault("has_number", False)
    chunk_text = c.get("chunk_text", c.get("text", ""))
    summary    = c.get("text", "")  # 예산/기간 요약
    if summary and summary != chunk_text and "사업예산" in summary:
        full_text = summary + "\n" + chunk_text
    else:
        full_text = chunk_text
        full_text = chunk_text
    return {
        "chunk_id": c.get("chunk_id", ""),
        "text"    : full_text,
        "metadata": meta,
    }


def load_chunks() -> list:
    path = Path(CHUNKS_PATH)
    if not path.exists():
        raise FileNotFoundError(f"청크 파일 없음: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw = data if isinstance(data, list) else [data]
    return [_normalize_chunk(c) for c in raw]


def tokenize_ko(text: str) -> list:
    if not isinstance(text, str) or not text.strip():
        return []
    keep_tags = {"NNG", "NNP", "NNB", "NR", "VV", "VA", "SL", "SN"}
    return [
        t.form for t in kiwi.tokenize(text, normalize_coda=True)
        if t.tag in keep_tags and len(t.form) > 1
    ]


def extract_year(query: str):
    for pat in [r"(20\d{2})년?", r"'(\d{2})년?", r"(\d{2})년도"]:
        m = re.search(pat, query)
        if m:
            y = m.group(1)
            return y if len(y) == 4 else f"20{y}"
    return None


def fuzzy_match_agency(query: str, threshold: float = 0.6):
    best_match, best_score = None, 0.0
    for agency in ALL_AGENCIES:
        score = difflib.SequenceMatcher(None, query, agency).ratio()
        for keyword in agency.split():
            if len(keyword) >= 2 and keyword in query:
                score = max(score, 0.75)
        if score > best_score:
            best_score = score
            best_match = agency
    return best_match if best_score >= threshold else None



def fuzzy_normalize_query(query: str, threshold: int = 80) -> str:
    """
    rapidfuzz로 쿼리 내 오타 토큰을 ALL_AGENCIES 및 주요 키워드와 매칭해 보정.
    threshold: 유사도 점수 0~100, 높을수록 엄격
    """
    try:
        from rapidfuzz import process, fuzz
    except ImportError:
        return query

    if not ALL_AGENCIES:
        return query

    # 주요 프로젝트 키워드 사전
    keyword_pool = list(ALL_AGENCIES) + [
        "전자조달", "그룹웨어", "시스템", "구축", "용역", "예산", "기간",
        "스마트", "프로젝트", "관개", "통합정보시스템", "ERP", "KOICA",
        "학사포털", "출입통제", "보안", "인프라", "클라우드",
    ]

    tokens = query.split()
    result_tokens = []
    for token in tokens:
        # 짧은 토큰, 영문, 숫자는 스킵
        if len(token) <= 2 or token.isascii():
            result_tokens.append(token)
            continue
        match = process.extractOne(token, keyword_pool, scorer=fuzz.ratio)
        if match and match[1] >= threshold and match[0] != token:
            result_tokens.append(match[0])
        else:
            result_tokens.append(token)
    return " ".join(result_tokens)

def parse_metadata_filter(query: str) -> dict:
    filters = {}
    # 복수 기관 감지 - 2개 이상이면 agency 필터 제거 (B타입 비교 쿼리)
    agency_pat = r"한국[가-힣]{1,6}공사|[가-힣]{2,8}공단|[가-힣]{2,8}은행|[가-힣]{2,8}공사|[가-힣]{2,8}연구원|[가-힣]{2,8}대학교|[가-힣]{2,8}의료원"
    found_agencies = list(dict.fromkeys(re.findall(agency_pat, query)))
    if len(found_agencies) >= 2:
        pass  # 복수 기관 비교 시 agency 필터 없음
    else:
        agency = fuzzy_match_agency(query)
        if agency:
            filters["agency"] = agency
    year = extract_year(query)
    if year:
        filters["year"] = year
    return filters


class BidMateRetriever:
    def __init__(self, collection, bm25_index, bm25_chunk_ids,
                 bm25_texts, embed_model, all_chunks, reranker=None):
        self.collection      = collection
        self.bm25_index      = bm25_index
        self.bm25_chunk_ids  = bm25_chunk_ids
        self.bm25_texts      = bm25_texts
        self.embed_model     = embed_model
        self.chunk_meta_map  = {c["chunk_id"]: c["metadata"] for c in all_chunks}
        self.chunk_text_map  = {c["chunk_id"]: c["text"]     for c in all_chunks}
        self._emb_cache: dict = {}
        self.reranker        = reranker

    def _build_chroma_where(self, meta_filter: dict):
        if not meta_filter:
            return None
        # ChromaDB 실제 키 매핑 (agency → organization_cleaned)
        key_map = {"agency": "organization_cleaned"}
        conditions = []
        for key, val in meta_filter.items():
            chroma_key = key_map.get(key, key)
            if not val:
                continue
            if isinstance(val, dict):
                conditions.append({chroma_key: val})
            elif isinstance(val, list):
                conditions.append({chroma_key: {"$in": [str(v) for v in val]}})
            else:
                conditions.append({chroma_key: {"$eq": str(val)}})
        if not conditions:
            return None
        return conditions[0] if len(conditions) == 1 else {"$and": conditions}

    def _filter_bm25_ids(self, meta_filter: dict):
        if not meta_filter:
            return None
        allowed = set()
        for i, cid in enumerate(self.bm25_chunk_ids):
            meta = self.chunk_meta_map.get(cid, {})
            if "agency" in meta_filter and meta.get("agency") != meta_filter["agency"]:
                continue
            if "year" in meta_filter and meta.get("year") != meta_filter["year"]:
                continue
            allowed.add(i)
        return list(allowed) if allowed else None

    def _decompose_query(self, query: str) -> list:
        clean_query     = query.replace("'", "").replace('"', "")
        sorted_agencies = sorted(ALL_AGENCIES, key=len, reverse=True)
        found_agencies  = []
        for agency in sorted_agencies:
            agency_core = re.sub(r"^\(주\)|\(사\)|주식회사", "", agency).strip()
            if len(agency_core) >= 2 and agency_core in clean_query:
                if not any(agency_core in fa[1] for fa in found_agencies):
                    found_agencies.append((agency, agency_core))
        if len(found_agencies) < 2:
            return [query]
        found_agencies = found_agencies[:3]
        found_agencies.sort(key=lambda x: clean_query.find(x[1]))
        segments = []
        for i, (agency_full, agency_core) in enumerate(found_agencies):
            start_idx    = clean_query.find(agency_core) + len(agency_core)
            end_idx      = clean_query.find(found_agencies[i+1][1]) if i < len(found_agencies)-1 else len(clean_query)
            segment_text = clean_query[start_idx:end_idx]
            tokens = kiwi.tokenize(segment_text, normalize_coda=True)
            nouns  = [t.form for t in tokens if (t.tag.startswith("N") or t.tag in ["SL","SN"]) and len(t.form) >= 2]
            segments.append({"agency": agency_full, "local_nouns": nouns})
        global_nouns = segments[-1]["local_nouns"]
        sub_queries  = []
        for seg in segments:
            local_nouns = seg["local_nouns"]
            final_nouns = global_nouns if not local_nouns and global_nouns else local_nouns
            sub_queries.append(f"{seg['agency']} {' '.join(final_nouns)}".strip())
        return sub_queries

    def _dense_search(self, query: str, where) -> list:
        q_emb  = self.embed_model.encode([query], normalize_embeddings=True).tolist()
        kwargs = dict(query_embeddings=q_emb, n_results=DENSE_K,
                      include=["documents","metadatas","distances","embeddings"])
        if where:
            kwargs["where"] = where
        results = self.collection.query(**kwargs)
        emb_list = results.get("embeddings")
        if emb_list is not None and len(emb_list) > 0 and len(emb_list[0]) > 0:
            for cid, emb in zip(results["ids"][0], emb_list[0]):
                if cid not in self._emb_cache:
                    self._emb_cache[cid] = emb
        return results["ids"][0]

    def _sparse_search(self, query: str, allowed_indices) -> list:
        tokens = tokenize_ko(query)
        if not tokens:
            return []
        scores = self.bm25_index.get_scores(tokens)
        if allowed_indices is not None:
            mask = np.zeros(len(scores))
            mask[allowed_indices] = 1.0
            scores = scores * mask
        top_indices = np.argsort(scores)[::-1][:SPARSE_K]
        return [self.bm25_chunk_ids[i] for i in top_indices if scores[i] > 0]

    def _multi_retrieve(self, queries, where, allowed_indices, original_query=""):
        import re as _re
        agency_pat = r"한국[가-힣]{1,6}공사|[가-힣]{2,8}공단|[가-힣]{2,8}은행|[가-힣]{2,8}공사|[가-힣]{2,8}연구원|[가-힣]{2,8}대학교|[가-힣]{2,8}의료원"
        dense_ids_all, sparse_ids_all = [], []
        seen_dense, seen_sparse = set(), set()
        # 각 sub_query별로 해당 기관 필터 개별 적용
        for q in queries:
            agencies = _re.findall(agency_pat, q)
            if agencies:
                q_filter = {"agency": agencies[0]}
                q_where = self._build_chroma_where(q_filter)
                q_allowed = self._filter_bm25_ids(q_filter)
            else:
                q_where = where
                q_allowed = allowed_indices
            for cid in self._dense_search(q, q_where):
                if cid not in seen_dense:
                    dense_ids_all.append(cid)
                    seen_dense.add(cid)
            for cid in self._sparse_search(q, q_allowed):
                if cid not in seen_sparse:
                    sparse_ids_all.append(cid)
                    seen_sparse.add(cid)
        # original_query는 필터 없이 보완 검색
        if original_query:
            for cid in self._dense_search(original_query, None):
                if cid not in seen_dense:
                    dense_ids_all.append(cid)
                    seen_dense.add(cid)
        return dense_ids_all, sparse_ids_all

    def _rrf_fusion(self, dense_ids, sparse_ids):
        rrf_scores = {}
        for rank, cid in enumerate(dense_ids,  start=1):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0/(RRF_K+rank)
        for rank, cid in enumerate(sparse_ids, start=1):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0/(RRF_K+rank)
        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    def _soft_boost(self, ranked):
        boosted = []
        for cid, score in ranked:
            meta = self.chunk_meta_map.get(cid, {})
            mul  = 1.0
            if meta.get("has_table",  False): mul += 0.10
            if meta.get("has_number", False): mul += 0.05
            boosted.append((cid, score * mul))
        return sorted(boosted, key=lambda x: x[1], reverse=True)

    def _mmr_rerank(self, boosted, query):
        candidates = boosted[:MMR_TOP_N]
        if len(candidates) <= 1:
            return candidates
        valid = [(cid, score) for cid, score in candidates if cid in self._emb_cache]
        if not valid:
            return candidates
        cids   = [cid for cid, _ in valid]
        scores = [score for _, score in valid]
        vecs   = np.array([self._emb_cache[cid] for cid in cids], dtype=np.float32)
        q_vec  = np.array(self.embed_model.encode([query], normalize_embeddings=True)[0], dtype=np.float32)
        norms     = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        vecs_norm = vecs / norms
        q_norm    = q_vec / (np.linalg.norm(q_vec) + 1e-9)
        rel_scores = vecs_norm @ q_norm
        def minmax(arr):
            mn, mx = arr.min(), arr.max()
            return (arr - mn) / (mx - mn + 1e-9)
        relevance     = (minmax(rel_scores) + minmax(np.array(scores))) / 2.0
        selected_idx  = []
        remaining_idx = list(range(len(cids)))
        while remaining_idx:
            if not selected_idx:
                best = max(remaining_idx, key=lambda i: relevance[i])
            else:
                sel_vecs = vecs_norm[selected_idx]
                best, best_mmr = -1, -float("inf")
                for i in remaining_idx:
                    mmr_score = MMR_LAMBDA * relevance[i] - (1-MMR_LAMBDA) * float(np.max(vecs_norm[i] @ sel_vecs.T))
                    if mmr_score > best_mmr:
                        best_mmr = mmr_score
                        best = i
            selected_idx.append(best)
            remaining_idx.remove(best)
        return [(cids[i], scores[i]) for i in selected_idx]

    def _rerank(self, boosted, query, rerank_top_n=RERANK_TOP_N):
        if self.reranker is None:
            return boosted
        candidates = boosted[:rerank_top_n]
        if not candidates:
            return boosted
        pairs    = [[query, self.chunk_text_map.get(cid, "")] for cid, _ in candidates]
        scores   = self.reranker.predict(pairs, show_progress_bar=False)
        reranked = sorted(zip([cid for cid, _ in candidates], scores), key=lambda x: x[1], reverse=True)
        reranked_ids = {cid for cid, _ in reranked}
        tail = [(cid, score) for cid, score in boosted[rerank_top_n:] if cid not in reranked_ids]
        return [(cid, float(score)) for cid, score in reranked] + tail

    def _build_context(self, top_chunks):
        lines = []
        for idx, (cid, score) in enumerate(top_chunks[:TOP_K], start=1):
            text = self.chunk_text_map.get(cid, "")
            meta = self.chunk_meta_map.get(cid, {})
            lines.append(f"[{idx}] {text}\n(출처: {meta.get('agency','미상')} {meta.get('year','')} | score: {score:.4f})")
        return "\n\n".join(lines)

    def retrieve(self, query: str, meta_filter=None, verbose: bool = False) -> dict:
        if meta_filter is None:
            meta_filter = parse_metadata_filter(query)
        where           = self._build_chroma_where(meta_filter)
        allowed_indices = self._filter_bm25_ids(meta_filter)
        sub_queries     = self._decompose_query(query)
        if len(sub_queries) > 1:
            dense_ids, sparse_ids = self._multi_retrieve(sub_queries, where, allowed_indices, original_query=query)
        else:
            dense_ids  = self._dense_search(query, where)
            sparse_ids = self._sparse_search(query, allowed_indices)
        ranked  = self._rrf_fusion(dense_ids, sparse_ids)
        boosted = self._soft_boost(ranked)
        boosted = self._mmr_rerank(boosted, query=query)
        boosted = self._rerank(boosted, query=query)
        # B타입 비교: sub_queries가 2개 이상이면 기관별 균등 배분
        if len(sub_queries) > 1:
            import re as _re
            agency_pat = r"한국[가-힣]{1,6}공사|[가-힣]{2,8}공단|[가-힣]{2,8}은행|[가-힣]{2,8}연구원|[가-힣]{2,8}대학교|[가-힣]{2,8}의료원"
            per_agency = max(2, TOP_K // len(sub_queries))
            agency_counts = {}
            top5 = []
            for cid, score in boosted:
                meta = self.chunk_meta_map.get(cid, {})
                ag = meta.get("agency", meta.get("organization_cleaned", ""))
                cnt = agency_counts.get(ag, 0)
                if cnt < per_agency:
                    top5.append((cid, score))
                    agency_counts[ag] = cnt + 1
                if len(top5) >= TOP_K:
                    break
        else:
            top5 = boosted[:TOP_K]
        # D타입 함정질문 대응: 최고 score 0.25 미만이면 문서 없음 처리
        if top5 and top5[0][1] < 0.25:
            top5 = []
        return {
            "context"    : self._build_context(top5),
            "top_chunks" : [{"rank": i+1, "chunk_id": cid, "boosted_score": score,
                             "text": self.chunk_text_map.get(cid,""), "metadata": self.chunk_meta_map.get(cid,{})}
                            for i, (cid, score) in enumerate(top5)],
            "meta_filter": meta_filter,
            "dense_ids"  : dense_ids,
            "sparse_ids" : sparse_ids,
            "sub_queries": sub_queries,
        }



import subprocess as _subprocess
import os

HWP_DIR = "/mnt/gukrul/dataset/original_data_list/files_advanced"


def _extract_hwp_text(original_name: str) -> str:
    hwp_path = os.path.join(HWP_DIR, original_name)
    if not os.path.exists(hwp_path):
        return ""
    try:
        result = _subprocess.run(
            ["hwp5txt", hwp_path],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _clean_hwp_text(text: str) -> str:
    import re
    text = re.sub("<.+?>", "", text)
    text = re.sub("\n{3,}", "\n\n", text)
    text = re.sub(" {2,}", " ", text)
    return text.strip()
def _split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    text = _clean_hwp_text(text)
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) <= chunk_size:
            current += "\n\n" + para if current else para
        else:
            if current:
                chunks.append(current)
            current = para
    if current:
        chunks.append(current)
    return [c for c in chunks if len(c.strip()) > 50]
def _select_relevant_sections(query: str, text: str, embed_model, top_n: int = 3) -> str:
    if not text:
        return ""
    sections = _split_text(text)
    if not sections:
        return ""
    try:
        q_emb  = embed_model.encode([query], normalize_embeddings=True)[0]
        s_embs = embed_model.encode(sections, normalize_embeddings=True, batch_size=32)
        scores = np.dot(s_embs, q_emb)
        top_idx = sorted(np.argsort(scores)[::-1][:top_n])
        return "\n\n".join([sections[i] for i in top_idx])
    except Exception:
        return text[:2000]


def _get_hwp_context(query: str, top_chunks: list, embed_model, top_docs: int = 2) -> str:
    seen, contexts = set(), []
    for chunk in top_chunks[:top_docs]:
        original_name = chunk.get("metadata", {}).get("original_name", "")
        if not original_name or original_name in seen:
            continue
        seen.add(original_name)
        full_text = _extract_hwp_text(original_name)
        if not full_text:
            continue
        selected = _select_relevant_sections(query, full_text, embed_model)
        if selected:
            agency  = chunk.get("metadata", {}).get("agency", "")
            project = chunk.get("metadata", {}).get("project_name", "")
            contexts.append(f"[{agency} - {project}]\n{selected}")
    return "\n\n---\n\n".join(contexts)

def get_context(query: str, history=None, meta_filter=None) -> dict:
    import re as _re
    effective_query = query
    if history:
        prev_user = [h["content"] for h in history if h["role"] == "user"]
        if prev_user:
            agency_pat = r"한국[가-힣]{1,6}공사|[가-힣]{2,8}공단|[가-힣]{2,8}은행|[가-힣]{2,8}공사|[가-힣]{2,8}연구원|[가-힣]{2,8}대학교|[가-힣]{2,8}의료원"
            curr_agency = _re.findall(agency_pat, query)
            prev_agency = _re.findall(agency_pat, prev_user[-1])
            # 현재 쿼리에 기관명 없을 때만 이전 기관명 보완 (전체 질문 붙이지 않음)
            if prev_agency and not curr_agency:
                effective_query = prev_agency[0] + " " + query
    result = retriever.retrieve(effective_query, meta_filter=meta_filter)

    context = result["context"]

    return {
        "context"    : context,
        "sources"    : [{"rank": c["rank"], "agency": c["metadata"].get("agency",""),
                         "year": c["metadata"].get("year",""),
                         "project": c["metadata"].get("project_name",""),
                         "score": c["boosted_score"]} for c in result["top_chunks"]],
        "filter"     : result["meta_filter"],
        "sub_queries": result["sub_queries"],
    }


def init_retriever() -> BidMateRetriever:
    global ALL_AGENCIES

    # 청크 로드
    all_chunks = load_chunks()
    ALL_AGENCIES = list({c["metadata"].get("agency","") for c in all_chunks if c["metadata"].get("agency","")})

    # 임베딩 모델
    embed_model = SentenceTransformer(
        EMBED_MODEL_ID,
        device="cpu",
        cache_folder="/mnt/gukrul/hf_cache/hub",
        local_files_only=True,
    )

    # ChromaDB
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # BM25
    with open(BM25_PATH, "rb") as f:
        bm25_data = pickle.load(f)
    bm25_index     = bm25_data["index"]
    bm25_chunk_ids = bm25_data["chunk_ids"]
    bm25_texts     = bm25_data["texts"]

    # Reranker
    reranker = CrossEncoder(
        RERANKER_ID,
        device=DEVICE,
    )

    return BidMateRetriever(
        collection=collection,
        bm25_index=bm25_index,
        bm25_chunk_ids=bm25_chunk_ids,
        bm25_texts=bm25_texts,
        embed_model=embed_model,
        all_chunks=all_chunks,
        reranker=reranker,
    )


# 모듈 임포트 시 자동 초기화
retriever: BidMateRetriever = None
