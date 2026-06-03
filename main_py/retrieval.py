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
    return {
        "chunk_id": c.get("chunk_id", ""),
        "text"    : c.get("chunk_text", c.get("text", "")),
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


def parse_metadata_filter(query: str) -> dict:
    filters = {}
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
        conditions = []
        for key, val in meta_filter.items():
            if not val:
                continue
            if isinstance(val, dict):
                conditions.append({key: val})
            elif isinstance(val, list):
                conditions.append({key: {"$in": [str(v) for v in val]}})
            else:
                conditions.append({key: {"$eq": str(val)}})
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
        all_queries    = ([original_query] if original_query else []) + list(queries)
        dense_ids_all, sparse_ids_all = [], []
        seen_dense, seen_sparse = set(), set()
        for q in all_queries:
            for cid in self._dense_search(q, where):
                if cid not in seen_dense:
                    dense_ids_all.append(cid)
                    seen_dense.add(cid)
            for cid in self._sparse_search(q, allowed_indices):
                if cid not in seen_sparse:
                    sparse_ids_all.append(cid)
                    seen_sparse.add(cid)
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
        top5    = boosted[:TOP_K]
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


def get_context(query: str, history=None, meta_filter=None) -> dict:
    effective_query = query
    if history:
        prev_user = [h["content"] for h in history if h["role"] == "user"]
        if prev_user:
            effective_query = f"{prev_user[-1]} {query}"
    result = retriever.retrieve(effective_query, meta_filter=meta_filter)
    return {
        "context"    : result["context"],
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
        device=DEVICE,
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
