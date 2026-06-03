import os
from pathlib import Path

# 환경 설정
ENV = os.environ.get("BIDMATE_ENV", "gcp")

# HuggingFace 캐시 경로
os.environ["HF_HOME"]           = "/mnt/gukrul/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/gukrul/hf_cache/hub"

# 경로
PROJECT_ROOT = Path("/mnt/gukrul")
DATASET_DIR  = PROJECT_ROOT / "dataset"
CHUNKS_PATH  = DATASET_DIR / "chunks" / "kh_fixed_1200_200_v2.json"
CHROMA_PATH  = DATASET_DIR / "chroma_db_hej"
BM25_PATH    = DATASET_DIR / "bm25_index.pkl"
EVAL_PATH    = DATASET_DIR / "eval"
RESULT_DIR   = DATASET_DIR / "eval_results"
ADAPTER_PATH = DATASET_DIR / "peft_output" / "gemma4-E4B_v2" / "lora_adapter_v2"
LOG_PATH     = PROJECT_ROOT / "web_user_access.log"

# 모델
BASE_MODEL_ID   = "google/gemma-4-e4b-it"
EMBED_MODEL_ID  = "nlpai-lab/KURE-v1"
RERANKER_ID     = "BAAI/bge-reranker-v2-m3"
LLM_MODEL       = "google/gemma-4-e4b-it"

# 생성 파라미터
MAX_TOKENS_REWRITE  = 300
MAX_TOKENS_GENERATE = 1500

# Retrieval 파라미터
COLLECTION_NAME = "bidmate_retrieval_v1"
DENSE_K         = 15
SPARSE_K        = 15
RRF_K           = 60
TOP_K           = 5
MMR_LAMBDA      = 0.6
MMR_TOP_N       = 20
RERANK_TOP_N    = 15
BATCH_SIZE      = 64
