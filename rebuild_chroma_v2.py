import json, pickle, os
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CHUNKS_PATH  = '/mnt/gukrul/dataset/chunks/kh_fixed_with_budget.json'
CHROMA_PATH  = '/mnt/gukrul/dataset/chroma_db_v2'
COLLECTION   = 'bidmate_v2'
EMBED_MODEL  = 'nlpai-lab/KURE-v1'
HF_CACHE     = '/mnt/gukrul/hf_cache/hub'
BATCH_SIZE   = 128

print("청크 로드 중...")
with open(CHUNKS_PATH) as f:
    chunks = json.load(f)
print(f"청크 수: {len(chunks)}")

print("KURE-v1 로드 중...")
embed_model = SentenceTransformer(
    EMBED_MODEL,
    cache_folder=HF_CACHE,
    local_files_only=True,
    device='cuda',
)
print("로드 완료")

# ChromaDB 재생성
client = chromadb.PersistentClient(path=CHROMA_PATH)
try:
    client.delete_collection(COLLECTION)
    print("기존 컬렉션 삭제")
except:
    pass

collection = client.create_collection(
    name=COLLECTION,
    metadata={'hnsw:space': 'cosine'}
)

print(f"인덱싱 시작...")
for i in tqdm(range(0, len(chunks), BATCH_SIZE)):
    batch = chunks[i:i+BATCH_SIZE]
    texts = [c.get('text', c.get('content', '')) for c in batch]
    ids   = [c.get('chunk_id', f'chunk_{i+j}') for j, c in enumerate(batch)]
    metas = [{k: str(v) if v is not None else '' 
              for k, v in c.get('metadata', {}).items()} for c in batch]

    embeddings = embed_model.encode(
        texts, normalize_embeddings=True, batch_size=64
    ).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metas
    )

print(f"완료: {collection.count()}개")
