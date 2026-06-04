import logging
import retrieval
import generation
from generation import format_sources
logger = logging.getLogger(__name__)
class GenerationService:
    """
    web_rag.py 전용 래퍼.
    서버 시작 시 1회 초기화 후 재사용.
    """
    def __init__(self):
        self._init()
    def _init(self):
        logger.info("Retriever 초기화 중...")
        retrieval.retriever = retrieval.init_retriever()
        retrieval.retriever_c = retrieval.init_retriever_for(
            collection_name="bidmate_kh_v3",
            chunks_path="/mnt/gukrul/dataset/chunks/kh_v3.json",
            bm25_path="/mnt/gukrul/dataset/bm25/bm25_index_kh_v3.pkl",
        )
        retrieval.retriever_bde = retrieval.init_retriever_for(
            collection_name="bidmate_chunks_all",
            chunks_path="/mnt/gukrul/dataset/chunks/chunks_all.json",
            bm25_path="/mnt/gukrul/dataset/bm25/bm25_index_bidmate_chunks_all_A-1.pkl",
            chroma_key_map={},
        )
        logger.info("Generator 초기화 중...")
        generation.generator = generation.init_generator(retrieval.get_context)
        self._gen = generation.generator
        logger.info("GenerationService 초기화 완료")

    def set_llm_config(self, provider: str, api_key: str, model: str = None):
        self._gen.set_llm_config(provider=provider, api_key=api_key, model=model)

    def ask(self, query: str, history: list = None, llm_config: dict = None) -> dict:
        if llm_config:
            self.set_llm_config(**llm_config)
        return self._gen.generate(query=query, history=history)

    def stream(self, query: str, history: list = None):
        yield from self._gen.generate_stream(query=query, history=history)

    @staticmethod
    def format_history(messages: list) -> list:
        return [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m["role"] in ("user", "assistant")
        ]
