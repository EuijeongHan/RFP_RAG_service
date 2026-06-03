"""
입찰메이트 RAG — 통합 서빙 파이프라인
======================================
담당 : Generation / Retrieval 파트 (한의정)

[설명]
- Retrieval(컨텍스트 추출) → Query Router → Prompt 구성 → Generation(답변 생성) 파이프라인.
- BidMateApp 이 세션 히스토리를 관리하며 C타입(후속 질문) 시나리오에 대응.
- CLI 테스트 / FastAPI 서빙 양쪽에서 사용 가능.

[히스토리 처리 설계]
  ① route_and_search(query, history) → 질문 유형을 판별해 최적 인덱스로 라우팅
  ② get_context(query, history) → Retrieval 단 쿼리 보정 (C타입: 직전 발화 접합)
  ③ generator.generate(prompt_dict, history) → [이전 대화] 블록 삽입으로 맥락 유지

[Query Router 설계 근거 (retrieval_history_7.ipynb 지표 기반)]
  chunks_all (300자) → MRR 기준 전체 1위
    A타입(Hit@5 0.9302), B타입(0.9579) → 단일 팩트 검색 최강
    → 일반 질문(A·B·D·E)에 사용: 노이즈 없이 정답을 1등으로 꽂아 넣음

  kh_v3_hybrid (600자 + parent_text) → C타입 히스토리 평가 1위(0.8852)
    → C타입(히스토리 복합), D타입(거절), E타입(요약)에 사용
    D: 더 넓은 문맥이 있어야 "답 없음" 을 확실히 판단 가능
    E: 요약은 파편화된 300자보다 600자 + 부모 텍스트가 필요

  ⚠️ 라우터 자체 오버헤드: if문 + 키워드 체크 = 1~5ms (무시 가능)
  ⚠️ 두 인덱스를 서버 기동 시 Warm-up 해두면 접속 지연 없음
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional

import chromadb
from sentence_transformers import CrossEncoder

# Retrieval 파트 내부 빌더 함수 직접 임포트
# get_context() 대신 BidMateRetriever 를 직접 두 개 생성해서 보유
# → os.environ race condition 완전 제거
from retrieval_interface_F import (
    _tokenize_ko,
    build_prompt,
    BidMateRetriever,
    EMBED_CONFIG,
    _load_chunks,
    _build_embed_model,
    _build_or_load_chroma,
    _build_or_load_bm25,
    _parse_meta_filter_with,
    _RERANK_MODEL,
    _DEVICE,
)
# Generation 파트
from generation_interface import get_generator, BidMateGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
_logger = logging.getLogger(__name__)

# ======================================================================
# 경로 설정 (retrieval_interface_F.py 와 동일 패턴)
# ======================================================================
try:
    import google.colab; _ENV = 'colab'
except ImportError:
    _ENV = os.environ.get('BIDMATE_ENV', 'gcp')

if _ENV == 'gcp':
    _PROJECT_ROOT = Path('/mnt/gukrul/hej')
    _CHROMA_PATH  = Path('/mnt/gukrul/dataset/hej/chroma_db')
    _BM25_DIR     = _PROJECT_ROOT / 'bm25'
    # 두 인덱스 청크 파일 경로
    _CHUNKS_ALL_PATH = _PROJECT_ROOT / 'chunks' / 'chunks_all.json'
    _CHUNKS_KH_PATH  = _PROJECT_ROOT / 'chunks' / 'kh_v3.json'
else:  # colab / 로컬
    _PROJECT_ROOT    = Path.home() / 'Desktop' / 'AI엔지니어_8기' / '프로젝트' / '중급'
    _CHROMA_PATH     = _PROJECT_ROOT / 'chroma_db'
    _BM25_DIR        = _PROJECT_ROOT / 'data'
    _CHUNKS_ALL_PATH = _PROJECT_ROOT / 'data' / 'chunks' / 'chunks_all.json'
    _CHUNKS_KH_PATH  = _PROJECT_ROOT / 'data' / 'chunks' / 'kh_v3.json'

# ChromaDB 클라이언트 — 프로세스 당 하나만 생성
_chroma_client = chromadb.PersistentClient(path=str(_CHROMA_PATH))

# 세션 히스토리 최대 유지 메시지 수
_MAX_HISTORY_MSGS = 20  # 10턴 (user + assistant 쌍)


# ======================================================================
# 듀얼 Retriever 초기화
# ======================================================================
# ⚠️ 설계 결정: os.environ 교체 방식을 쓰지 않는 이유
#
#   retrieval_interface_F.py 는 USE_HYBRID 를 모듈 로드 시점에
#   상수로 고정한다 (USE_HYBRID = os.environ.get(...)).
#   따라서 런타임에 os.environ 을 바꿔도 Retriever 내부 값은 변하지 않음.
#   또한 FastAPI 멀티 요청 환경에서 os.environ 은 프로세스 공유 자원이므로
#   동시 요청 시 race condition 발생 위험이 있음.
#
#   → 해결책: BidMateRetriever 인스턴스를 두 개 직접 생성해서 보유.
#     _retriever_main : chunks_all + use_hybrid=False  (A·B타입 전용)
#     _retriever_ctx  : kh_v3     + use_hybrid=True   (C·D·E타입 전용)
#     각 인스턴스는 독립적이며 상태를 공유하지 않으므로 thread-safe.

def _build_retriever(
    chunks_path : Path,
    embed_scenario: str,
    use_hybrid  : bool,
    label       : str,
) -> BidMateRetriever:
    """
    BidMateRetriever 인스턴스 생성 헬퍼.

    Parameters
    ----------
    chunks_path    : 청크 JSON 파일 경로
    embed_scenario : 'KURE' | 'KOE5' | 'SMALL' (임베딩 모델 선택)
    use_hybrid     : True → Child-to-Parent Retrieval 활성화
    label          : 로그 식별용 레이블
    """
    _logger.info(f'[Router] {label} Retriever 초기화 시작 | use_hybrid={use_hybrid}')

    chunks      = _load_chunks(chunks_path)
    embed_model = _build_embed_model(embed_scenario)
    col_name    = EMBED_CONFIG[embed_scenario]['collection']

    # ⚠️ 두 Retriever 가 같은 임베딩 시나리오를 쓰면 컬렉션이 겹침
    #    chunks_all 과 kh_v3 는 서로 다른 청크이므로 컬렉션명을 분리해야 함
    #    충돌 방지를 위해 레이블을 컬렉션명에 suffix 로 추가
    col_name_with_label = f'{col_name}_{label}'

    col                             = _build_or_load_chroma(
        _chroma_client, col_name_with_label, chunks, embed_model
    )
    bm25_path                       = _BM25_DIR / f'bm25_index_{col_name_with_label}.pkl'
    bm25_idx, bm25_cids, bm25_texts = _build_or_load_bm25(chunks, bm25_path)
    reranker                        = CrossEncoder(_RERANK_MODEL, device=_DEVICE)

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
    _logger.info(f'[Router] ✅ {label} Retriever 초기화 완료')
    return retriever


def _retriever_search(
    retriever  : BidMateRetriever,
    query      : str,
    history    : list,
    meta_filter: dict = None,
) -> dict:
    """
    retrieval_interface_F.get_context() 와 동일한 로직을 Retriever 인스턴스 직접 호출로 구현.

    C타입 쿼리 보정 (히스토리 직전 user 발화 + 현재 쿼리 접합)도 동일하게 적용.
    """
    # C타입 쿼리 보정 — get_context() 내부 로직과 동일
    effective = query
    if history:
        prev = [h['content'] for h in history if h.get('role') == 'user']
        if prev:
            effective = f'{prev[-1]} {query}'

    # 메타 필터 자동 감지 (없으면 쿼리에서 기관명/연도 추출)
    if meta_filter is None:
        meta_filter = _parse_meta_filter_with(effective, retriever._all_agencies)

    result = retriever.retrieve(effective, meta_filter=meta_filter)

    return {
        'context'    : result['context'],
        'sources'    : [
            {
                'rank'   : c['rank'],
                'agency' : c['metadata'].get('agency', ''),
                'year'   : c['metadata'].get('year', ''),
                'project': c['metadata'].get('project_name', ''),
                'score'  : c['boosted_score'],
            }
            for c in result['top_chunks']
        ],
        'filter'     : result['meta_filter'],
        'sub_queries': result['sub_queries'],
    }


# ======================================================================
# Query Router — 질문 유형 판별
# ======================================================================
# C타입: 대명사/지시어 포함 후속 질문
_CTYPE_KEYWORDS = [
    '그 ', '그거 ', '그것 ', '저 ', '저것 ',
    '위에서 ', '앞서 ', '이전 ', '아까 ',
    '해당 사업', '해당 기관', '거기 ',
]

# ⚠️ D·E타입을 kh_v3_hybrid로 보내지 않는 이유 (데이터 근거):
# retrieval_history_7.ipynb 청크 전략 비교 (579개, Hit@5) 기준:
#   D타입: chunks_all 0.8769  vs  kh_v3_hybrid 0.8308  → chunks_all +4.6%p
#   E타입: chunks_all 0.8154  vs  kh_v3_hybrid 0.6923  → chunks_all +12.3%p
# kh_v3_hybrid가 chunks_all을 이기는 타입은 C타입(히스토리)뿐.
# Gemini는 D·E도 긴 청크가 낫다고 주장했으나 실측 데이터와 반대.


def _is_ctype(query: str, history: list) -> bool:
    """
    C타입(히스토리 의존 후속 질문) 판별.

    두 조건 중 하나라도 충족하면 C타입:
      1) 지시어/대명사 포함 — '그 사업 납기는?', '앞서 말한 예산은?'
      2) 직전 user 발화와 현재 쿼리 간 공통 형태소 키워드 없음
         '한국가스공사 ERP 예산은?' → '납기는?' (공통 없음 → C타입)
         '한국가스공사 ERP 예산은?' → '한국가스공사 계약기간은?' (공통 있음 → A타입)

    ⚠️ len(query) < 15 조건 제거 이유:
       '한국가스공사 ERP 예산' (13자) 같은 정상 A타입이 오분류되는 문제.
       (Gemini 코드 리뷰에서 지적된 휴리스틱 취약점)

    ⚠️ _tokenize_ko() 재사용:
       retrieval_interface_F.py 의 kiwi 기반 형태소 분석기.
       조사 제거로 '예산은' = '예산' 매칭 가능. 의존성 추가 없음.
    """
    if not history:
        return False

    # 조건 1: 지시어/대명사 포함
    if any(kw in query for kw in _CTYPE_KEYWORDS):
        return True

    # 조건 2: 직전 user 발화와 현재 쿼리 공통 키워드 없음
    prev_user = [h['content'] for h in history if h.get('role') == 'user']
    if not prev_user:
        return False

    prev_tokens = set(_tokenize_ko(prev_user[-1]))
    curr_tokens  = set(_tokenize_ko(query))

    if not curr_tokens:
        return False

    if not (prev_tokens & curr_tokens):
        _logger.debug(f'[Router] 키워드 오버랩 없음 → C타입 | prev={prev_tokens} | curr={curr_tokens}')
        return True

    return False


def _classify_query(query: str, history: list) -> str:
    """
    데이터 기반 라우팅 (retrieval_history_7.ipynb, Hit@5):
      C타입(히스토리): kh_v3_hybrid 0.8852 > chunks_all 0.8361  → ctx
      A·B·D·E타입   : chunks_all 전체 1위                        → main
    """
    if _is_ctype(query, history):
        _logger.info(f'[Router] C타입 → ctx(kh_v3_hybrid) | {query[:30]}')
        return 'ctx'

    _logger.info(f'[Router] A/B/D/E타입 → main(chunks_all) | {query[:30]}')
    return 'main'


def route_and_search(
    query             : str,
    history           : list,
    meta_filter       : dict = None,
    retriever_main    : BidMateRetriever = None,
    retriever_ctx     : BidMateRetriever = None,
) -> dict:
    """
    쿼리 유형에 따라 적절한 Retriever 인스턴스로 검색 수행.

    Parameters
    ----------
    retriever_main : chunks_all 기반 Retriever (A·B타입)
    retriever_ctx  : kh_v3_hybrid 기반 Retriever (C·D·E타입)
                     None 이면 retriever_main 으로 fallback

    ⚠️ 두 Retriever 는 BidMateApp 초기화 시 생성되어 인스턴스로 보유.
       os.environ 을 전혀 건드리지 않으므로 멀티 요청 환경에서 완전 안전.
    """
    target = _classify_query(query, history)

    if target == 'ctx' and retriever_ctx is not None:
        retriever = retriever_ctx
    else:
        # retriever_ctx 가 None(미초기화)이면 main 으로 fallback
        if target == 'ctx':
            _logger.warning('[Router] kh_v3_hybrid Retriever 미초기화 → chunks_all fallback')
        retriever = retriever_main

    result             = _retriever_search(retriever, query, history, meta_filter)
    result['routed_to'] = target
    return result


class BidMateApp:
    """
    입찰메이트 RAG 애플리케이션 메인 클래스.

    단일 인스턴스를 서버 프로세스 생애주기 동안 재사용.
    세션 히스토리는 인스턴스 내부에 보관 (멀티유저 환경에서는 Redis 등으로 외부화 필요).

    사용 예:
        app = BidMateApp()
        result = app.chat("한국가스공사 ERP 예산은?")
        print(result['answer'])

        # 세션 초기화 (새 사용자/새 주제)
        app.reset()
    """

    def __init__(self, scenario: Optional[str] = None):
        self.scenario = scenario or os.environ.get('EMBED_SCENARIO', 'KURE_GEMMA')
        self.generator: BidMateGenerator = get_generator(self.scenario)
        embed_scenario = self.scenario.split('_')[0]  # 'KURE_GEMMA' → 'KURE'
        self._retriever_main = _build_retriever(
            chunks_path=_CHUNKS_ALL_PATH, embed_scenario=embed_scenario,
            use_hybrid=False, label='main',
        )
        if _CHUNKS_KH_PATH.exists():
            self._retriever_ctx = _build_retriever(
                chunks_path=_CHUNKS_KH_PATH, embed_scenario=embed_scenario,
                use_hybrid=True, label='ctx',
            )
        else:
            _logger.warning(f'[App] kh_v3 청크 없음. C/D/E타입 → chunks_all fallback.')
            self._retriever_ctx = None
        self.chat_history: List[Dict] = []
        _logger.info(f'[App] ✅ BidMateApp 초기화 완료 | scenario={self.scenario}')

    def chat(self, user_query: str, meta_filter: Optional[dict] = None) -> dict:
        """
        사용자 쿼리를 받아 RAG 파이프라인 실행 후 결과 반환.

        Parameters
        ----------
        user_query  : 사용자 질문
        meta_filter : 메타데이터 필터 (agency, year 등). None이면 자동 감지.

        Returns
        -------
        {
            'answer'     : str,   # 최종 답변
            'sources'    : list,  # 출처 정보
            'sub_queries': list,  # 쿼리 분해 결과 (B타입 멀티 기관 질문)
        }
        """
        _logger.info(f'[App] Query 수신: {user_query[:60]}...')

        # ─── Step 1. Query Router + Retrieval ──────────────────────────
        # 질문 유형(A/B/C/D/E)을 판별하여 최적 인덱스로 라우팅 후 검색.
        # C타입: 지시어·히스토리 감지 → kh_v3_hybrid (600자, 문맥 우선)
        # A·B·D·E타입: chunks_all (300자, 팩트 정밀도 우선, MRR 전체 1위)
        # ⚠️ D·E타입도 실측 데이터 기준 chunks_all이 우세 (retrieval_history_7.ipynb)
        retrieval_result = route_and_search(
            query          = user_query,
            history        = self.chat_history,
            meta_filter    = meta_filter,
            retriever_main = self._retriever_main,
            retriever_ctx  = self._retriever_ctx,
        )

        # ─── Step 2. Prompt 구성 ────────────────────────────────────────
        # retrieval_interface_F.build_prompt() → scenario에 맞는 포맷 반환
        # GEMMA: {'prompt': '<start_of_turn>...', 'system': None, 'user': None}
        # OPENAI: {'prompt': None, 'system': '...', 'user': '...'}
        prompt_dict = build_prompt(
            query    = user_query,
            context  = retrieval_result['context'],
            scenario = self.scenario,
        )

        # ─── Step 3. Generation ─────────────────────────────────────────
        # history를 넘겨 [이전 대화] 블록을 프롬프트에 삽입 (대화 맥락 유지 목적)
        # 역할 분리:
        #   Step 1의 history 활용 = 검색 쿼리 보정 (retrieval 레이어)
        #   Step 3의 history 활용 = 답변 맥락 유지 (generation 레이어)
        answer = self.generator.generate(
            prompt_dict = prompt_dict,
            history     = self.chat_history,   # Generation 단 히스토리 활용 (맥락 유지 목적)
        )

        # ─── Step 4. 히스토리 업데이트 ──────────────────────────────────
        # generate() 호출 완료 후 업데이트해야 현재 턴이 중복 입력되지 않음
        self.chat_history.append({'role': 'user',      'content': user_query})
        self.chat_history.append({'role': 'assistant', 'content': answer})

        # 최대 유지 길이 초과 시 오래된 것부터 제거
        if len(self.chat_history) > _MAX_HISTORY_MSGS:
            self.chat_history = self.chat_history[-_MAX_HISTORY_MSGS:]

        return {
            'answer'     : answer,
            'sources'    : retrieval_result['sources'],
            'sub_queries': retrieval_result['sub_queries'],
        }

    def reset(self):
        """
        세션 초기화 (새 주제 또는 새 사용자 시작 시 호출).
        히스토리를 비우면 C타입 추적도 초기화된다.
        """
        self.chat_history.clear()
        _logger.info('[App] 세션 히스토리 초기화')

    def release(self):
        """GPU 메모리 해제. 프로세스 종료 또는 시나리오 전환 전 호출."""
        self.generator.release()
        # Retriever 는 별도 GPU 메모리를 점유하지 않으나
        # Reranker(CrossEncoder) 가 CUDA 메모리를 사용하므로 명시적 해제
        del self._retriever_main
        del self._retriever_ctx
        _logger.info('[App] Retriever 해제 완료')


# ======================================================================
# CLI 테스트용 실행부
# ======================================================================
if __name__ == '__main__':
    # 시나리오는 환경변수로 제어
    # 예: EMBED_SCENARIO=KURE_GEMMA python serving_main.py
    # 예: EMBED_SCENARIO=SMALL_OPENAI OPENAI_LLM_MODEL=gpt-5-mini python serving_main.py
    # 예: EMBED_SCENARIO=SMALL_OPENAI OPENAI_LLM_MODEL=gpt-4.1-mini python serving_main.py

    app = BidMateApp()
    print('=' * 60)
    print(f'입찰메이트 챗봇 (시나리오: {app.scenario})')
    print("종료: 'exit' | 세션 초기화: 'reset'")
    print('=' * 60)

    while True:
        query = input('\nQ: ').strip()
        if not query:
            continue
        if query.lower() == 'exit':
            app.release()
            break
        if query.lower() == 'reset':
            app.reset()
            print('[세션 초기화 완료]')
            continue

        result = app.chat(query)
        print(f'\nA: {result["answer"]}')

        if result['sources']:
            print('\n[출처]')
            for src in result['sources']:
                print(
                    f"  {src['rank']}. {src['agency']} | {src['project']} "
                    f"({src['year']}년) | score={src['score']:.4f}"
                )

        if len(result['sub_queries']) > 1:
            print(f'\n[쿼리 분해] {result["sub_queries"]}')
