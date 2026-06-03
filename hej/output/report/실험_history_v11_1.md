# 입찰메이트 RAG — 실험 히스토리 v11

**작성일** : 2026-05-28  
**담당** : 한의정 (Retrieval + Generation 평가)  
**목적** : EDA부터 현재까지 전 과정을 v11 하나로 파악 가능하도록 정리

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [EDA 및 데이터 분석](#2-eda-및-데이터-분석)
3. [프로젝트 초기 설정](#3-프로젝트-초기-설정)
4. [Retrieval 파이프라인 구축](#4-retrieval-파이프라인-구축)
5. [Retrieval 평가 및 전략 선정](#5-retrieval-평가-및-전략-선정)
6. [코드 리팩토링 및 통합](#6-코드-리팩토링-및-통합)
7. [시나리오 구조 개편](#7-시나리오-구조-개편)
8. [Generation 모듈 구축](#8-generation-모듈-구축)
9. [E2E 평가 파이프라인 구조](#9-e2e-평가-파이프라인-구조)
10. [모델 스택 확정 및 업그레이드](#10-모델-스택-확정-및-업그레이드)
11. [E2E 평가 전체 실행 완료](#11-e2e-평가-전체-실행-완료)
12. [Judge 채점 현황](#12-judge-채점-현황)
13. [파일 구성 최종 현황](#13-파일-구성-최종-현황)
14. [GCP 환경](#14-gcp-환경)
15. [Release Gate 기준](#15-release-gate-기준)
16. [현재 진행 상태 및 다음 순서](#16-현재-진행-상태-및-다음-순서)

---

## 1. 프로젝트 개요

**프로젝트명**: 입찰메이트 (BidMate) — B2G RAG 시스템  
**목적**: 690개 공공조달 제안요청서(RFP/HWP) 기반 Q&A 서비스  
**팀**: 5명 (한의정 — Retrieval + Generation 평가 담당)  
**환경**: GCP L4 24GB (codeit-ai-02), Mac M2, Python 3.12

### 서빙 아키텍처 다이어그램

```
사용자 쿼리
    │
    ▼
BidMateApp.chat()  ← serving_main.py
    │
    ├─ _classify_query()
    │   ├─ 지시어 키워드 포함? ('그 ', '앞서 ', '해당 사업' 등)
    │   └─ 직전 발화와 형태소 오버랩 없음?
    │        ↓ Yes → C타입       ↓ No → A/B/D/E타입
    │
    ├─ _retriever_ctx            _retriever_main
    │  kh_v3 + use_hybrid=True   chunks_all + use_hybrid=False
    │
    ▼
retrieval_interface_F.py — BidMateRetriever
    ① 메타데이터 Hard Filter (agency/year 퍼지 매칭)
    ② Dense 검색 (k=15, KURE-v1/KoE5/OpenAI)
    ③ Sparse 검색 (k=15, BM25+Kiwi)
    ④ RRF 융합 (K=60)
    ⑤ Soft Boost (has_table×1.10, has_number×1.05)
    ⑥ MMR 재정렬 (λ=0.6)
    ⑦ Reranker (bge-reranker-v2-m3) → Top-5
    │
    ▼
generation_interface.py — get_generator(scenario)
    ├─ GEMMA  → LocalHFGenerator (AutoModelForImageTextToText)
    ├─ QWEN   → LocalHFGenerator (AutoModelForCausalLM)
    ├─ PHI    → LocalHFGenerator (AutoModelForCausalLM)
    └─ OPENAI → APIGenerator (gpt-5-mini)
    │
    ▼
최종 답변 반환 {'answer', 'sources', 'sub_queries'}
```

---

## 2. EDA 및 데이터 분석

### 2-1. 데이터 현황

| 항목 | 내용 |
|---|---|
| 총 문서 수 | 690개 (HWP/HWPX/PDF 혼합) |
| 기관 수 | 406개 기관 |
| eval 질문 수 | 1,100개 → 579개 유니크 질문 |

### 2-2. eval 질문 타입 분류

| 타입 | 의미 | 개수 |
|---|---|---|
| A | 단일 문서 사실 추출 | 172개 |
| B | 멀티 문서 비교 | 214개 |
| C | 히스토리 의존 후속 질문 | 63개 |
| D | 함정 질문 (문서에 답 없음) | 65개 |
| E | 오타/변형 질문 | 65개 |

### 2-3. EDA 주요 발견사항

- **Mac OS NFD 이슈**: 파일명 한글이 NFD로 저장 → `unicodedata.normalize('NFC', ...)` 필수
- **HWP 파싱 노이즈**: 한자 연속 3자 이상은 인코딩 찌꺼기로 제거, `[별지]/[서식]/붙임` 이후 행정 양식 절삭
- **예산 파싱**: 단순 숫자 추출 불가, 앞뒤 500자 내 단위 문맥(천원/백만원/억원) 탐색 후 multiplier 적용
- **메타데이터 구조**: `organization_cleaned`, `project_name`, `year`, `domains` 6개 필드 (v2 청크)

### 2-4. 청크 전략 설계 (4종)

| 전략 | 크기 | 특징 |
|---|---|---|
| kh_fixed_v1 | 1000자/overlap 150 | 청크 수 19,118개 |
| kh_fixed_v2 | 1200자/overlap 200 | 청크 수 16,248개, 메타데이터 풍부 |
| kh_v3 | 600자 | Child-to-Parent 구조 |
| chunks_all | 300자 | 최소 단위, 전체 10,068개 |

---

## 3. 프로젝트 초기 설정

### 3-1. 확정 스택

| 항목 | 선택 | 이유 |
|---|---|---|
| 임베딩 | KURE-v1 (주), KoE5 (비교), text-embedding-3-small (비교) | 한국어 특화 |
| VectorDB | ChromaDB | 메타데이터 필터 내장 |
| BM25 | Kiwi 기반 | 한국어 형태소 분석 |
| Reranker | bge-reranker-v2-m3 | |
| 서버 | GCP L4 24GB | CUDA 13.0 |

### 3-2. ChromaDB 컬렉션 목록

```
C타입용 (kh_v3_hybrid):
  KURE  → bidmate_kh_v3_A-1
  KOE5  → bidmate_kh_v3_A-2
  SMALL → bidmate_kh_v3_B

ABDE용 (chunks_all):
  KURE  → bidmate_kure
  KOE5  → bidmate_chunks_all_A-2  (구버전명 유지)
  SMALL → bidmate_chunks_all_B
```

### 3-3. Phase 1 베이스라인 (naive dense, 초기)

| 타입 | Hit@5 | MRR | nDCG |
|---|---|---|---|
| A | 0.9390 | 0.9123 | 0.9192 |
| B | 0.8875 | 0.8529 | 0.7126 |
| C | 0.7622 | 0.7383 | 0.7444 |
| D | 0.8827 | 0.8410 | 0.8513 |
| E | 0.8269 | 0.8154 | 0.8183 |

---

## 4. Retrieval 파이프라인 구축

### 4-1. 핵심 설계 결정

**① Child-to-Parent Retrieval (kh_v3)**
- Child(300자)로 정밀 검색 → parent_text(600자) LLM에 전달
- 검색 정밀도 + 문맥 완결성 동시 확보

**② Race Condition 원천 차단**
- 기존: `os.environ` 교체 방식 (멀티 요청 시 상태 충돌)
- 변경: 서빙 초기화 시 두 개의 독립 BidMateRetriever 인스턴스 보유
  - `_retriever_main`: chunks_all, use_hybrid=False
  - `_retriever_ctx`: kh_v3, use_hybrid=True

**③ C타입 분류기 개선**
- 제거된 오류 로직: `len(query) < 15` 휴리스틱 → 정상 A타입 질문도 C타입으로 오분류
- 채택된 로직: 지시어 키워드 매칭 + `_tokenize_ko()` 형태소 오버랩 기반

**④ GCP Reranker 레이턴시 측정**
- 단일 쿼리 평균: 889ms
- 멀티 쿼리(B타입) 평균: 1,270ms
- 결정: Reranker 포함 여부는 Generation E2E 통합 후 결정하기로 유보

### 4-2. chunks_all 버그 수정

`chunks_all.json`의 `parent_text`가 metadata 안에 있어서 Child-to-Parent Retrieval이 빈 문자열로 silently fail하는 버그 발견 및 수정

### 4-3. Thomas 데이터 이슈

- Thomas 제공 청크: `RFP_한국전력_2023_0001.hwp` (내부 코드명)
- Eval ground truth: `한국전력공사_....hwp` (원본 파일명)
- 결과: Hit@5 = 0.0000 → doc_id-to-filename 매핑 테이블 요청했으나 미해결

---

## 5. Retrieval 평가 및 전략 선정

### 5-1. 전체 평가 결과 (579개 유니크 질문)

**chunks_all (최고 성능):**

| 임베딩 | Hit@5 | MRR | nDCG |
|---|---|---|---|
| KURE-v1 (A-1) | **0.8998** | **0.8487** | 0.8138 |
| KoE5 (A-2) | 0.8900 | 0.8380 | — |
| OpenAI Small (B) | 0.8800 | 0.8341 | — |

**kh_v3 hybrid (C타입 특화):**

| 임베딩 | Hit@5 | MRR |
|---|---|---|
| KURE-v1 | 0.8689 | 0.7342 |
| KoE5 | **0.8852** | 0.7281 |

### 5-2. 타입별 채택 전략 (MRR + Hit@5 기준)

| 타입 | chunks_all MRR | kh_v3h MRR | chunks_all Hit@5 | kh_v3h Hit@5 | **채택** |
|---|---|---|---|---|---|
| A | 0.8329 | 0.8380 | **0.9302** | 0.9012 | chunks_all (Hit@5 우세) |
| B | 0.8359 | 0.8660 | **0.9579** | 0.9252 | chunks_all (Hit@5 우세) |
| C (히스토리) | 0.7415 | 0.7342 | 0.8361 | **0.8689** | kh_v3 hybrid (Hit@5 +3.3%p) |
| D | **0.8487** | 0.7090 | **0.8769** | 0.8154 | chunks_all |
| E | **0.8138** | 0.5923 | **0.8154** | 0.6769 | chunks_all |

> MRR이 높을수록 정답이 Top-1에 가까워 LLM의 Lost in the Middle 영향 감소.  
> A/B타입: kh_v3 MRR 앞서지만 Hit@5에서 chunks_all이 +2~3%p 우세 → chunks_all 채택.  
> C타입: 두 지표 모두 kh_v3 hybrid 우세 → kh_v3 hybrid 채택.  
> D/E타입: chunks_all이 MRR, Hit@5 모두 압도적 우세.

---

## 6. 코드 리팩토링 및 통합

**기존**: 15개 이상 개별 Python 파일  
**변경**: 3개 통합 파일

| 파일 | 역할 |
|---|---|
| `retrieval_interface_F.py` | Retrieval 핵심 (BidMateRetriever, EMBED_CONFIG, build_prompt, Prompt Compression, HyDE) |
| `retrieval_eval_F.py` | 평가 전용 Retriever (BidMateEvaluator, GPU 메모리 명시적 해제) |
| `run_eval_F.py` | 배치 평가 실행기 (16개 → 1개 통합, CLI 인자 기반) |

---

## 7. 시나리오 구조 개편

### 기존 (A-1/A-2/B)

```
A-1: KURE-v1 임베딩 + Gemma 3-4b 로컬
A-2: KoE5 임베딩 + LLaMA 로컬
B  : OpenAI API
```

### 변경 후 (임베딩_LLM 12개 조합)

```
임베딩 3개: KURE / KOE5 / SMALL
LLM    4개: GEMMA / QWEN / PHI / OPENAI

12개 조합:
  KURE_GEMMA  / KURE_QWEN  / KURE_PHI  / KURE_OPENAI
  KOE5_GEMMA  / KOE5_QWEN  / KOE5_PHI  / KOE5_OPENAI
  SMALL_GEMMA / SMALL_QWEN / SMALL_PHI / SMALL_OPENAI
```

### 코드 변경 내역

```python
# EMBED_CONFIG 변경
{'A-1': KURE, 'A-2': KoE5, 'B': SMALL}
→ {'KURE': ..., 'KOE5': ..., 'SMALL': ...}

# _PROMPT_TEMPLATES 변경
{'A-1': Gemma포맷, 'A-2': LLaMA포맷, 'B': OpenAI포맷}
→ {'GEMMA': ..., 'QWEN': ..., 'PHI': ..., 'OPENAI': ...}

# get_generator() 변경
llm_key = scenario.split('_')[1]  # 'KURE_GEMMA' → 'GEMMA'
if llm_key == 'GEMMA': return LocalHFGenerator(scenario='GEMMA')
elif llm_key in ('QWEN', 'PHI'): return LocalHFGenerator(scenario=llm_key)
elif llm_key == 'OPENAI': return APIGenerator(scenario='OPENAI')
```

---

## 8. Generation 모듈 구축

### 8-1. 신규 작성 파일 5개 (5.24~5.25)

| 파일 | 역할 |
|---|---|
| `generation_interface.py` | LLM 생성 모듈 (LocalHFGenerator + APIGenerator, 팩토리 패턴) |
| `serving_main.py` | 통합 서빙 파이프라인 (듀얼 Retriever, Query Router, 히스토리 관리) |
| `fastapi_app.py` | FastAPI 엔드포인트 (SSE 스트리밍) |
| `eval_e2e_ctype.py` | C타입 E2E 비교 평가 |
| `eval_quant_judge.py` | LLM-as-a-Judge 정량 평가 |
| `eval_qual_analyzer.py` | 정성 평가 리포트 추출 |

### 8-2. Gemini 제공 코드 버그 7개 수정

| 버그 | 수정 |
|---|---|
| APIGenerator 시스템 프롬프트 누락 | 추가 |
| `apply_chat_template()` 이중 적용 | `_inject_history_into_prompt()`로 대체 |
| Iterator vs String 반환 타입 불일치 | 수정 |
| max_tokens → gpt-5-mini 미지원 | `max_completion_tokens`로 변경 |
| temperature=0.1 → gpt-5-mini 미지원 | temperature=1로 변경 |
| `__mro__[0].__dict__.get()` 패턴 | `from serving_main import _MAX_HISTORY_MSGS`로 수정 |
| C타입 분류기 `len(query) < 15` 오류 | 제거 |

### 8-3. 텍스트 생성 파라미터

```python
_LOCAL_GEN_PARAMS = {
    'max_new_tokens'    : 512,
    'temperature'       : 0.1,   # 입찰 공고 수치 정확도 최우선
    'top_p'             : 0.9,
    'do_sample'         : True,
    'repetition_penalty': 1.1,   # 로컬 모델 반복 방지
}

_API_GEN_PARAMS = {
    'max_completion_tokens': 512,  # gpt-5-mini 전용
    'temperature'          : 1,    # gpt-5-mini 기본값만 지원
}
```

---

## 9. E2E 평가 파이프라인 구조

### 9-1. Context 사전 추출 도입

**기존**: E2E 생성 시마다 `get_context()` 호출 → Retrieval 반복 실행  
**변경**: `extract_contexts.py`로 context 1회 추출 → CSV 저장 → E2E에서 재사용

```bash
BIDMATE_ENV=gcp python extract_contexts.py --embed KURE
→ eval_contexts_abde_kure.csv  (A/B/D/E 516개)
→ eval_contexts_c_kure.csv     (C타입 63개)
```

### 9-2. 파일명 규칙 확정

```
eval_results/generation/
├── eval_contexts_abde_{embed}.csv       ← 임베딩별 ABDE context
├── eval_contexts_c_{embed}.csv          ← 임베딩별 C타입 context
├── e2e_abde_{embed}_mid_{scenario}.csv  ← ABDE 중간파일 (재시작용)
├── e2e_ctype_{embed}_mid_{scenario}.csv ← C타입 중간파일
├── e2e_all_comparison.csv               ← 최종 합본 (Judge 입력)
├── quantitative_scores_all.csv          ← Judge 채점 결과
└── judge1_checkpoint/                   ← 체크포인트 폴더
```

---

## 10. 모델 스택 확정 및 업그레이드

### 10-1. LLM 확정 (4개)

| 키 | 모델 | 아키텍처 | VRAM | 비고 |
|---|---|---|---|---|
| GEMMA | google/gemma-4-E4B-it | **AutoModelForImageTextToText** | ~8GB | 멀티모달, gemma-3→4 업그레이드 |
| QWEN | Qwen/Qwen2.5-7B-Instruct | **AutoModelForCausalLM** | ~14GB | 7B 파라미터 장문 이해 |
| PHI | microsoft/Phi-4-mini-instruct | **AutoModelForCausalLM** | ~8GB | 경량 4B, 빠른 추론 |
| OPENAI | gpt-5-mini | API | — | max_completion_tokens |

### ⚠️ 모델 클래스 분기 (중요)

```python
# GEMMA만 멀티모달 → ImageTextToText
# QWEN/PHI는 순수 텍스트 → CausalLM
if scenario == 'GEMMA':
    AutoModelForImageTextToText.from_pretrained(...)
else:  # QWEN, PHI
    AutoModelForCausalLM.from_pretrained(...)
```

### 10-2. 임베딩 (3개)

| 키 | 모델 | 비고 |
|---|---|---|
| KURE | nlpai-lab/KURE-v1 | 한국어 특화, 주력 |
| KOE5 | nlpai-lab/KoE5 | 비교용 |
| SMALL | text-embedding-3-small (OpenAI API) | API 기반 |

### 10-3. QWEN/PHI LocalHF 전환

기존: OpenRouterGenerator (외부 API)  
변경: LocalHFGenerator (GCP 로컬 실행) → 파인튜닝 후 어댑터 로드 가능

### 10-4. LoRA 어댑터 로드 분기

```python
# 파인튜닝 완료 후 어댑터 경로 지정
export GEMMA_ADAPTER_PATH=/home/euijeong/.../lora_adapter   # PeftModel 로드
export QWEN_ADAPTER_PATH=/home/euijeong/.../merged_model    # merge_and_unload() 완료 모델
```

---

## 11. E2E 평가 전체 실행 완료

### 11-1. 주요 버그 수정 이력 (v10 이후)

| 파일 | 버그 | 수정 |
|---|---|---|
| `generation_interface.py` | PHI/QWEN에 AutoModelForImageTextToText 적용 | GEMMA/QWEN/PHI 분기 추가 |
| `retrieval_interface_F.py` | NaN context → `AttributeError: 'float' has no attribute 'split'` | `isinstance(context, str)` 체크 추가 |
| `retrieval_interface_F.py` | SMALL 분기 `scenario == 'B'` → SMALL 미인식 | `scenario in ('B', 'SMALL')`로 수정 |
| `eval_e2e_abde.py` | `_SAVE_PATH` 고정으로 시나리오 바뀔 때마다 덮어쓰기 | `_SAVE_PATH_TPL` 동적 경로 |

### 11-2. ABDE (516개) 완료 현황

| 임베딩 | GEMMA | PHI | QWEN | OPENAI |
|---|---|---|---|---|
| KURE | ✅ | ✅ | ✅ | ✅ |
| KOE5 | ✅ | ✅ | ✅ | ✅ |
| SMALL | ✅ | ✅ (OOM 후 재실행) | ✅ | ✅ |

### 11-3. C타입 (63개) 완료 현황

| 임베딩 | GEMMA | PHI | QWEN | OPENAI |
|---|---|---|---|---|
| KURE | ✅ | ✅ | ✅ | ✅ |
| KOE5 | ✅ | ✅ | ✅ | ✅ |
| SMALL | ✅ | ✅ (OOM 후 재실행) | ✅ | ✅ |

### 11-4. 생성 속도 비교

| 모델 | 속도 | 516개 소요시간 |
|---|---|---|
| GEMMA (gemma-4-E4B-it) | 37~52초/개 | 약 5시간 |
| QWEN (Qwen2.5-7B-Instruct) | 6~7초/개 | 약 55분 |
| PHI (Phi-4-mini-instruct) | 1.5~3초/개 | 약 25분 |
| OPENAI (gpt-5-mini) | 4~8초/개 | 약 40분 |

### 11-5. e2e_all_comparison.csv 최종 현황

```
경로: /home/euijeong/2Team_Project/hej/eval_results/generation/e2e_all_comparison.csv
579개 (A:172 / B:214 / C:63 / D:65 / E:65)
12개 답변 컬럼:
  ans_kure_gemma  / ans_kure_phi  / ans_kure_qwen  / ans_kure_openai
  ans_koe5_gemma  / ans_koe5_phi  / ans_koe5_qwen  / ans_koe5_openai
  ans_small_gemma / ans_small_phi / ans_small_qwen / ans_small_openai
```

**합치기 방식**: 중간파일 기준 직접 병합  
```
ans_kure_openai  = e2e_abde_kure_mid_SMALL_OPENAI.csv  (KURE 임베딩 검색 context 기반)
ans_koe5_openai  = e2e_abde_koe5_mid_SMALL_OPENAI.csv
ans_small_openai = e2e_abde_small_mid_SMALL_OPENAI.csv
```

**잔존 오류 (max_tokens 초과)**:
```
ans_kure_openai  : 6건
ans_koe5_openai  : 4건
ans_small_openai : 3건
→ 총 13/579 = 2.2%, Judge 채점 시 NaN 처리
```

---

## 12. Judge 채점 현황

### 12-1. eval_quant_judge1_all.py (신규)

- 기존 `eval_quant_judge_dual.py` (2개 시나리오 비교) → 12개 시나리오 독립 채점으로 새로 작성
- Judge: gpt-5-mini 단독 (OpenRouter gemma-4-26b는 일 200 요청 제한으로 비현실적)
- temperature=1 (gpt-5-mini 기본값만 지원)
- 100행마다 체크포인트 저장 (`judge1_checkpoint/checkpoint.csv`)

### 12-2. 6개 평가 지표

| 지표 | 설명 | GT 필요 |
|---|---|---|
| Faithfulness | 환각 탐지 (Context 근거 여부) | ❌ |
| Answer Relevance | 질문 의도 부합 여부 | ❌ |
| Rejection | 답 없을 때 거절 적절성 | ❌ |
| Correctness | GT 팩트 일치 | ✅ |
| Context Precision | 검색 청크 관련성 (RAGAS) | ✅ |
| Context Recall | GT 정보 포함 여부 (RAGAS) | ✅ |

### 12-3. 실행 현황

```
GCP:   eval_quant_judge1_all.py    — 579행 전체, 약 4.8시간 예상
Colab: eval_quant_judge1_colab.py  — 215행 샘플링 (A60/B75/C30/D25/E25), 약 1.8시간
```

### 12-4. 출력 컬럼 형식

```
{임베딩}_{llm}_avg_{지표}
예: kure_gemma_avg_faithfulness, koe5_phi_avg_relevance
```

---

## 13. 파일 구성 최종 현황

```
2Team_Project/hej/
├── retrieval_interface_F.py          # Retrieval 핵심
├── retrieval_eval_F.py               # 평가 전용 Retriever
├── run_eval_F.py                     # 배치 평가 실행기
├── generation_interface.py           # LLM 생성 모듈 (LoRA 어댑터 분기)
├── serving_main.py                   # 통합 서빙 파이프라인
├── fastapi_app.py                    # FastAPI 엔드포인트 (SSE)
├── extract_contexts.py               # Context 사전 추출
├── eval_e2e_abde.py                  # A/B/D/E E2E 평가 (동적 경로)
├── eval_e2e_ctype.py                 # C타입 E2E 평가
├── eval_quant_judge.py               # 단일 Judge
├── eval_quant_judge_dual.py          # 듀얼 Judge (2시나리오 비교)
├── eval_quant_judge1_all.py          # ★ 12시나리오 독립 채점 (gpt-5-mini)
├── eval_quant_judge1_colab.py        # ★ Colab 버전 (샘플링 215행)
├── eval_quant_judge1_all_sampled.py  # ★ 샘플링 버전 (GCP용)
├── eval_qual_analyzer.py             # 정성 평가 리포트
├── check_release_gate.py             # ★ PASS/GOOD/FAIL Gate 판정
└── eval_results/generation/
    ├── eval_contexts_abde_{embed}.csv      (KURE/KOE5/SMALL)
    ├── eval_contexts_c_{embed}.csv         (KURE/KOE5/SMALL)
    ├── e2e_abde_{embed}_mid_{scenario}.csv (중간파일 12개 완료)
    ├── e2e_ctype_{embed}_mid_{scenario}.csv (중간파일 12개 완료)
    ├── e2e_all_comparison.csv              ✅ 완료 (579개 × 12컬럼)
    ├── quantitative_scores_all.csv         🔄 생성 중 (GCP)
    ├── quantitative_scores_all_sampled.csv 🔄 생성 중 (Colab)
    └── judge1_checkpoint/
```

---

## 14. GCP 환경

### 서버 정보

```
서버: codeit-ai-02
GPU : NVIDIA L4 24GB
CUDA: 13.0
OS  : Ubuntu 24
Python: 3.12
```

### 매 세션 설정

```bash
source /home/euijeong/2Team_Project/serve_env/bin/activate
cd /home/euijeong/2Team_Project/hej
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/jhub-venv/lib/python3.12/site-packages/nvidia/cu13/lib
export BIDMATE_ENV=gcp
export OPENAI_API_KEY=팀키
```

### 가상환경 공유

```
경로: /home/euijeong/2Team_Project/serve_env
권한: chmod -R 755 완료 → 타 팀원 사용 가능
```

### 디스크 현황

```
전체 96GB / 현재 94GB 사용 (2GB 여유)
HuggingFace 캐시:
  gemma-4-E4B-it      15GB  (삭제 예정 — 파인튜닝 후 재다운로드)
  Qwen2.5-7B-Instruct 15GB  (삭제 예정)
  Phi-4-mini           7.2GB (삭제 예정)
  bge-reranker-v2-m3   2.2GB (서빙 필수 유지)
  KURE-v1              2.2GB (서빙 필수 유지)
  KoE5                 2.2GB (서빙 필수 유지)
```

---

## 15. Release Gate 기준

### Retrieval

| 지표 | PASS | GOOD |
|---|---|---|
| Hit@5 | ≥ 0.90 | ≥ 0.95 |
| MRR | ≥ 0.82 | ≥ 0.87 |
| nDCG | ≥ 0.78 | ≥ 0.83 |

**타입별 MRR 기준:**

| 타입 | PASS | GOOD |
|---|---|---|
| A | ≥ 0.92 | ≥ 0.95 |
| B | ≥ 0.77 | ≥ 0.82 |
| C | ≥ 0.88 | ≥ 0.93 |
| D | ≥ 0.81 | ≥ 0.86 |
| E | ≥ 0.82 | ≥ 0.87 |

### Generation

| 지표 | PASS | GOOD |
|---|---|---|
| Faithfulness / Relevance / Rejection / Context Precision / Context Recall | ≥ 3.5 | ≥ 4.0 |

---

## 16. 현재 진행 상태 및 다음 순서

### 현재 상태

```
✅ 완료
  - 579개 유니크 질문 기반 Retrieval 평가 (chunks_all 채택)
  - 12개 시나리오 × ABDE 516개 생성 완료
  - 12개 시나리오 × C타입 63개 생성 완료
  - e2e_all_comparison.csv (579개 × 12컬럼)

🔄 진행 중
  - GCP: eval_quant_judge1_all.py (579행 전체, 약 4.8시간)
  - Colab: eval_quant_judge1_colab.py (215행 샘플링, 약 1.8시간)

⬜ 대기 중
  - check_release_gate.py 실행
  - eval_qual_analyzer.py 실행
  - 파인튜닝 (팀장 peft 노트북 준비됨, Judge 결과 확인 후)
```

### 파인튜닝 계획

| 파일 | 모델 | 환경 | 방식 |
|---|---|---|---|
| `peft_qwen25_7B_A100.ipynb` | Qwen2.5-7B-Instruct | Colab A100 | LoRA + merge_and_unload() |
| `peft_gemma4_E4B_v2.ipynb` | gemma-4-E4B-it | Colab L4 | LoRA 어댑터만 저장 |

**학습 데이터 포맷:**
```python
{
    'instruction': row['question'],
    'input'      : f"[참고 문서]\n{row['retrieved_context']}",
    'output'     : row['ans_small_openai']  # GPT 답변을 정답 레이블로
}
```
※ eval 579개 재사용 (leakage 감수, 발표 시 명시 예정)

### 다음 실행 순서

```bash
# 1. Judge 완료 확인
tail -5 ~/2Team_Project/hej/judge1_all_run.log

# 2. Release Gate 체크
python check_release_gate.py

# 3. 정성 평가
export SCORE_PREFIX=avg
python eval_qual_analyzer.py

# 4. 파인튜닝 (Colab)
# Gemma: peft_gemma4_E4B_v2.ipynb
# Qwen : peft_qwen25_7B_A100.ipynb

# 5. 파인튜닝 후 E2E 재평가
export GEMMA_ADAPTER_PATH=/home/euijeong/.../lora_adapter
export QWEN_ADAPTER_PATH=/home/euijeong/.../merged_model
```
