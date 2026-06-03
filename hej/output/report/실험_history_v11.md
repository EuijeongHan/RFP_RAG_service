# 입찰메이트 RAG — 실험 히스토리 v11

**작성일** : 2026-05-28  
**담당** : 한의정 (Retrieval + Generation)  
**목적** : v10 이후 E2E 전체 실행 완료 및 Judge 채점 현황 기록

---

## 목차

1. [v10 이후 주요 변경 사항 요약](#1-v10-이후-주요-변경-사항-요약)
2. [모델 스택 확정](#2-모델-스택-확정)
3. [코드 버그 수정 이력](#3-코드-버그-수정-이력)
4. [E2E 평가 전체 실행 완료](#4-e2e-평가-전체-실행-완료)
5. [e2e_all_comparison.csv 합치기](#5-e2e_all_comparisoncsv-합치기)
6. [Judge 채점 현황](#6-judge-채점-현황)
7. [파일 구성 최종 현황](#7-파일-구성-최종-현황)
8. [파인튜닝 노트북 현황](#8-파인튜닝-노트북-현황)
9. [GCP 환경 현황](#9-gcp-환경-현황)
10. [현재 진행 상태](#10-현재-진행-상태)
11. [다음 실행 순서](#11-다음-실행-순서)

---

## 1. v10 이후 주요 변경 사항 요약

| 항목 | v10 상태 | v11 결과 |
|---|---|---|
| ABDE E2E (KURE_GEMMA) | 🔄 실행 중 | ✅ 완료 |
| ABDE E2E (12개 전 시나리오) | ⬜ | ✅ 완료 |
| C타입 E2E (12개 전 시나리오) | ⬜ | ✅ 완료 |
| e2e_all_comparison.csv | ⬜ | ✅ 완료 (579개, 12컬럼) |
| Judge 채점 (gpt-5-mini 단독) | ⬜ | 🔄 실행 중 |
| GEMMA → gemma-4-E4B-it 업그레이드 | gemma-3-4b-it | ✅ 변경 완료 |
| QWEN → Qwen2.5-7B-Instruct | Qwen3.5-4B | ✅ 변경 완료 |
| QWEN/PHI → LocalHFGenerator 전환 | OpenRouterGenerator | ✅ 변경 완료 |
| LoRA 어댑터 로드 분기 추가 | ⬜ | ✅ 추가 완료 |
| AutoModelForCausalLM 분기 | GEMMA만 ImageTextToText | ✅ GEMMA/QWEN/PHI 분기 추가 |
| eval_quant_judge1_all.py | ⬜ | ✅ 신규 작성 (12시나리오 독립 채점) |
| check_release_gate.py | ⬜ | ✅ 신규 작성 |

---

## 2. 모델 스택 확정

### 임베딩 (3개)
| 시나리오 키 | 모델 | 컬렉션 |
|---|---|---|
| KURE | nlpai-lab/KURE-v1 | bidmate_kure |
| KOE5 | nlpai-lab/KoE5 | bidmate_chunks_all_A-2 (구버전명 유지) |
| SMALL | text-embedding-3-small (OpenAI API) | bidmate_chunks_all_B |

### LLM (4개)
| 시나리오 키 | 모델 | 아키텍처 | 비고 |
|---|---|---|---|
| GEMMA | google/gemma-4-E4B-it | AutoModelForImageTextToText | bfloat16 ~8GB |
| QWEN | Qwen/Qwen2.5-7B-Instruct | AutoModelForCausalLM | bfloat16 ~14GB |
| PHI | microsoft/Phi-4-mini-instruct | AutoModelForCausalLM | bfloat16 ~8GB |
| OPENAI | gpt-5-mini (OpenAI API) | — | max_completion_tokens |

### ⚠️ 모델 클래스 분기 (중요)
```python
# GEMMA만 멀티모달 아키텍처
if scenario == 'GEMMA':
    AutoModelForImageTextToText.from_pretrained(...)
else:  # QWEN, PHI
    AutoModelForCausalLM.from_pretrained(...)
```

### 12개 시나리오 조합
```
KURE_GEMMA  / KURE_PHI  / KURE_QWEN  / KURE_OPENAI
KOE5_GEMMA  / KOE5_PHI  / KOE5_QWEN  / KOE5_OPENAI
SMALL_GEMMA / SMALL_PHI / SMALL_QWEN / SMALL_OPENAI
```

### ChromaDB 컬렉션 현황
```
C타입용 (kh_v3_hybrid):
  KURE → bidmate_kh_v3_A-1
  KOE5 → bidmate_kh_v3_A-2
  SMALL → bidmate_kh_v3_B

ABDE용 (chunks_all):
  KURE  → bidmate_kure
  KOE5  → bidmate_chunks_all_A-2
  SMALL → bidmate_chunks_all_B
```

---

## 3. 코드 버그 수정 이력 (v10 이후)

| 파일 | 버그 | 수정 |
|---|---|---|
| `generation_interface.py` | PHI/QWEN에 `AutoModelForImageTextToText` 적용 → ValueError | GEMMA만 ImageTextToText, QWEN/PHI는 AutoModelForCausalLM으로 분기 |
| `retrieval_interface_F.py` | `_compress_context`에서 NaN context → `AttributeError: 'float' has no attribute 'split'` | `isinstance(context, str)` 체크 추가 |
| `retrieval_interface_F.py` | SMALL 임베딩 분기 `scenario == 'B'` → SMALL 미인식 | `scenario in ('B', 'SMALL')`로 수정 |
| `eval_e2e_abde.py` | `_SAVE_PATH` 고정으로 시나리오 바꿀 때마다 덮어쓰기 | `_SAVE_PATH_TPL` 동적 경로 (embed + scenario_a 포함) |
| `generation_interface.py` | _A2_MODEL_ID `Qwen3.5-4B` | `Qwen2.5-7B-Instruct`로 변경 |

---

## 4. E2E 평가 전체 실행 완료

### ABDE (516개) 완료 현황

| 임베딩 | GEMMA | PHI | QWEN | OPENAI |
|---|---|---|---|---|
| KURE | ✅ | ✅ | ✅ | ✅ (중간파일 재사용) |
| KOE5 | ✅ | ✅ | ✅ | ✅ (중간파일 재사용) |
| SMALL | ✅ | ✅ | ✅ | ✅ |

### C타입 (63개) 완료 현황

| 임베딩 | GEMMA | PHI | QWEN | OPENAI |
|---|---|---|---|---|
| KURE | ✅ | ✅ | ✅ | ✅ (중간파일 재사용) |
| KOE5 | ✅ | ✅ | ✅ | ✅ (중간파일 재사용) |
| SMALL | ✅ | ✅ (OOM 후 재실행) | ✅ | ✅ |

### 생성 오류 현황 (e2e_all_comparison.csv 기준)
```
ans_kure_openai  : 6건  (max_tokens 초과)
ans_koe5_openai  : 4건  (max_tokens 초과)
ans_small_openai : 3건  (max_tokens 초과)
```
→ 전부 max_completion_tokens 초과, Judge 채점 시 NaN 처리

### 생성 속도 비교
| 모델 | 속도 | 516개 소요시간 |
|---|---|---|
| GEMMA (gemma-4-E4B-it) | 37~52초/개 | 약 5시간 |
| QWEN (Qwen2.5-7B-Instruct) | 6~7초/개 | 약 55분 |
| PHI (Phi-4-mini-instruct) | 1.5~3초/개 | 약 25분 |
| OPENAI (gpt-5-mini) | 4~8초/개 | 약 40분 |

---

## 5. e2e_all_comparison.csv 합치기

### 최종 파일 구조
```
579개 (A:172 / B:214 / C:63 / D:65 / E:65)
컬럼: question, type, difficulty, history, retrieved_context
      + 12개 답변 컬럼:
        ans_kure_gemma, ans_kure_phi, ans_kure_qwen, ans_kure_openai
        ans_koe5_gemma, ans_koe5_phi, ans_koe5_qwen, ans_koe5_openai
        ans_small_gemma, ans_small_phi, ans_small_qwen, ans_small_openai
```

### 합치기 방식
- 중간파일(`e2e_abde_{embed}_mid_{scenario}.csv`) 기준으로 직접 병합
- KURE_OPENAI = `e2e_abde_kure_mid_SMALL_OPENAI.csv` (KURE 임베딩으로 검색한 context 기반)
- KOE5_OPENAI = `e2e_abde_koe5_mid_SMALL_OPENAI.csv`
- SMALL_OPENAI = `e2e_abde_small_mid_SMALL_OPENAI.csv`

---

## 6. Judge 채점 현황

### eval_quant_judge1_all.py (신규)
- 기존 `eval_quant_judge_dual.py` (2개 시나리오 비교) → 12개 시나리오 독립 채점으로 새로 작성
- Judge: gpt-5-mini 단독 (OpenRouter Judge2는 일 200 요청 제한으로 비현실적)
- 6개 지표: Faithfulness, Relevance, Rejection, Correctness, Context Precision, Context Recall
- 100행마다 체크포인트 저장 (`judge1_checkpoint/checkpoint.csv`)
- temperature=1 (gpt-5-mini 기본값만 지원)

### 실행 현황
```
GCP: eval_quant_judge1_all.py 실행 중 (579행 전체, 약 4.8시간 예상)
Colab: eval_quant_judge1_colab.py 실행 중 (215행 샘플링, 약 1.8시간 예상)
```

### 출력 컬럼 형식
```
{시나리오}_{지표}
예: kure_gemma_avg_faithfulness, koe5_phi_avg_relevance, ...
```

---

## 7. 파일 구성 최종 현황

```
2Team_Project/hej/
├── retrieval_interface_F.py         # Retrieval 핵심 (GEMMA/QWEN/PHI/OPENAI 템플릿, Prompt Compression, HyDE)
├── retrieval_eval_F.py              # Retrieval 평가 전용
├── run_eval_F.py                    # 배치 평가 실행
├── generation_interface.py          # LLM 생성 모듈 (LoRA 어댑터 분기 포함)
├── serving_main.py                  # 통합 서빙 파이프라인 (듀얼 Retriever, Query Router)
├── fastapi_app.py                   # FastAPI 엔드포인트 (SSE 스트리밍)
├── extract_contexts.py              # Context 사전 추출
├── eval_e2e_abde.py                 # A/B/D/E E2E 평가 (동적 저장 경로)
├── eval_e2e_ctype.py                # C타입 E2E 평가
├── eval_quant_judge.py              # 단일 Judge (OpenRouter gemma-4-26b, 2시나리오 비교)
├── eval_quant_judge_dual.py         # 듀얼 Judge (gpt-5-mini + gemma-4-26b, 2시나리오 비교)
├── eval_quant_judge1_all.py         # ★신규: 12시나리오 독립 채점 (gpt-5-mini 단독)
├── eval_quant_judge1_colab.py       # ★신규: eval_quant_judge1_all.py Colab 버전 (215행 샘플링)
├── eval_quant_judge1_all_sampled.py # ★신규: 샘플링 버전 (A60/B75/C30/D25/E25)
├── eval_qual_analyzer.py            # 정성 평가 리포트
├── check_release_gate.py            # ★신규: PASS/GOOD/FAIL 3단계 Gate 판정
└── eval_results/generation/
    ├── eval_contexts_abde_{embed}.csv     # 임베딩별 ABDE context
    ├── eval_contexts_c_{embed}.csv        # 임베딩별 C타입 context
    ├── e2e_abde_{embed}_mid_{scenario}.csv  # ABDE 중간파일 (12개)
    ├── e2e_ctype_{embed}_mid_{scenario}.csv # C타입 중간파일 (12개)
    ├── e2e_all_comparison.csv             # ★완료: 579개 × 12컬럼 합본
    ├── quantitative_scores_all.csv        # Judge 채점 결과 (GCP, 생성 중)
    ├── quantitative_scores_all_sampled.csv # Judge 채점 결과 (Colab 샘플링, 생성 중)
    └── judge1_checkpoint/                 # 체크포인트 폴더
```

---

## 8. 파인튜닝 노트북 현황

### 팀장이 준비한 노트북
| 파일 | 모델 | 환경 | 방식 | 상태 |
|---|---|---|---|---|
| `peft_qwen25_7B_A100.ipynb` | Qwen2.5-7B-Instruct | Colab A100 | LoRA + merge_and_unload() | 준비됨 |
| `peft_gemma4_E4B_v2.ipynb` | gemma-4-E4B-it | Colab L4 | LoRA 어댑터만 저장 (merge 없음) | 준비됨 |

### 파인튜닝 후 서빙 방식
```python
# Gemma: PeftModel로 어댑터 로드
export GEMMA_ADAPTER_PATH=/home/euijeong/.../peft_output/gemma4-E4B/lora_adapter

# Qwen: merge_and_unload() 완료된 병합 모델 직접 로드
export QWEN_ADAPTER_PATH=/home/euijeong/.../peft_output/qwen25-7B/merged_model
```

### 학습 데이터
```python
{
    'instruction': row['question'],
    'input'      : f"[참고 문서]\n{row['retrieved_context']}",
    'output'     : row['ans_small_openai']  # GPT 답변을 정답으로
}
```
- leakage 문제 감수 (발표 시 명시)
- 선행 조건: Judge 채점 완료 후 진행

---

## 9. GCP 환경 현황

### 서버 정보
```
서버: codeit-ai-02
GPU : NVIDIA L4 24GB
OS  : Ubuntu 24
```

### 가상환경
```bash
source /home/euijeong/2Team_Project/serve_env/bin/activate
# 타 팀원도 사용 가능 (chmod -R 755 완료)
```

### 매 세션 설정
```bash
source /home/euijeong/2Team_Project/serve_env/bin/activate
cd /home/euijeong/2Team_Project/hej
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/jhub-venv/lib/python3.12/site-packages/nvidia/cu13/lib
export BIDMATE_ENV=gcp
export OPENAI_API_KEY=팀키
```

### 디스크 현황
```
전체 96GB / 현재 94GB 사용 (2GB 여유)
HuggingFace 캐시:
  gemma-4-E4B-it      15GB
  Qwen2.5-7B-Instruct 15GB  ← E2E 완료, 삭제 예정
  Phi-4-mini           7.2GB ← E2E 완료, 삭제 예정
  bge-reranker-v2-m3   2.2GB
  KURE-v1              2.2GB
  KoE5                 2.2GB
```

### ChromaDB 컬렉션 현황
```python
['bidmate_chunks_all', 'bidmate_chunks_all_B', 'bidmate_retrieval_v1',
 'bidmate_kh_fixed_v1_B', 'bidmate_retrieval_openai_v1', 'bidmate_kh_v3_A-1',
 'bidmate_kh_v3_B', 'bidmate_kh_fixed_v1_A-2', 'bidmate_kh_fixed_v2_B',
 'bidmate_kh_fixed_v1_A-1', 'bidmate_chunks_all_A-1', 'bidmate_kure',
 'bidmate_kh_v3_A-2', 'bidmate_kh_fixed_v2_A-2', 'bidmate_retrieval_koe5_v1',
 'bidmate_kh_fixed_v2_A-1', 'bidmate_chunks_all_A-2']
```

---

## 10. 현재 진행 상태

```
✅ 완료
  - 12개 시나리오 × ABDE 516개 생성
  - 12개 시나리오 × C타입 63개 생성
  - e2e_all_comparison.csv 생성 (579행 × 12컬럼)

🔄 진행 중
  - GCP: eval_quant_judge1_all.py (579행 전체, 약 4.8시간)
  - Colab: eval_quant_judge1_colab.py (215행 샘플링, 약 1.8시간)

⬜ 대기 중
  - check_release_gate.py 실행
  - eval_qual_analyzer.py 실행
  - 파인튜닝 (Judge 결과 확인 후)
```

---

## 11. 다음 실행 순서

```bash
# 1. Judge 완료 확인
tail -5 ~/2Team_Project/hej/judge1_all_run.log

# 2. Release Gate 체크
python check_release_gate.py \
  --retrieval_csv eval_results/eval_results_chunks_all_KURE_GEMMA_579_v1.csv \
  --gen_csv eval_results/generation/quantitative_scores_all.csv

# 3. 정성 평가
export SCORE_PREFIX=avg
python eval_qual_analyzer.py

# 4. 파인튜닝 (Colab)
# Gemma: peft_gemma4_E4B_v2.ipynb (Colab L4)
# Qwen : peft_qwen25_7B_A100.ipynb (Colab A100)

# 5. 파인튜닝 후 어댑터 GCP 이전
export GEMMA_ADAPTER_PATH=/home/euijeong/.../lora_adapter
export QWEN_ADAPTER_PATH=/home/euijeong/.../merged_model

# 6. 파인튜닝 후 E2E 재평가
# (GEMMA/QWEN만 재실행, 나머지는 기존 결과 재사용)
```

---

## Release Gate 기준

### Retrieval
| 지표 | PASS | GOOD |
|---|---|---|
| Hit@5 | ≥ 0.90 | ≥ 0.95 |
| MRR | ≥ 0.82 | ≥ 0.87 |
| nDCG | ≥ 0.78 | ≥ 0.83 |

### 타입별 MRR
| 타입 | PASS | GOOD |
|---|---|---|
| A (추출 정밀도) | ≥ 0.92 | ≥ 0.95 |
| B (종합 능력) | ≥ 0.77 | ≥ 0.82 |
| C (맥락 유지) | ≥ 0.88 | ≥ 0.93 |
| D (환각 방지) | ≥ 0.81 | ≥ 0.86 |
| E (오타 질문) | ≥ 0.82 | ≥ 0.87 |

### Generation
| 지표 | PASS | GOOD |
|---|---|---|
| Faithfulness / Relevance / Rejection / Context Precision / Context Recall | ≥ 3.5 | ≥ 4.0 |
