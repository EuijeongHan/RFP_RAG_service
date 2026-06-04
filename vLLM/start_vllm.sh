#!/bin/bash

echo "기존 프로세스 종료 중..."
pkill -f "fastapi_server"
pkill -f "web_rag_admin"
pkill -f "web_rag_mobile"
pkill -f "vllm"
sleep 3

# vLLM 설치 확인
if ! python3 -c "import vllm" 2>/dev/null; then
    echo "vLLM 설치 중..."
    pip install vllm --break-system-packages -q
fi

# vLLM 서버 실행 (포트 8100)
echo "vLLM 서버 시작 (포트 8100)..."
echo "  모델: google/gemma-4-E4B-it"
echo "  LoRA: /mnt/gukrul/dataset/peft_output/gemma4-E4B_v2/lora_adapter_v2"

nohup python3 -m vllm.entrypoints.openai.api_server \
    --model google/gemma-4-E4B-it \
    --download-dir /mnt/gukrul/hf_cache/hub \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --port 8100 \
    --enable-lora \
    --lora-modules bidmate=/mnt/gukrul/dataset/peft_output/gemma4-E4B_v2/lora_adapter_v2 \
    > /mnt/gukrul/vllm_server.log 2>&1 &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# vLLM 준비 대기
echo "vLLM 초기화 대기 중..."
for i in $(seq 1 120); do
    LAST_LOG=$(tail -1 /mnt/gukrul/vllm_server.log 2>/dev/null)
    echo "  [$i/120] $LAST_LOG"

    if curl -s http://localhost:8100/health 2>/dev/null | grep -q "{}"; then
        echo "vLLM 서버 준비 완료"
        break
    fi

    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "vLLM 비정상 종료. 로그 확인:"
        tail -20 /mnt/gukrul/vllm_server.log
        exit 1
    fi
    sleep 5
done

# FastAPI 서버 실행
echo "FastAPI 서버 시작 (포트 2026)..."
nohup python /mnt/gukrul/main_py/fastapi_server.py > /mnt/gukrul/api_server.log 2>&1 &

# FastAPI 준비 대기
for i in $(seq 1 30); do
    if curl -s http://localhost:2026/health | grep -q '"service_ready":true'; then
        echo "FastAPI 서버 준비 완료"
        break
    fi
    sleep 3
done

# Admin, Mobile 실행
echo "Admin UI 시작 (포트 8443)..."
nohup streamlit run /mnt/gukrul/main_py/web_rag_admin.py --server.port 8443 > /mnt/gukrul/admin_server.log 2>&1 &
sleep 2

echo "Mobile UI 시작 (포트 8480)..."
nohup streamlit run /mnt/gukrul/main_py/web_rag_mobile.py --server.port 8480 > /mnt/gukrul/mobile_server.log 2>&1 &
sleep 2

echo "완료"
echo "  vLLM  : http://localhost:8100"
echo "  Admin : http://$(hostname -I | awk '{print $1}'):8443"
echo "  Mobile: http://$(hostname -I | awk '{print $1}'):8480"
