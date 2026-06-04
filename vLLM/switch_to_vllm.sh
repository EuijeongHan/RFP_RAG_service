#!/bin/bash
echo "=== vLLM 전환 시작 ==="
bash /mnt/gukrul/stop.sh
pkill -f "vllm" 2>/dev/null
sleep 2

python3 /mnt/gukrul/switch_vllm.py
echo ""
echo "전환 완료. 서비스 시작 중..."
bash /mnt/gukrul/start_vllm.sh
