#!/bin/bash
echo "=== Gemma4 E4B 전환 시작 ==="
bash /mnt/gukrul/stop.sh
python3 /mnt/gukrul/switch_gemma4.py
echo ""
echo "전환 완료. 서비스 시작 중..."
bash /mnt/gukrul/start.sh
