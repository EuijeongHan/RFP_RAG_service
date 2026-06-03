#!/bin/bash

echo "서비스 종료 중..."
pkill -f "fastapi_server"
pkill -f "web_rag_admin"
pkill -f "web_rag_mobile"
sleep 2

# 종료 확인
remaining=$(ps aux | grep -E "fastapi_server|web_rag_admin|web_rag_mobile" | grep -v grep | wc -l)
if [ "$remaining" -eq 0 ]; then
    echo "모든 서비스 종료 완료"
else
    echo "강제 종료 중..."
    pkill -9 -f "fastapi_server"
    pkill -9 -f "web_rag_admin"
    pkill -9 -f "web_rag_mobile"
    echo "강제 종료 완료"
fi

nvidia-smi | grep -E "MiB|Process"
