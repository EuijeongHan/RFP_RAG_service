#!/bin/bash

echo "기존 프로세스 종료 중..."
pkill -f "fastapi_server"
pkill -f "web_rag_admin"
pkill -f "web_rag_mobile"
sleep 3

echo "FastAPI 서버 시작 (포트 2026)..."
nohup python /mnt/gukrul/main_py/fastapi_server.py > /mnt/gukrul/api_server.log 2>&1 &
API_PID=$!
echo "FastAPI PID: $API_PID"

echo "서버 초기화 대기 중... (로그: /mnt/gukrul/api_server.log)"
for i in $(seq 1 120); do
    # 로그 마지막 줄 출력
    LAST_LOG=$(tail -1 /mnt/gukrul/api_server.log 2>/dev/null)
    echo "  [$i/120] $LAST_LOG"

    if curl -s http://localhost:2026/health | grep -q '"service_ready":true'; then
        echo "FastAPI 서버 준비 완료"
        break
    fi

    # 프로세스 죽었는지 확인
    if ! kill -0 $API_PID 2>/dev/null; then
        echo "FastAPI 서버 비정상 종료. 로그 확인:"
        tail -20 /mnt/gukrul/api_server.log
        exit 1
    fi

    sleep 5
done

echo "Admin UI 시작 (포트 8443)..."
nohup streamlit run /mnt/gukrul/main_py/web_rag_admin.py --server.port 8443 > /mnt/gukrul/admin_server.log 2>&1 &
sleep 2

echo "Mobile UI 시작 (포트 8480)..."
nohup streamlit run /mnt/gukrul/main_py/web_rag_mobile.py --server.port 8480 > /mnt/gukrul/mobile_server.log 2>&1 &
sleep 2

echo "완료"
echo "  Admin  : http://$(hostname -I | awk '{print $1}'):8443"
echo "  Mobile : http://$(hostname -I | awk '{print $1}'):8480"
