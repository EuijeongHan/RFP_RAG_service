#!/bin/bash

# 기존 프로세스 종료
echo "기존 프로세스 종료 중..."
pkill -f "fastapi_server"
pkill -f "web_rag_admin"
pkill -f "web_rag_mobile"
sleep 3

# FastAPI 서버 실행
echo "FastAPI 서버 시작 (포트 2026)..."
nohup python /mnt/gukrul/main_py/fastapi_server.py > /mnt/gukrul/api_server.log 2>&1 &

# 서버 준비 대기
echo "서버 초기화 대기 중..."
for i in $(seq 1 60); do
    if curl -s http://localhost:2026/health | grep -q '"service_ready":true'; then
        echo "FastAPI 서버 준비 완료"
        break
    fi
    echo "  대기 중... ($i/60)"
    sleep 5
done

# Admin 실행
echo "Admin UI 시작 (포트 8443)..."
nohup streamlit run /mnt/gukrul/main_py/web_rag_admin.py --server.port 8443 > /mnt/gukrul/admin_server.log 2>&1 &
sleep 2

# Mobile 실행
echo "Mobile UI 시작 (포트 8480)..."
nohup streamlit run /mnt/gukrul/main_py/web_rag_mobile.py --server.port 8480 > /mnt/gukrul/mobile_server.log 2>&1 &
sleep 2

echo "완료"
echo "  Admin  : http://$(hostname -I | awk '{print $1}'):8443"
echo "  Mobile : http://$(hostname -I | awk '{print $1}'):8480"
