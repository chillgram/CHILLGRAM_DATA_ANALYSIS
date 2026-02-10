#!/bin/bash
redis-server --daemonize yes
echo "[INFO] Redis 서버 시작됨"
uvicorn main:app --host 0.0.0.0 --port 8080
