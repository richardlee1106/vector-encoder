@echo off
cd /d %~dp0
echo [Vector Encoder] service: http://127.0.0.1:8100
python run.py serve --port 8100
