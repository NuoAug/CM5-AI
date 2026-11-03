@echo off
cd /d "%~dp0"
if "%~1"=="" (echo Usage: 启动测试台.bat WEB_PORT VLM_PORT & exit /b 2)
if "%~2"=="" (echo Both explicit ports are required & exit /b 2)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0app\start.ps1" -WebPort %1 -VlmPort %2
