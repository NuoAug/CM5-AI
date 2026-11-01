@echo off
chcp 65001 >nul
title Qwen3-VL-2B Q4_K_M 一键部署
cd /d "%~dp0"
echo.
echo  即将检查环境、下载运行时与模型（首次约 4.9 GB），然后启动服务。
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0deploy.ps1"
if errorlevel 1 pause