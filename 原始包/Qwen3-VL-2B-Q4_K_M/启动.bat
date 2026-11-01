@echo off
chcp 65001 >nul
title Qwen3-VL-2B Q4_K_M 启动
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0deploy.ps1"
if errorlevel 1 pause