@echo off
chcp 65001 >nul
title Qwen3-VL-2B Q4_K_M 状态
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0status.ps1"