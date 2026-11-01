@echo off
chcp 65001 >nul
title Qwen3-VL-2B Q4_K_M 停止
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop.ps1"