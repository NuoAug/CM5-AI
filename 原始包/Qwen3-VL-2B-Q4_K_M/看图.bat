@echo off
chcp 65001 >nul
title Qwen3-VL-2B Q4_K_M 看图
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0see.ps1"
if errorlevel 1 pause
