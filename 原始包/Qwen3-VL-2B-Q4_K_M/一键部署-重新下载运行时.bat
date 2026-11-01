@echo off
chcp 65001 >nul
title Qwen3-VL-2B Q4_K_M 从零复现（强制重下运行时）
cd /d "%~dp0"
echo.
echo  [注意] 这个是"从零复现"用的：会无视本机已有的 llama.cpp，额外下载约 560 MB。
echo  后面跟 一键部署.bat 做的完全一样，只是运行时换新的。
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0deploy.ps1" -Runtime download
if errorlevel 1 pause