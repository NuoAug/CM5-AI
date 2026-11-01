@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "NODE="
if exist "%USERPROFILE%\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\node\bin\node.exe" set "NODE=%USERPROFILE%\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\node\bin\node.exe"
if not defined NODE if exist "%ProgramFiles%\nodejs\node.exe" set "NODE=%ProgramFiles%\nodejs\node.exe"
if not defined NODE for /f "delims=" %%i in ('where node 2^>nul') do if not defined NODE set "NODE=%%i"
if not defined NODE goto nonode

echo Starting camera page ...  [Qwen3-VL-2B Q4_K_M]
echo   page  : http://127.0.0.1:8093/
echo   model : http://127.0.0.1:8091/v1
echo.
"%NODE%" "%~dp0camera\server.mjs"
echo.
echo Camera page stopped.
pause
exit /b 0

:nonode
echo [X] node.exe not found.
echo     Install Node.js, or check this path:
echo     %USERPROFILE%\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\node\bin\node.exe
echo.
pause
exit /b 1