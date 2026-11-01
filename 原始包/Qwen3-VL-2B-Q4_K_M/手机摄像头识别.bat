@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "CAM_PAGE=phone.html"
set "CAM_PORT=8095"

set "NODE="
if exist "%USERPROFILE%\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\node\bin\node.exe" set "NODE=%USERPROFILE%\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\node\bin\node.exe"
if not defined NODE if exist "%ProgramFiles%\nodejs\node.exe" set "NODE=%ProgramFiles%\nodejs\node.exe"
if not defined NODE for /f "delims=" %%i in ('where node 2^>nul') do if not defined NODE set "NODE=%%i"
if not defined NODE goto nonode

echo Starting PHONE camera mode ... [Qwen3-VL-2B Q4_K_M]
echo   page  : http://127.0.0.1:8095/phone.html
echo   model : http://127.0.0.1:8091/v1
echo.
echo   Put the phone snapshot URL into the box, e.g. http://192.168.1.5:8080/shot.jpg
echo.
"%NODE%" "%~dp0camera\server.mjs"
echo.
echo Phone camera page stopped.
pause
exit /b 0

:nonode
echo [X] node.exe not found.
echo     Install Node.js, or check this path:
echo     %USERPROFILE%\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\node\bin\node.exe
echo.
pause
exit /b 1