# 启动 VLM（llama-server）与本地测试网页
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root "app\.venv\Scripts\python.exe"
$vlm = Join-Path $root "runtime\llama.cpp\llama-server.exe"
if (-not (Test-Path $vlm)) {
    $vlm = Join-Path $root "runtime\llama.cpp-src\build4\bin\llama-server.exe"
}
$model = Join-Path $root "models\vlm\minicpm-v-4.7-1b\MiniCPM-V-4_7-1B-Q4_K_M.gguf"
$mmproj = Join-Path $root "models\vlm\minicpm-v-4.7-1b\mmproj-model-f16.gguf"
$logs = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null

foreach ($p in @($vlm, $model, $mmproj, $python)) {
    if (-not (Test-Path $p)) { throw "缺少文件：$p" }
}

$vlmArgs = @(
    "-m", $model,
    "--mmproj", $mmproj,
    "-c", "1024",
    "-t", "8",
    "-ngl", "0",
    "--parallel", "1",
    "--host", "127.0.0.1",
    "--port", "8091"
)
$vlmProc = Start-Process -FilePath $vlm -ArgumentList $vlmArgs `
    -WorkingDirectory (Split-Path -Parent $vlm) `
    -RedirectStandardOutput (Join-Path $logs "vlm.log") `
    -RedirectStandardError (Join-Path $logs "vlm.err.log") `
    -WindowStyle Hidden -PassThru

Write-Host "VLM 启动中 (PID $($vlmProc.Id)) ..."
$ok = $false
for ($i = 0; $i -lt 90; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:8091/health" -TimeoutSec 2 -UseBasicParsing
        if ($r.StatusCode -eq 200) { $ok = $true; break }
    } catch { }
}
if (-not $ok) {
    Write-Host "VLM 未在 90 秒内就绪，请查看 logs\vlm.err.log" -ForegroundColor Yellow
}

$webProc = Start-Process -FilePath $python -ArgumentList (Join-Path $root "app\server.py") `
    -WorkingDirectory (Join-Path $root "app") `
    -RedirectStandardOutput (Join-Path $logs "web.log") `
    -RedirectStandardError (Join-Path $logs "web.err.log") `
    -WindowStyle Hidden -PassThru

@{ vlm = $vlmProc.Id; web = $webProc.Id } | ConvertTo-Json | Set-Content (Join-Path $logs "pids.json")
Start-Sleep -Seconds 3
Write-Host ""
Write-Host "测试页面： http://127.0.0.1:8090" -ForegroundColor Cyan
Write-Host "VLM 接口： http://127.0.0.1:8091" -ForegroundColor DarkGray
Write-Host "服务运行中；关闭本窗口或运行 app\stop.ps1 停止。" -ForegroundColor Yellow
Wait-Process -Id $vlmProc.Id, $webProc.Id
