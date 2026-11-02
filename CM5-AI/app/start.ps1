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

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$sessionDir = Join-Path $logs "session-$stamp"
New-Item -ItemType Directory -Force -Path $sessionDir | Out-Null
$eventLog = Join-Path $logs "sessions.log"
function Write-Event($msg) {
    Add-Content -Path $eventLog -Encoding UTF8 -Value ("{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg)
}

# 只保留最近 10 次会话日志
Get-ChildItem $logs -Directory -Filter "session-*" |
    Sort-Object Name -Descending |
    Select-Object -Skip 10 |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

foreach ($p in @($vlm, $model, $mmproj, $python)) {
    if (-not (Test-Path $p)) { throw "缺少文件：$p" }
}

$vlmThreads = if ($env:VLM_THREADS) { [int]$env:VLM_THREADS } elseif ([Environment]::ProcessorCount -le 4) { 4 } else { 8 }
$vlmCtx = if ($env:VLM_CTX) { [int]$env:VLM_CTX } else { 1024 }
$apiKey = if ($env:VLM_API_KEY) { $env:VLM_API_KEY } else {
    -join ((48..57 + 65..90 + 97..122) | Get-Random -Count 32 | ForEach-Object { [char]$_ })
}
$env:VLM_API_KEY = $apiKey
Set-Content -Path (Join-Path $logs "vlm.key") -Value $apiKey -NoNewline -Encoding ascii

$vlmArgs = @(
    "-m", $model,
    "--mmproj", $mmproj,
    "-c", "$vlmCtx",
    "-t", "$vlmThreads",
    "-ngl", "0",
    "--parallel", "1",
    "--host", "127.0.0.1",
    "--port", "8091",
    "--api-key", $apiKey,
    "--cors-origins", "http://127.0.0.1:8090"
)
$vlmLog = Join-Path $sessionDir "vlm.log"
$vlmErr = Join-Path $sessionDir "vlm.err.log"
$vlmProc = Start-Process -FilePath $vlm -ArgumentList $vlmArgs `
    -WorkingDirectory (Split-Path -Parent $vlm) `
    -RedirectStandardOutput $vlmLog `
    -RedirectStandardError $vlmErr `
    -WindowStyle Hidden -PassThru

# 先登记 VLM PID，并启动守护进程：主窗口关闭后自动结束模型与网页服务
@{ vlm = $vlmProc.Id; web = 0; session = $sessionDir; started = $stamp } | ConvertTo-Json | Set-Content (Join-Path $logs "pids.json")
Write-Event ("start vlm pid={0} threads={1} ctx={2} session={3}" -f $vlmProc.Id, $vlmThreads, $vlmCtx, $sessionDir)

$stopScript = Join-Path $PSScriptRoot "stop.ps1"
Start-Process -FilePath "powershell.exe" -ArgumentList @(
    "-NoProfile", "-WindowStyle", "Hidden", "-Command",
    "Wait-Process -Id $PID -ErrorAction SilentlyContinue; & '$stopScript' -Reason window_closed"
) -WindowStyle Hidden | Out-Null

Write-Host "VLM 启动中 (PID $($vlmProc.Id), threads $vlmThreads, ctx $vlmCtx) ..."
$ok = $false
$headers = @{ Authorization = "Bearer $apiKey" }
for ($i = 0; $i -lt 90; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:8091/health" -Headers $headers -TimeoutSec 2 -UseBasicParsing
        if ($r.StatusCode -eq 200) { $ok = $true; break }
    } catch { }
}
if (-not $ok) {
    Write-Host "VLM 未在 90 秒内就绪，请查看 $vlmErr" -ForegroundColor Yellow
}

$webLog = Join-Path $sessionDir "web.log"
$webErr = Join-Path $sessionDir "web.err.log"
$webProc = Start-Process -FilePath $python -ArgumentList (Join-Path $root "app\server.py") `
    -WorkingDirectory (Join-Path $root "app") `
    -RedirectStandardOutput $webLog `
    -RedirectStandardError $webErr `
    -WindowStyle Hidden -PassThru

@{ vlm = $vlmProc.Id; web = $webProc.Id; session = $sessionDir; started = $stamp } | ConvertTo-Json | Set-Content (Join-Path $logs "pids.json")
Write-Event ("start web pid={0} session={1}" -f $webProc.Id, $sessionDir)
Start-Sleep -Seconds 3
Write-Host ""
Write-Host "测试页面： http://127.0.0.1:8090" -ForegroundColor Cyan
Write-Host "VLM 接口： http://127.0.0.1:8091（需要 API key，见 logs\vlm.key）" -ForegroundColor DarkGray
Write-Host "本次日志： $sessionDir" -ForegroundColor DarkGray
Write-Host "服务运行中；关闭此窗口即停止模型与网页服务。" -ForegroundColor Yellow

while ($true) {
    Start-Sleep -Seconds 2
    if ($vlmProc.HasExited) {
        $code = if ($null -ne $vlmProc.ExitCode) { $vlmProc.ExitCode } else { "unknown" }
        Write-Event ("vlm exited code={0}" -f $code)
        break
    }
    if ($webProc.HasExited) {
        $code = if ($null -ne $webProc.ExitCode) { $webProc.ExitCode } else { "unknown" }
        Write-Event ("web exited code={0}" -f $code)
        break
    }
}
Write-Event "component exit, stopping remaining service"
& $stopScript -Reason component_exit
