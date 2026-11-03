param(
    [Parameter(Mandatory=$true)][ValidateRange(1024,65535)][int]$WebPort,
    [Parameter(Mandatory=$true)][ValidateRange(1024,65535)][int]$VlmPort,
    [ValidateRange(1,16)][int]$Threads = 4,
    [ValidateRange(512,8192)][int]$Context = 2048,
    [switch]$RunnerOnly,
    [switch]$NoWait
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$profile = Get-Content -LiteralPath (Join-Path $root "config/model-profile.json") -Raw | ConvertFrom-Json
# Do not inherit another project's model/cache locations.
foreach ($name in @('DATA_DIR','PIPER_BIN','PIPER_MODEL','KWS_MODEL_DIR','KWS_KEYWORDS','ASR_STREAMING_MODEL_DIR','STREAMING_ASR_MODEL_DIR')) {
    [Environment]::SetEnvironmentVariable($name, $null, 'Process')
}
$env:DATA_DIR = Join-Path $root 'data'
$env:PIPER_BIN = Join-Path $root 'runtime/piper/piper/piper.exe'
$env:PIPER_MODEL = Join-Path $root 'models/tts/piper-zh_CN-huayan-medium/zh_CN-huayan-medium.onnx'
$env:KWS_MODEL_DIR = Join-Path $root 'models/kws/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01'
$env:KWS_KEYWORDS = Join-Path $root 'app/config/keywords.txt'
$env:ASR_STREAMING_MODEL_DIR = Join-Path $root 'models/asr-streaming/sherpa-onnx-streaming-paraformer-bilingual-zh-en'
$python = Join-Path $root "app/.venv/Scripts/python.exe"
foreach ($property in $profile.PSObject.Properties) {
    if ($property.Name -match '(_file|_file_posix|_dir)$') {
        $resolved = [System.IO.Path]::GetFullPath((Join-Path $root $property.Value))
        $prefix = [System.IO.Path]::GetFullPath($root).TrimEnd('\') + '\'
        if (-not $resolved.StartsWith($prefix,[System.StringComparison]::OrdinalIgnoreCase)) { throw "Candidate profile path escapes its own project: $($property.Name)" }
    }
}
$runner = Join-Path $root $profile.runner_file
$model = Join-Path $root $profile.model_file
$mmproj = Join-Path $root $profile.mmproj_file
if ($WebPort -eq $VlmPort) { throw "网页与运行器端口不能相同" }
foreach ($port in @($WebPort,$VlmPort)) {
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback,$port)
    try { $listener.Start() } catch { throw "端口 $port 已占用；不接管或停止其他模型" } finally { $listener.Stop() }
}
foreach ($path in @($python,$runner,$model,$mmproj)) { if (-not (Test-Path -LiteralPath $path)) { throw "候选未就绪，缺少 $path；禁止回退其他模型" } }
$env:PYTHONIOENCODING = "utf-8"
$checkArgs = @("-X","utf8",(Join-Path $PSScriptRoot "precheck.py"))
if ($RunnerOnly) { $checkArgs += "--vlm-only" }
& $python @checkArgs
if ($LASTEXITCODE -ne 0) { throw "候选预检未通过" }
$logs = Join-Path $root "logs"
[System.IO.Directory]::CreateDirectory($logs) | Out-Null
$pidFile = Join-Path $logs "pids.json"
if (Test-Path -LiteralPath $pidFile) { throw "已有候选PID记录，请先运行本工程stop.ps1核对停止" }
$sessionDir = Join-Path $logs ("session-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
[System.IO.Directory]::CreateDirectory($sessionDir) | Out-Null
if (-not $env:VLM_API_KEY) {
    $bytes = New-Object byte[] 32
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try { $rng.GetBytes($bytes) } finally { $rng.Dispose() }
    $env:VLM_API_KEY = -join ($bytes | ForEach-Object {$_.ToString("x2")})
}
$env:VLM_URL = "http://127.0.0.1:$VlmPort"
$env:VLM_MODEL_ID = $profile.model_id
$env:WEB_PORT = "$WebPort"
$env:WEB_HOST = "127.0.0.1"
$env:VOICE_SOURCE = "browser"
$env:OUTPUT_SINK = "browser"
$env:IMAGE_SOURCE = "main-control"
$records = @()
function Save-Record($process) {
    $process.Refresh()
    $script:records += @{ id=$process.Id; start_ticks=$process.StartTime.ToUniversalTime().Ticks; path=$process.Path }
    @{ profile_id=$profile.profile_id; processes=$script:records; session=$sessionDir; vlm_port=$VlmPort; web_port=$WebPort } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $pidFile -Encoding UTF8
}
try {
    $vlmArgs = @("-m",('"'+$model+'"'),"--mmproj",('"'+$mmproj+'"'),"--no-mmproj-offload","-c","$Context","-t","$Threads","-ngl","0","--parallel","1","--host","127.0.0.1","--port","$VlmPort","--alias",$profile.model_id,"--api-key",$env:VLM_API_KEY)
    $vlm = Start-Process -FilePath $runner -ArgumentList $vlmArgs -WorkingDirectory (Split-Path -Parent $runner) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $sessionDir "vlm.log") -RedirectStandardError (Join-Path $sessionDir "vlm.err.log")
    Save-Record $vlm
    $ready = $false
    for ($i=0;$i -lt 90;$i++) {
        if ($vlm.HasExited) { throw "候选运行器提前退出，详见本工程日志" }
        try {
            $models = Invoke-RestMethod -Uri ($env:VLM_URL+"/v1/models") -Headers @{Authorization="Bearer "+$env:VLM_API_KEY} -TimeoutSec 2
            if (@($models.data | Where-Object {$_.id -eq $profile.model_id}).Count -gt 0) {$ready=$true;break}
        } catch {}
        Start-Sleep -Seconds 1
    }
    if (-not $ready) { throw "候选90秒内未就绪，禁止改连其他模型" }
    if (-not $RunnerOnly) {
        $web=Start-Process -FilePath $python -ArgumentList @('-X','utf8',('"'+(Join-Path $PSScriptRoot 'server.py')+'"')) -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $sessionDir 'web.log') -RedirectStandardError (Join-Path $sessionDir 'web.err.log')
        Save-Record $web
    }
    Write-Host "Qwen候选运行器：$env:VLM_URL；网页端口：$WebPort；设备采集未启动"
    if ($NoWait) { return }
    while (-not $vlm.HasExited) { Start-Sleep -Seconds 1; if ($web -and $web.HasExited) {break} }
} catch {
    & (Join-Path $PSScriptRoot 'stop.ps1') -Reason 'start_failed'
    throw
} finally {
    if (-not $NoWait) { & (Join-Path $PSScriptRoot 'stop.ps1') -Reason 'launcher_exit' }
}
