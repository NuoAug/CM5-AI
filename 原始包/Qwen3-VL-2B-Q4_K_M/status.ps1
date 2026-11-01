#Requires -Version 5.1
<#
  查看 Qwen3-VL-2B 本地服务状态
#>
$ErrorActionPreference = 'Continue'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

# ===== 与 deploy.ps1 保持一致 =====
$TagName    = 'Qwen3-VL-2B-Q4_K_M'
$ModelAlias = 'qwen3-vl-2b-q4km'
$Port       = 8091
# =================================

$PidFile  = Join-Path $Root 'server.pid'
$KeyFile  = Join-Path $Root 'api-key.txt'
$Base     = "http://127.0.0.1:$Port"

function Row { param([string]$K, [string]$V, [string]$Color = 'White')
    Write-Host ("    {0,-14}{1}" -f $K, $V) -ForegroundColor $Color }

Write-Host ''
Write-Host "===== $TagName 状态 =====" -ForegroundColor Cyan
Write-Host ''

$running = $false
if (Test-Path $PidFile) {
    $pidText = (Get-Content $PidFile -Raw).Trim()
    if ($pidText -and (Get-Process -Id $pidText -ErrorAction SilentlyContinue)) {
        $running = $true
        Row '进程' "PID $pidText  llama-server" 'Green'
    }
}
$conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($conn) {
    $running = $true
    Row '端口' "$Port 监听中（PID $($conn[0].OwningProcess)）" 'Green'
} else {
    Row '端口' "$Port 无监听" 'Yellow'
}

$health = $null
try { $health = Invoke-RestMethod -Uri "$Base/health" -TimeoutSec 3 -ErrorAction Stop } catch { }
if ($health -and $health.status) {
    Row '健康检查' "$($health.status)" 'Green'
} else {
    Row '健康检查' '不可达' 'Yellow'
}

$key = ''
if (Test-Path $KeyFile) { $key = (Get-Content $KeyFile -Raw).Trim() }
if ($key -and $health) {
    try {
        $models = Invoke-RestMethod -Uri "$Base/v1/models" -Headers @{ Authorization = "Bearer $key" } -TimeoutSec 5 -ErrorAction Stop
        $ids = ($models.data | ForEach-Object { $_.id }) -join ', '
        Row '模型' $ids 'Green'
    } catch {
        Row '模型' '查询失败（API Key 可能不匹配）' 'Yellow'
    }
}

$modelFile = Get-ChildItem (Join-Path $Root 'models') -Filter '*.gguf' -ErrorAction SilentlyContinue |
             Where-Object { $_.Name -notlike 'mmproj*' } | Select-Object -First 1
if ($modelFile) { Row '模型文件' ("{0}  {1} MB" -f $modelFile.Name, [math]::Round($modelFile.Length / 1MB)) }

$smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if ($smi) {
    $mem = & $smi.Source --query-gpu=memory.used,memory.total --format=csv,noheader 2>$null
    if ($mem) { Row '显存' ($mem -join ' / ') }
}

Write-Host ''
if ($running) {
    Write-Host "    网页：$Base/   （填 api-key.txt 里的 Key）" -ForegroundColor Gray
} else {
    Write-Host '    服务没在跑。双击 启动.bat 或 一键部署.bat' -ForegroundColor Gray
}
Write-Host ''
Read-Host '按回车关闭'
