# 停止 VLM 与本地测试网页
param([string]$Reason = "manual")
$root = Split-Path -Parent $PSScriptRoot
$logs = Join-Path $root "logs"
$pidFile = Join-Path $logs "pids.json"
$eventLog = Join-Path $logs "sessions.log"
function Write-Event($msg) {
    Add-Content -Path $eventLog -Encoding UTF8 -Value ("{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg)
}

$sessionDir = ""
if (Test-Path $pidFile) {
    $pids = Get-Content $pidFile -Raw | ConvertFrom-Json
    $sessionDir = [string]$pids.session
    foreach ($procId in @($pids.vlm, $pids.web)) {
        if ($procId) { Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue }
    }
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    Write-Event ("stop reason={0} pids=vlm:{1},web:{2} session={3}" -f $Reason, $pids.vlm, $pids.web, $sessionDir)
} else {
    Write-Event ("stop reason={0} no pids.json" -f $Reason)
}

foreach ($port in @(8090, 8091)) {
    Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
}
if ($sessionDir -and (Test-Path $sessionDir)) {
    Set-Content -Path (Join-Path $sessionDir "stopped.txt") -Value (("reason={0} at {1}" -f $Reason, (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))) -Encoding UTF8
}
Write-Event ("stopped 8090/8091 reason={0}" -f $Reason)
Write-Host "已停止 8090 / 8091 服务（$Reason）"
