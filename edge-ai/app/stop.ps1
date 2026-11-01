# 停止 VLM 与本地测试网页
$root = Split-Path -Parent $PSScriptRoot
$pidFile = Join-Path $root "logs\pids.json"
if (Test-Path $pidFile) {
    $pids = Get-Content $pidFile -Raw | ConvertFrom-Json
    foreach ($id in @($pids.vlm, $pids.web)) {
        if ($id) { Stop-Process -Id $id -Force -ErrorAction SilentlyContinue }
    }
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
}
foreach ($port in @(8090, 8091)) {
    Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
}
Write-Host "已停止 8090 / 8091 服务"
