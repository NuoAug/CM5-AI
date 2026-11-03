param([string]$Reason="manual")
$ErrorActionPreference="Stop"
$root=Split-Path -Parent $PSScriptRoot
$pidFile=Join-Path $root "logs/pids.json"
if (-not (Test-Path -LiteralPath $pidFile)) { Write-Host "本候选没有PID记录";return }
$state=Get-Content -LiteralPath $pidFile -Raw | ConvertFrom-Json
$expected=@((Join-Path $root "app/.venv/Scripts/python.exe"),(Join-Path $root "runtime/llama.cpp-b12066/llama-server.exe")) | ForEach-Object {[System.IO.Path]::GetFullPath($_)}
foreach($record in $state.processes) {
    $process=Get-Process -Id $record.id -ErrorAction SilentlyContinue
    if ($null -eq $process) {continue}
    $sameStart=$process.StartTime.ToUniversalTime().Ticks -eq [long]$record.start_ticks
    $samePath=$process.Path -and ($expected -contains [System.IO.Path]::GetFullPath($process.Path)) -and $process.Path -eq $record.path
    if (-not ($sameStart -and $samePath)) { throw "PID身份不匹配，拒绝停止其他进程：$($record.id)" }
    Stop-Process -Id $record.id -Force
}
Remove-Item -LiteralPath $pidFile
Write-Host "仅已停止本候选记录的进程（$Reason）"
