#Requires -Version 5.1
<#
  停止 Qwen3-VL-2B 本地服务
#>
$ErrorActionPreference = 'Continue'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

# ===== 与 deploy.ps1 保持一致 =====
$TagName = 'Qwen3-VL-2B-Q4_K_M'
$Port    = 8091
# =================================

$PidFile = Join-Path $Root 'server.pid'

Write-Host ''
Write-Host "==> 停止 $TagName" -ForegroundColor Cyan

$stopped = $false

if (Test-Path $PidFile) {
    $pidText = (Get-Content $PidFile -Raw).Trim()
    if ($pidText) {
        $p = Get-Process -Id $pidText -ErrorAction SilentlyContinue
        if ($p) {
            Write-Host "    结束进程 PID $pidText（$($p.ProcessName)）"
            & taskkill /PID $pidText /F /T 2>$null | Out-Null
            $stopped = $true
        }
    }
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}

# 兜底：按端口找进程
$conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($conn) {
    $ownPid = $conn[0].OwningProcess
    Write-Host "    端口 $Port 仍被 PID $ownPid 占用，强制结束"
    & taskkill /PID $ownPid /F /T 2>$null | Out-Null
    $stopped = $true
}

Start-Sleep -Milliseconds 800
$left = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($left) {
    Write-Host "    [X] 端口 $Port 仍在监听" -ForegroundColor Red
} elseif ($stopped) {
    Write-Host "    [OK] 已停止，端口 $Port 已释放" -ForegroundColor Green
} else {
    Write-Host "    [OK] 本来就没在运行" -ForegroundColor Green
}
Write-Host ''
Read-Host '按回车关闭'
