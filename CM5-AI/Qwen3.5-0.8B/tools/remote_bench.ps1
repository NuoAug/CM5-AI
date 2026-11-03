# 本机查看 CM5 测试台的 SSH 隧道脚本。只管理自己创建的 ssh 进程。
# 用法：.\remote_bench.ps1 start | check | stop
param(
    [Parameter(Position = 0)]
    [ValidateSet("start", "check", "stop")]
    [string]$Action = "check",
    [Parameter(Mandatory=$true)][string]$HostName,
    [Parameter(Mandatory=$true)][string]$UserName,
    [Parameter(Mandatory=$true)][string]$KeyPath,
    [int]$LocalPort = 18110,
    [int]$RemotePort = 18100
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$StateDir = Join-Path $Root "data\tunnel"
$PidFile = Join-Path $StateDir "ssh-$LocalPort.pid"
$LogFile = Join-Path $StateDir "ssh-$LocalPort.log"
New-Item -ItemType Directory -Force -Path $StateDir | Out-Null

function Test-PortInUse([int]$Port) {
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $client.Connect("127.0.0.1", $Port)
        $client.Close()
        return $true
    } catch {
        return $false
    }
}

function Get-RemoteListening([int]$Port) {
    $cmd = "ss -tln | grep ':$Port ' || true"
    $out = & ssh -i $KeyPath -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=yes "$UserName@$HostName" $cmd
    return ($out -join "`n")
}

function Invoke-Api([string]$Path) {
    $uri = "http://127.0.0.1:$LocalPort$Path"
    return Invoke-RestMethod -Uri $uri -TimeoutSec 10
}

switch ($Action) {
    "start" {
        if (-not (Test-Path $KeyPath)) { throw "找不到 SSH 私钥：$KeyPath" }
        if (Test-PortInUse $LocalPort) { throw "本机 $LocalPort 已被占用；先运行 stop 或换端口" }
        $listening = Get-RemoteListening $RemotePort
        if (-not $listening) { throw "CM5 上 $RemotePort 未监听；先确认服务已启动" }
        if (Test-Path $PidFile) {
            $old = Get-Content $PidFile -ErrorAction SilentlyContinue
            if ($old -and (Get-Process -Id $old -ErrorAction SilentlyContinue)) {
                throw "已有隧道进程 $old；先运行 stop"
            }
            Remove-Item $PidFile -Force
        }
        $sshArgs = @(
            "-N", "-T",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            "-o", "StrictHostKeyChecking=yes",
            "-i", $KeyPath,
            "-L", "127.0.0.1:${LocalPort}:127.0.0.1:${RemotePort}",
            "$UserName@$HostName"
        )
        $proc = Start-Process -FilePath "ssh" -ArgumentList $sshArgs -PassThru -WindowStyle Hidden `
            -RedirectStandardOutput $LogFile -RedirectStandardError "$LogFile.err"
        Set-Content -Path $PidFile -Value $proc.Id
        Start-Sleep -Seconds 2
        if (-not (Get-Process -Id $proc.Id -ErrorAction SilentlyContinue)) {
            throw "ssh 启动后立即退出；查看 $LogFile.err"
        }
        Write-Output "隧道已启动 pid=$($proc.Id) http://127.0.0.1:$LocalPort/"
    }
    "check" {
        if (-not (Test-PortInUse $LocalPort)) { throw "本机 $LocalPort 没有隧道；先运行 start" }
        $h = Invoke-Api "/api/health"
        $s = Invoke-Api "/api/status"
        Write-Output "health=$($h.ok)"
        Write-Output "image_source=$($s.image_source) voice_source=$($s.device.voice_source) sink=$($s.output.sink)"
        Write-Output "vlm_up=$($s.vlm.up) ocr=$($s.ocr.ready) asr_loaded=$($s.asr.ready) tts=$($s.tts.ready)"
        Write-Output "memory_gb=$($s.memory.sum_gb) available_mb=$($s.memory.available_mb) over_budget=$($s.memory.over_budget)"
        Write-Output "URL http://127.0.0.1:$LocalPort/"
    }
    "stop" {
        if (-not (Test-Path $PidFile)) { Write-Output "没有本脚本创建的隧道记录"; break }
        $pidValue = [int](Get-Content $PidFile)
        $proc = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
        if ($proc) {
            if ($proc.ProcessName -ne "ssh") { throw "PID $pidValue 不是 ssh，拒绝停止" }
            Stop-Process -Id $pidValue -Force
            Write-Output "已停止隧道 pid=$pidValue"
        } else {
            Write-Output "隧道进程已不存在"
        }
        Remove-Item $PidFile -Force
    }
}
