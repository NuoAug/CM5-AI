#Requires -Version 5.1
<#
  给 DSH「本地模型」插件用的中转启动器。
  插件只会认 serve-bonsai.ps1 / qwen-deploy.ps1 / scripts\serve.ps1 三种名字，
  参数固定传 -Background -NoBrowser [-Port <n>] [-Ctx <n>]，这里原样转给 deploy.ps1。
#>
param(
    [switch]$Background,
    [switch]$NoBrowser,
    [int]$Port = 0,
    [int]$Ctx = 0
)

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$target = Join-Path $here 'deploy.ps1'
if (-not (Test-Path $target)) { Write-Host "[X] 找不到 deploy.ps1：$target" -ForegroundColor Red; exit 1 }

$argv = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $target)
if ($Background) { $argv += '-Background' }
if ($NoBrowser)  { $argv += '-NoBrowser' }
if ($Port -gt 0) { $argv += @('-Port', "$Port") }
if ($Ctx  -gt 0) { $argv += @('-Ctx', "$Ctx") }

& powershell.exe @argv
exit $LASTEXITCODE