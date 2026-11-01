#Requires -Version 5.1
<#
  给 Qwen3-VL-2B 发一张图，让它描述
  用法：双击 看图.bat → 粘贴/拖入图片路径 → 回车
       或 powershell -File see.ps1 -Image "C:\a.jpg" -Question "图里是什么？"
#>
param(
    [string]$Image = '',
    [string]$Question = '这张图里有什么？用中文详细描述。'
)

$ErrorActionPreference = 'Stop'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

# ===== 与 deploy.ps1 保持一致 =====
$TagName = 'Qwen3-VL-2B-Q4_K_M'
$Port    = 8091
# =================================

$KeyFile = Join-Path $Root 'api-key.txt'

Write-Host ''
Write-Host "===== $TagName · 看图 =====" -ForegroundColor Cyan
Write-Host ''

if (-not (Test-Path $KeyFile)) {
    Write-Host '[X] 还没部署。先双击 一键部署.bat' -ForegroundColor Red
    Read-Host '按回车关闭'; exit 1
}
$key = (Get-Content $KeyFile -Raw).Trim()

$base = "http://127.0.0.1:$Port"
$ok = $false
try { $h = Invoke-RestMethod -Uri "$base/health" -TimeoutSec 3 -ErrorAction Stop; if ($h.status -eq 'ok') { $ok = $true } } catch { }
if (-not $ok) {
    Write-Host "[X] 服务没在跑（端口 $Port 无响应）。先双击 启动.bat" -ForegroundColor Red
    Read-Host '按回车关闭'; exit 1
}

if (-not $Image) {
    Write-Host '把图片拖进这个窗口（或粘贴完整路径），回车：' -ForegroundColor Gray
    $Image = (Read-Host '  图片路径').Trim('"', ' ')
}
if (-not (Test-Path -LiteralPath $Image)) {
    Write-Host "[X] 找不到文件：$Image" -ForegroundColor Red
    Read-Host '按回车关闭'; exit 1
}
$img = Get-Item -LiteralPath $Image
if ($img.Length -gt 20MB) {
    Write-Host ("[!]  图片 {0} MB，偏大，稍慢" -f [math]::Round($img.Length / 1MB)) -ForegroundColor Yellow
}

$ext = $img.Extension.ToLowerInvariant()
$mime = 'image/jpeg'
if ($ext -eq '.png') { $mime = 'image/png' }
elseif ($ext -eq '.webp') { $mime = 'image/webp' }
elseif ($ext -eq '.gif') { $mime = 'image/gif' }
elseif ($ext -eq '.bmp') { $mime = 'image/bmp' }

Write-Host ("    图片：{0}  {1}  {2} KB" -f $img.Name, $mime, [math]::Round($img.Length / 1KB)) -ForegroundColor Gray
Write-Host ("    问题：{0}" -f $Question) -ForegroundColor Gray
Write-Host ''
Write-Host '    推理中…' -NoNewline

$b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($img.FullName))
$payload = @{
    messages = @(
        @{
            role    = 'user'
            content = @(
                @{ type = 'text'; text = $Question },
                @{ type = 'image_url'; image_url = @{ url = ("data:{0};base64,{1}" -f $mime, $b64) } }
            )
        }
    )
    max_tokens = 600
    temperature = 0.2
} | ConvertTo-Json -Depth 12 -Compress

$sw = [Diagnostics.Stopwatch]::StartNew()
try {
    $resp = Invoke-RestMethod -Uri "$base/v1/chat/completions" -Method Post -TimeoutSec 600 `
        -Headers @{ Authorization = "Bearer $key" } `
        -ContentType 'application/json; charset=utf-8' `
        -Body ([Text.Encoding]::UTF8.GetBytes($payload))
} catch {
    Write-Host ''
    Write-Host "[X] 请求失败：$($_.Exception.Message)" -ForegroundColor Red
    Read-Host '按回车关闭'; exit 1
}
$sw.Stop()

Write-Host "`r" -NoNewline
Write-Host '---- 回答 ----' -ForegroundColor Green
Write-Host $resp.choices[0].message.content
Write-Host ''
if ($resp.usage) {
    Write-Host ("    用时 {0:N1}s · 提示 {1} token · 输出 {2} token" -f $sw.Elapsed.TotalSeconds, $resp.usage.prompt_tokens, $resp.usage.completion_tokens) -ForegroundColor DarkGray
}
Write-Host ''
Read-Host '按回车关闭'
