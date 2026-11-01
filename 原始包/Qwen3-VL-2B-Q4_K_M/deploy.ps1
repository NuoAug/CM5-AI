#Requires -Version 5.1
<#
  Qwen3-VL-2B 一键部署（llama.cpp / Windows x64 / CUDA）
  用法：
    double click  一键部署.bat
    或  powershell -NoProfile -ExecutionPolicy Bypass -File deploy.ps1
  可选参数：
    -SkipStart      只部署、不启动
    -Runtime auto   运行时来源：auto（默认，能复用就复用）/ download / reuse / <本地目录>
#>
[CmdletBinding()]  # 已扩参：支持插件以 -Background -NoBrowser -Port <n> -Ctx <n> 调用
param(
    [switch]$SkipStart,
    [string]$Runtime = 'auto',
    [int]$Port = 0,
    [int]$Ctx = 0,
    [switch]$Background,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

# ==================== 本套包的配置（另一套包的脚本只在这几行不同） ====================
$TagName    = 'Qwen3-VL-2B-Q4_K_M'
$ModelFile  = 'Qwen3-VL-2B-Instruct-Q4_K_M.gguf'
$MmprojFile = 'mmproj-F16.gguf'
$ModelAlias = 'qwen3-vl-2b-q4km'
$DefaultPort = 8091
# ==================================================================================

$Repo       = 'unsloth/Qwen3-VL-2B-Instruct-GGUF'
$CtxSize    = 16384
$GpuLayers  = 99
if ($Port -le 0) { $Port = $DefaultPort }
if ($Ctx -gt 0) { $CtxSize = $Ctx }

# 固定 Key：与 DSH 里的 apiKeyEnv 对齐，删了 api-key.txt 也能复现同一个 Key
$FixedKey   = 'sk-vlq4kmc68b4dac97ed216ff6cc91858c'

$ModelDir   = Join-Path $Root 'models'
$RuntimeDir = Join-Path $Root 'runtime'
$LogDir     = Join-Path $Root 'logs'
$PidFile    = Join-Path $Root 'server.pid'
$KeyFile    = Join-Path $Root 'api-key.txt'
$InfoFile   = Join-Path $Root 'api-info.txt'
$CacheDir   = Join-Path $env:LOCALAPPDATA 'llama-cpp-cache'
$ModelPath  = Join-Path $ModelDir $ModelFile
$MmprojPath = Join-Path $ModelDir $MmprojFile

$GhMirrors  = @('https://gh-proxy.com/', 'https://ghfast.top/', 'https://ghproxy.net/', '')
$HfMirrors  = @('https://hf-mirror.com', 'https://huggingface.co')
$FallbackTag = 'b11960'
$Curl       = Join-Path $env:SystemRoot 'System32\curl.exe'

# ==================== 输出小工具 ====================
function Line { param([string]$Text = '') Write-Host $Text }
function Step { param([string]$Text) Write-Host ''; Write-Host "==> $Text" -ForegroundColor Cyan }
function Ok   { param([string]$Text) Write-Host "    [OK] $Text" -ForegroundColor Green }
function Warn { param([string]$Text) Write-Host "    [!]  $Text" -ForegroundColor Yellow }
function Info { param([string]$Text) Write-Host "    $Text" -ForegroundColor Gray }
function Die  {
    param([string]$Text)
    Write-Host ''
    Write-Host "[X] $Text" -ForegroundColor Red
    Write-Host ''
    if (-not $Background) { try { Read-Host '按回车关闭' } catch { } }
    exit 1
}

function Get-SizeMB { param([int64]$Bytes) return [math]::Round($Bytes / 1MB, 0) }

# ==================== 0. 基本环境 ====================
Step "环境检查"
if (-not [Environment]::Is64BitProcess) { Die '需要 64 位 PowerShell（请在 64 位 Windows 上运行）' }
if (-not (Test-Path $Curl)) { Die "找不到 $Curl（需要 Windows 10 1803 以上）" }
Ok ("PowerShell {0} / {1}" -f $PSVersionTable.PSVersion, [Environment]::OSVersion.VersionString)

$drive = (Get-Item $Root).PSDrive
if ($drive.Free -lt 12GB) {
    Warn ("{0} 盘剩余 {1} GB，模型+运行时约需 10 GB" -f $drive.Name, [math]::Round($drive.Free / 1GB, 1))
} else {
    Ok ("磁盘剩余 {0} GB" -f [math]::Round($drive.Free / 1GB, 1))
}

# 端口占用
$busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($busy) {
    Warn "端口 $Port 已被占用（PID $($busy[0].OwningProcess)），启动可能失败"
}

# ==================== 1. 显卡 / 驱动 ====================
Step "显卡与驱动"
$cudaMajor = 0
$smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if ($smi) {
    try {
        $q = & $smi.Source --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>$null
        if ($q) { Ok ("GPU: " + ($q -join ' / ')) }
        $cv = & $smi.Source 2>$null | Select-String -Pattern 'CUDA Version:\s*([0-9]+)\.([0-9]+)' | Select-Object -First 1
        if ($cv) {
            $cudaMajor = [int]$cv.Matches[0].Groups[1].Value
            Ok ("驱动支持的 CUDA 版本：$($cv.Matches[0].Groups[1].Value).$($cv.Matches[0].Groups[2].Value)")
        }
    } catch { Warn "nvidia-smi 调用失败：$($_.Exception.Message)" }
} else {
    Warn '没找到 nvidia-smi，如果不是 NVIDIA 显卡，本包只能用 CPU 跑（会很慢）'
}
if ($cudaMajor -lt 13) { $cudaFlavor = '12.4' } else { $cudaFlavor = '13.4' }
Info "将使用 CUDA 运行库：$cudaFlavor"

# ==================== 2. 下载工具 ====================
function Get-RemoteSize {
    param([string]$Url)
    $h = & $Curl -sIL --max-time 25 $Url 2>$null
    if (-not $h) { return 0 }
    $m = [regex]::Matches(($h -join "`n"), '(?im)^content-length:\s*(\d+)')
    if ($m.Count -eq 0) { return 0 }
    return [int64]$m[$m.Count - 1].Groups[1].Value
}

function Get-FileLen {
    param([string]$Path)
    try {
        if (-not (Test-Path -LiteralPath $Path)) { return [int64]0 }
        return (New-Object System.IO.FileInfo($Path)).Length
    } catch { return [int64]0 }
}

function Format-Size {
    param([int64]$Bytes)
    if ($Bytes -ge 1GB) { return ('{0:N2} GB' -f ($Bytes / 1GB)) }
    if ($Bytes -ge 1MB) { return ('{0:N0} MB' -f ($Bytes / 1MB)) }
    return ('{0:N0} KB' -f ($Bytes / 1KB))
}

function Get-ConWidth {
    $w = 0
    try {
        if (-not [Console]::IsOutputRedirected) { $w = [Console]::WindowWidth }
    } catch { $w = 0 }
    if ($w -le 0) {
        try { $w = $Host.UI.RawUI.WindowSize.Width } catch { $w = 0 }
    }
    if ($w -le 40) { $w = 100 }
    return $w
}

function Fit-Cells {
    param([string]$Text, [int]$MaxCells)
    $sb = New-Object System.Text.StringBuilder
    $w = 0
    foreach ($ch in $Text.ToCharArray()) {
        $c = [int][char]$ch
        $cw = 1
        if (($c -ge 0x1100 -and $c -le 0x115F) -or ($c -ge 0x2E80 -and $c -le 0xA4CF) -or
            ($c -ge 0xAC00 -and $c -le 0xD7A3) -or ($c -ge 0xF900 -and $c -le 0xFAFF) -or
            ($c -ge 0xFE30 -and $c -le 0xFE4F) -or ($c -ge 0xFF00 -and $c -le 0xFF60)) { $cw = 2 }
        if (($w + $cw) -gt $MaxCells) { break }
        [void]$sb.Append($ch)
        $w += $cw
    }
    while ($w -lt $MaxCells) { [void]$sb.Append(' '); $w += 1 }
    return $sb.ToString()
}

function Get-LineLimit {
    $w = Get-ConWidth
    $limit = $w - 10
    if ($limit -gt 64) { $limit = 64 }
    if ($limit -lt 30) { $limit = 30 }
    return $limit
}

function Format-Bar {
    param([double]$Pct, [int]$Width = 16)
    $n = [int][math]::Round($Pct / 100.0 * $Width)
    if ($n -lt 0) { $n = 0 }
    if ($n -gt $Width) { $n = $Width }
    return ('[' + ('#' * $n) + ('-' * ($Width - $n)) + ']')
}

function Show-Progress {
    param([int64]$Now, [int64]$Total, [double]$Speed)
    $limit = Get-LineLimit
    $line = ''
    if ($Total -gt 0) {
        $pct = [math]::Min(100.0, ($Now * 100.0 / $Total))
        $tail = ''
        if ($Speed -gt 0.05 -and $Now -lt $Total) {
            $sec = [int](($Total - $Now) / ($Speed * 1MB))
            if ($sec -ge 60) { $tail = ('剩余{0}分{1:00}秒' -f [int][math]::Floor($sec / 60), ($sec % 60)) }
            else { $tail = ('剩余{0}秒' -f $sec) }
        }
        if ($Total -ge 1GB) {
            $sz = '{0:N2}/{1:N2} GB' -f ($Now / 1GB), ($Total / 1GB)
        } else {
            $sz = '{0:N0}/{1:N0} MB' -f ($Now / 1MB), ($Total / 1MB)
        }
        $line = '{0} {1,5:N1}%  {2}  {3,4:N1} MB/s  {4}' -f (Format-Bar $pct 16), $pct, $sz, $Speed, $tail
        if ($limit -lt 56) {
            $line = '{0} {1,5:N1}% {2} {3,3:N1}MB/s' -f (Format-Bar $pct 10), $pct, $sz, $Speed
        }
    } else {
        $line = '{0}  {1}  {2,4:N1} MB/s' -f (Format-Bar 0 16), (Format-Size $Now), $Speed
    }
    $fitted = Fit-Cells $line $limit
    if ([Console]::IsOutputRedirected) {
        Write-Host ('  ' + $fitted.TrimEnd())
        return
    }
    Write-Host ("`r  " + $fitted) -NoNewline
}

function End-Progress {
    if ([Console]::IsOutputRedirected) { return }
    $limit = Get-LineLimit
    Write-Host ("`r" + (' ' * ($limit + 2)) + "`r") -NoNewline
}
function Invoke-CurlDownload {
    param(
        [string]$Url,
        [string]$OutFile,
        [string]$Label,
        [int64]$Total = 0,
        [switch]$NoResume
    )
    $tmpOut = Join-Path $env:TEMP ('vldl-out-' + [guid]::NewGuid().ToString('N') + '.log')
    $tmpErr = Join-Path $env:TEMP ('vldl-err-' + [guid]::NewGuid().ToString('N') + '.log')

    $cargs = @('-L', '--fail', '--retry', '8', '--retry-delay', '3', '--retry-all-errors',
               '--connect-timeout', '30', '--speed-limit', '20000', '--speed-time', '60',
               '-s', '-S', '-w', '%{http_code}', '-o', $OutFile)
    if (-not $NoResume) { $cargs += '-C'; $cargs += '-' }
    $cargs += $Url

    $proc = Start-Process -FilePath $Curl -ArgumentList $cargs -NoNewWindow -PassThru `
        -RedirectStandardOutput $tmpOut -RedirectStandardError $tmpErr

    $sw = [Diagnostics.Stopwatch]::StartNew()
    $lastT = 0.0
    $lastB = Get-FileLen $OutFile
    while (-not $proc.HasExited) {
        Start-Sleep -Milliseconds 400
        $t = $sw.Elapsed.TotalSeconds
        if (($t - $lastT) -ge 0.6) {
            $now = Get-FileLen $OutFile
            $spd = 0.0
            if (($t - $lastT) -gt 0.05) { $spd = (($now - $lastB) / 1MB) / ($t - $lastT) }
            if ($spd -lt 0) { $spd = 0 }
            Show-Progress -Now $now -Total $Total -Speed $spd
            $lastT = $t
            $lastB = $now
        }
    }
    $proc.WaitForExit()
    $got = Get-FileLen $OutFile
    Show-Progress -Now $got -Total $Total -Speed 0
    End-Progress

    $http = ''
    if (Test-Path $tmpOut) { $http = ((@(Get-Content -LiteralPath $tmpOut -ErrorAction SilentlyContinue) -join '') -replace '\s', '') }
    $errText = ''
    if (Test-Path $tmpErr) {
        $lines = @(Get-Content -LiteralPath $tmpErr -ErrorAction SilentlyContinue | Where-Object { $_.Trim() } | Select-Object -Unique)
        $errText = (($lines -join ' ') -replace '\s+', ' ').Trim()
    }

    $ok = ($http -eq '200' -or $http -eq '206')
    if (-not $ok -and $Total -gt 0 -and $got -eq $Total) { $ok = $true }
    if (-not $ok -and $errText) { Warn ('curl 报错信息：' + $errText) }

    Remove-Item $tmpOut, $tmpErr -Force -ErrorAction SilentlyContinue
    return [pscustomobject]@{ Ok = $ok; Http = $http; Bytes = $got; Err = $errText }
}

function Get-Download {
    param(
        [string]$Url,
        [string]$OutFile,
        [string]$Label,
        [int64]$Expect = 0
    )
    $dir = Split-Path -Parent $OutFile
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }

    if (Test-Path $OutFile) {
        $have = (Get-Item $OutFile).Length
        if ($Expect -gt 0 -and $have -eq $Expect) {
            Ok ("{0} 已下载完整（{1:N0} MB），跳过" -f $Label, ($have / 1MB))
            return $true
        }
        if ($Expect -gt 0 -and $have -gt $Expect) {
            Remove-Item $OutFile -Force
        } elseif ($have -gt 0) {
            Info ("上次下了 {0:N0} MB 没完成，从断点继续" -f ($have / 1MB))
        }
    }

    Info ("开始下载 $Label")
    if ($Expect -gt 0) { Info ("共 {0:N0} MB" -f ($Expect / 1MB)) }
    Write-Host ''

    $r = Invoke-CurlDownload -Url $Url -OutFile $OutFile -Total $Expect
    if (-not $r.Ok -and $r.Http -eq '416') {
        Warn '服务器说续传范围无效，改为从头下载'
        Remove-Item $OutFile -Force -ErrorAction SilentlyContinue
        $r = Invoke-CurlDownload -Url $Url -OutFile $OutFile -Total $Expect -NoResume
    }
    if (-not $r.Ok) {
        Warn ("{0} 下载没成功（HTTP {1}）；已下的部分保留，重跑会接着下" -f $Label, $r.Http)
        return $false
    }
    if ($Expect -gt 0) {
        $got = (Get-Item $OutFile).Length
        if ($got -ne $Expect) {
            Warn ("{0} 大小不符：{1:N0} / {2:N0} MB；重跑会继续补" -f $Label, ($got / 1MB), ($Expect / 1MB))
            return $false
        }
    }
    Ok ("{0} 完成（{1:N0} MB）" -f $Label, ((Get-Item $OutFile).Length / 1MB))
    return $true
}

function Get-DownloadAny {
    param(
        [string[]]$Urls,
        [string]$OutFile,
        [string]$Label,
        [int64]$Expect = 0
    )
    foreach ($u in $Urls) {
        Info "来源：$u"
        if (Get-Download -Url $u -OutFile $OutFile -Label $Label -Expect $Expect) { return $true }
        if ((Test-Path $OutFile) -and ((Get-Item $OutFile).Length -gt 0)) {
            Warn '已保留下载进度，换源会丢，先停在这里'
            return $false
        }
    }
    return $false
}
# ==================== 3. 运行时 llama.cpp ====================
function Test-RuntimeDir {
    param([string]$Dir)
    if (-not $Dir) { return $false }
    $exe = Join-Path $Dir 'llama-server.exe'
    return (Test-Path $exe)
}

function Find-LocalRuntime {
    $cands = @(
        $RuntimeDir,
        'D:\1\Qwen3.5-9B · Q4_K_XL\llama.cpp',
        'D:\1\Qwen3.5-9B · Q4_K_XL'
    )
    foreach ($c in $cands) {
        if (Test-RuntimeDir $c) { return $c }
    }
    return $null
}

function Get-LlamaTag {
    foreach ($m in $GhMirrors) {
        try {
            $j = & $Curl -sL --max-time 25 ("{0}https://api.github.com/repos/ggml-org/llama.cpp/releases/latest" -f $m) 2>$null
            if (-not $j) { continue }
            $o = ($j -join "`n") | ConvertFrom-Json
            $tag = $o.tag_name
            if (-not $tag) { continue }
            $nt = & $Curl -sL --max-time 25 ("{0}https://github.com/ggml-org/llama.cpp/releases/download/{1}/nightly-tag.txt" -f $m, $tag) 2>$null
            if ($nt) {
                $build = ($nt -join '').Trim()
                if ($build -match '^b\d+$') { return $build }
            }
        } catch { }
    }
    return $null
}

function Expand-ZipTo {
    param([string]$Zip, [string]$Dest)
    if (Test-Path $Dest) { Remove-Item $Dest -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $Dest | Out-Null
    $tar = Join-Path $env:SystemRoot 'System32\tar.exe'
    if (Test-Path $tar) {
        & $tar -xf $Zip -C $Dest
        if ($LASTEXITCODE -ne 0) { throw "解压失败：$Zip" }
    } else {
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        [System.IO.Compression.ZipFile]::ExtractToDirectory($Zip, $Dest)
    }
}

function Install-RuntimeFromZip {
    param([string]$Build, [string]$Flavor)
    if (-not (Test-Path $CacheDir)) { New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null }
    $binZip = Join-Path $CacheDir ("llama-{0}-bin-win-cuda-{1}-x64.zip" -f $Build, $Flavor)
    $cudZip = Join-Path $CacheDir ("cudart-llama-bin-win-cuda-{0}-x64.zip" -f $Flavor)

    $binUrl = @()
    $cudUrl = @()
    foreach ($m in $GhMirrors) {
        $binUrl += ("{0}https://github.com/ggml-org/llama.cpp/releases/download/{1}/llama-{1}-bin-win-cuda-{2}-x64.zip" -f $m, $Build, $Flavor)
        $cudUrl += ("{0}https://github.com/ggml-org/llama.cpp/releases/download/{1}/cudart-llama-bin-win-cuda-{2}-x64.zip" -f $m, $Build, $Flavor)
    }

    Step "下载 llama.cpp 运行时（build $Build / CUDA $Flavor）"
    $sizeBin = Get-RemoteSize $binUrl[0]
    if (-not (Get-DownloadAny -Urls $binUrl -OutFile $binZip -Label 'llama.cpp 主程序' -Expect $sizeBin)) {
        return $false
    }
    $sizeCud = Get-RemoteSize $cudUrl[0]
    Step "下载 CUDA 运行库（cudart / cublas）"
    if (-not (Get-DownloadAny -Urls $cudUrl -OutFile $cudZip -Label 'CUDA 运行库' -Expect $sizeCud)) {
        return $false
    }

    Step "解压运行时"
    $tmp = Join-Path $CacheDir 'unpack'
    Expand-ZipTo -Zip $binZip -Dest $tmp
    if (Test-Path $RuntimeDir) { Remove-Item $RuntimeDir -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
    Get-ChildItem -Path $tmp -Recurse -File | ForEach-Object {
        Move-Item -LiteralPath $_.FullName -Destination (Join-Path $RuntimeDir $_.Name) -Force
    }
    Expand-ZipTo -Zip $cudZip -Dest $tmp
    Get-ChildItem -Path $tmp -Recurse -File | ForEach-Object {
        Move-Item -LiteralPath $_.FullName -Destination (Join-Path $RuntimeDir $_.Name) -Force
    }
    Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
    Ok '运行时解压完成'
    return (Test-RuntimeDir $RuntimeDir)
}

function Ensure-Runtime {
    if (Test-RuntimeDir $RuntimeDir) {
        Ok "运行时已就绪：$RuntimeDir"
        return
    }
    Step "准备 llama.cpp 运行时"
    if ($Runtime -eq 'reuse' -or $Runtime -eq 'auto') {
        $local = Find-LocalRuntime
        if ($local -and $local -ne $RuntimeDir) {
            Info "复用本机已有运行时：$local"
            New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
            Copy-Item -Path (Join-Path $local '*') -Destination $RuntimeDir -Recurse -Force
            if (Test-RuntimeDir $RuntimeDir) { Ok '运行时复制完成'; return }
            Warn '复制后仍找不到 llama-server.exe，改为下载'
        }
        if ($Runtime -eq 'reuse') { Die '没有可复用的运行时，请改用 -Runtime download' }
    }
    if ($Runtime -ne 'auto' -and $Runtime -ne 'download' -and $Runtime -ne 'reuse') {
        if (Test-RuntimeDir $Runtime) {
            New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
            Copy-Item -Path (Join-Path $Runtime '*') -Destination $RuntimeDir -Recurse -Force
            Ok "已从 $Runtime 复制运行时"
            return
        }
        Die "指定的运行时目录无效：$Runtime"
    }

    $build = Get-LlamaTag
    if (-not $build) { Warn "没取到最新 build 号，改用 $FallbackTag"; $build = $FallbackTag }
    Ok "目标 build：$build"

    $flavors = @($cudaFlavor)
    if ($cudaFlavor -ne '12.4') { $flavors += '12.4' }
    foreach ($f in $flavors) {
        if (Install-RuntimeFromZip -Build $build -Flavor $f) { return }
        Warn "CUDA $f 版安装失败，换下一个"
    }
    Die '运行时下载失败。可手动下载 llama.cpp 的 win-cuda 压缩包解压到 runtime\ 目录后重跑。'
}

# ==================== 4. 模型 ====================
function Ensure-Model {
    param([string]$File, [string]$Label)
    $dst = Join-Path $ModelDir $File
    if ((Test-Path $dst) -and (Get-Item $dst).Length -gt 100MB) {
        Ok ("{0} 已存在（{1} MB）" -f $Label, (Get-SizeMB (Get-Item $dst).Length))
        return
    }
    Step "下载 $Label"
    $urls = @()
    foreach ($h in $HfMirrors) {
        $urls += ("{0}/{1}/resolve/main/{2}?download=true" -f $h, $Repo, $File)
    }
    $size = Get-RemoteSize $urls[0]
    if ($size -le 0) { Warn '取不到文件大小，按未知大小下载' }
    if (-not (Get-DownloadAny -Urls $urls -OutFile $dst -Label $Label -Expect $size)) {
        Die ("{0} 下载失败（重跑可续传）。也可浏览器打开 https://hf-mirror.com/{1}/tree/main 手动下载后放进 models\ 目录" -f $Label, $Repo)
    }
}

# ==================== 5. 启动 ====================
function Ensure-ApiKey {
    if ((Test-Path $KeyFile) -and ((Get-Content $KeyFile -Raw).Trim().Length -gt 10)) { return (Get-Content $KeyFile -Raw).Trim() }
    $k = 'sk-' + ([guid]::NewGuid().ToString('N') + [guid]::NewGuid().ToString('N')).Substring(0, 32)
    Set-Content -Path $KeyFile -Value $k -Encoding ASCII
    Ok "已写入 API Key（$KeyFile）"
    return $k
}

function Get-FreeVramMB {
    try {
        $v = & nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>$null
        if ($v) { return [int](@($v)[0].Trim()) }
    } catch { }
    return -1
}

function Get-PortOwner {
    param([int]$P)
    try {
        $c = Get-NetTCPConnection -LocalPort $P -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($c) { return [int]$c.OwningProcess }
    } catch { }
    return 0
}

function Wait-Health {
    param([int]$TimeoutSec = 240, [string]$Key = '')
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    $url = "http://127.0.0.1:$Port/health"
    $headers = @{}
    if ($Key) { $headers['Authorization'] = "Bearer $Key" }
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Milliseconds 700
        $proc = Get-Process -Name 'llama-server' -ErrorAction SilentlyContinue
        if (-not $proc) { return 'exited' }
        if (-not (Get-PortOwner $Port)) { continue }
        try {
            $r = Invoke-RestMethod -Uri $url -Headers $headers -TimeoutSec 3 -ErrorAction Stop
            if ($r.status -eq 'ok') { return 'ok' }
        } catch {
            $resp = $_.Exception.Response
            if ($resp -and [int]$resp.StatusCode -eq 401) { return 'ok' }
        }
    }
    return 'timeout'
}

function Start-Server {
    param([string]$Key)
    Step "启动 llama-server"
    $exe = Join-Path $RuntimeDir 'llama-server.exe'
    if (Test-Path $PidFile) {
        $old = (Get-Content $PidFile -Raw).Trim()
        if ($old) {
            $op = Get-Process -Id $old -ErrorAction SilentlyContinue
            if ($op) {
                Warn "已有实例在跑（PID $old），先停掉"
                Stop-Server | Out-Null
                Start-Sleep -Seconds 2
            }
        }
    }
    if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Force -Path $LogDir | Out-Null }
    foreach ($sp in @(8090, 8091)) {
        if ($sp -eq $Port) { continue }
        $owner = Get-PortOwner $sp
        if ($owner) {
            Warn "另一个 Qwen3-VL 服务还在 $sp 上跑（PID $owner），会和本服务抢显存，先停掉它"
            Stop-Process -Id $owner -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 2
        }
    }
    $freeMB = Get-FreeVramMB
    if ($freeMB -gt 0) {
        $needMB = [int](((Get-Item $ModelPath).Length + (Get-Item $MmprojPath).Length) / 1MB) + 2600
        if ($freeMB -lt $needMB) {
            Warn ("显存偏紧：空闲 {0} MB，预计需要约 {1} MB；不够的话会加载失败" -f $freeMB, $needMB)
            Warn '建议先关掉其它占显存的程序（浏览器、游戏、另一个模型服务）再跑'
        } else {
            Info ("空闲显存 {0} MB，预计需要约 {1} MB" -f $freeMB, $needMB)
        }
    }

    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $outLog = Join-Path $LogDir "server-$stamp.out.log"
    $errLog = Join-Path $LogDir "server-$stamp.err.log"

    $cargs = @(
        '-m', $ModelPath,
        '--mmproj', $MmprojPath,
        '-c', "$CtxSize",
        '-ngl', "$GpuLayers",
        '--host', '127.0.0.1',
        '--port', "$Port",
        '--api-key', $Key,
        '--alias', $ModelAlias,
        '--image-min-tokens', '1024',
        '--jinja'
    )
    Info ("参数：" + ($cargs -join ' '))
    $p = Start-Process -FilePath $exe -ArgumentList $cargs -WorkingDirectory $RuntimeDir -PassThru `
        -RedirectStandardOutput $outLog -RedirectStandardError $errLog
    Set-Content -Path $PidFile -Value $p.Id -Encoding ASCII
    Info "PID $($p.Id)，日志：$errLog"

    Info '等待模型加载（首次约 10-60 秒）'
    $state = Wait-Health -TimeoutSec 240 -Key $Key
    if ($state -ne 'ok') {
        if ($state -eq 'exited') { Warn 'llama-server 进程已退出（最常见原因是显存不够，或模型文件不完整）' }
        else { Warn '等了 240 秒服务仍没就绪（端口在监听但 /health 没回应）' }
        Warn '日志最后 20 行：'
        if (Test-Path $errLog) { Get-Content $errLog -Tail 20 | ForEach-Object { Write-Host "      $_" -ForegroundColor DarkGray } }
        if (Test-Path $outLog) { Get-Content $outLog -Tail 10 | ForEach-Object { Write-Host "      $_" -ForegroundColor DarkGray } }
        $freeNow = Get-FreeVramMB
        if ($freeNow -gt 0) { Warn ("当前空闲显存：{0} MB" -f $freeNow) }
        Warn '排查顺序：① 8090 与 8091 同时只能跑一个，先运行「停止.bat」把另一个停掉；'
        Warn '          ② 关掉浏览器/游戏等占显存的程序；③ 用 -c 8192 减小上下文再试'
        Die '服务没能就绪'
    }
    Ok '服务已就绪'

    $base = "http://127.0.0.1:$Port"
    $info = @(
        "Qwen3-VL-2B 本地服务（$TagName）",
        "生成时间：$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
        "",
        "BASE_URL   : $base/v1",
        "MODEL      : $ModelAlias",
        "API_KEY    : $Key",
        "HEALTH     : $base/health",
        "WEB UI     : $base/",
        "",
        "curl 测试：",
        "curl $base/v1/chat/completions -H `"Content-Type: application/json`" -H `"Authorization: Bearer $Key`" -d `"{\`"model\`":\`"$ModelAlias\`",\`"messages\`":[{\`"role\`":\`"user\`",\`"content\`":\`"你好\`"}]}`"",
        "",
        "视觉测试（把图片换成你自己的路径）：",
        "curl $base/v1/chat/completions -H `"Content-Type: application/json`" -H `"Authorization: Bearer $Key`" -d `"{\`"model\`":\`"$ModelAlias\`",\`"messages\`":[{\`"role\`":\`"user\`",\`"content\`":[{\`"type\`":\`"text\`",\`"text\`":\`"图里是什么？\`"},{\`"type\`":\`"image_url\`",\`"image_url\`":{\`"url\`":\`"data:image/jpeg;base64,<BASE64>\`"}}]}]}`""
    )
    Set-Content -Path $InfoFile -Value $info -Encoding UTF8
    Ok "接入信息已写入 $InfoFile"
}

function Stop-Server {
    if (-not (Test-Path $PidFile)) { Info '没有 pid 文件，跳过'; return }
    $pidText = (Get-Content $PidFile -Raw).Trim()
    if (-not $pidText) { return }
    $p = Get-Process -Id $pidText -ErrorAction SilentlyContinue
    if (-not $p) { Info "PID $pidText 已不在运行"; Remove-Item $PidFile -Force -ErrorAction SilentlyContinue; return }
    Info "结束 PID $pidText"
    & taskkill /PID $pidText /F /T 2>$null | Out-Null
    Start-Sleep -Seconds 1
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}

# ==================== 主流程 ====================
Line ''
Line '=================================================='
Line "  Qwen3-VL-2B 一键部署  ·  $TagName"
Line "  目录：$Root"
Line '=================================================='

Ensure-Runtime
Step '准备模型'
if (-not (Test-Path $ModelDir)) { New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null }
Ensure-Model -File $ModelFile -Label $TagName 模型
Ensure-Model -File $MmprojFile -Label '视觉投影 mmproj'
Ok '模型文件齐全'

Step '生成访问密钥'
$apiKey = Ensure-ApiKey
Ok "API Key：$apiKey"

if ($SkipStart) {
    Line ''
    Ok '部署完成（按要求未启动）。双击 启动.bat 即可运行。'
    Read-Host '按回车关闭'
    exit 0
}

Start-Server -Key $apiKey

Line ''
Line '=================================================='
Ok '部署完成，服务已在跑'
Info "打开浏览器：http://127.0.0.1:$Port/  （网页里要填上面的 API Key）"
Info '停止服务：双击 停止.bat      查看状态：双击 状态.bat'
Line '=================================================='
Line ''
Read-Host '按回车关闭'
