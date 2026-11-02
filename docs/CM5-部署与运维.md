# CM5 ARM64 部署与运维（用户级，无需 sudo）

目标：Raspberry Pi Compute Module 5，四核 Cortex-A76，4 GB 内存，32 GB eMMC，纯 CPU 推理。实机为 Debian 13（trixie）+ Python 3.13.5。

本版对应 `v1.4.0`（M0 清理 + interaction.v0.1 适配）。历史实测报告见 [CM5 实机测试报告](CM5-实机测试报告-2026-11-02.md)，其中引导与 Qwen 数据只作历史证据，不再代表当前默认路径。

## 1. 目录与权限

```text
/home/mylin/cm5-ai/
├─ releases/v1.4.0-cm5/     发布：app、venv、deploy、tools、testsets、samples、manifest
├─ current -> releases/v1.4.0-cm5
├─ models/                  共享模型（只放本机发布清单中的制品）
├─ runtime/
│  ├─ llama.cpp -> llama.cpp-8ca4060/build
│  └─ piper
├─ env/cm5-ai.env           权限 600，含随机 VLM_API_KEY
├─ data/ logs/ bin/ test-runs/
```

- 用户需在 `audio`、`video`、`render` 组；`mylin` 已满足。
- `loginctl enable-linger mylin` 必须为 `yes`，否则退出登录后 user 服务会停。
- 不要把 `env` 或模型目录设为 777；`env` 600。
- 历史数据保存在 `/home/mylin/cm5-ai-preserved-20261102/`（test-runs、data、logs、脱敏 env、旧 release manifest）。

## 2. 本机唯一发布清单

部署前在 Windows 本机生成：

```powershell
cd D:\Desktop\AI-Deployment\CM5-AI
.\app\.venv\Scripts\python.exe tools\make_release_manifest.py
```

清单：[模型与工具发布清单](../docs/模型与工具发布清单.md)。当前发布模型：

| 用途 | 位置 | 说明 |
|---|---|---|
| VLM 基线 | `models/vlm/minicpm-v-4.7-1b/` | MiniCPM-V 4.7 1B Q4_K_M + mmproj f16；CM5 视觉编码约 118 s/帧，不能实时 |
| OCR | `models/ocr/pp-ocrv6-small/` | det/rec/cls 三个 ONNX |
| ASR | `models/asr/sherpa-onnx-paraformer-zh-small-2024-03-09/` | int8，按需加载 |
| TTS | `models/tts/piper-zh_CN-huayan-medium/` | 常驻 Piper；音色许可 Unknown |
| KWS | `models/kws/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/` | 唤醒词 |

不得部署：设备旧 Qwen3-VL-2B、SegFormer 引导模型、Kokoro/Matcha 对照模型。它们不在发布清单，也不得作为隐式备用。

## 3. ARM64 运行时

llama.cpp 由本机源码 `CM5-AI/runtime/llama.cpp-src`（commit `8ca4060`，0.7.0-dev）构建，不使用设备上遗留的其它源码版本。

```bash
# 源码已复制到 ~/cm5-ai/runtime/llama.cpp-src-8ca4060
cmake -S ~/cm5-ai/runtime/llama.cpp-src-8ca4060 \
  -B ~/cm5-ai/runtime/llama.cpp-8ca4060/build \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_OPENMP=ON \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=ON
cmake --build ~/cm5-ai/runtime/llama.cpp-8ca4060/build --target llama-server -j 2
ln -sfn llama.cpp-8ca4060/build ~/cm5-ai/runtime/llama.cpp
```

构建参数与二进制 SHA256 写入发布记录。Windows 的 `llama-server.exe`、`.dll`、`.venv` 不能复制到 ARM64。

Piper 使用 Pi 端 aarch64 可执行文件；不能复制 Windows 的 `piper.exe`。

## 4. 安装

在设备上的代码包目录执行：

```bash
cd ~/cm5-ai/staging
GIT_COMMIT=<commit> GIT_DIRTY=<entries> RELEASE_ID=v1.4.0-cm5 bash deploy/install_cm5_user.sh
```

脚本只做：复制代码到 `releases/<id>`、建共享目录链接、创建 venv、生成 env（已存在则保留）、写 user systemd 单元、`current` 指向新发布、只读预检。不下载模型、不升级系统包、不启动服务。

## 5. 配置

`~/cm5-ai/env/cm5-ai.env` 关键项（600）：

```bash
WEB_HOST=127.0.0.1
WEB_PORT=8090
VLM_URL=http://127.0.0.1:8091
VLM_MODEL=/home/mylin/cm5-ai/models/vlm/minicpm-v-4.7-1b/MiniCPM-V-4_7-1B-Q4_K_M.gguf
VLM_MMPROJ=/home/mylin/cm5-ai/models/vlm/minicpm-v-4.7-1b/mmproj-model-f16.gguf
VLM_API_KEY=<openssl rand -hex 16>
VLM_THREADS=3
VLM_CTX=1024
VLM_IMAGE_MAX_SIDE=512
MEM_BUDGET_GB=3.6

IMAGE_SOURCE=main_control
MAIN_CONTROL_URL=
MAIN_CONTROL_RPC_PATH=/interaction/rpc
MAIN_CONTROL_EVENTS_PATH=/interaction/events
MAIN_CONTROL_ASSET_ROOT=/home/mylin/cm5-ai/data/main-control
MAIN_CONTROL_ALLOW_FIXTURE=0

ASR_THREADS=1
ASR_IDLE_SECONDS=180
OCR_INTRA_THREADS=2
OCR_INTER_THREADS=1
KWS_THREADS=1
VOICE_SOURCE=device
OUTPUT_SINK=alsa
ALSA_INPUT=plughw:CARD=Array,DEV=0
ALSA_OUTPUT=plughw:CARD=Array,DEV=0
```

- `VLM_API_KEY` 用 `openssl rand -hex 16`，vlm 与 web 必须一致；不写日志、不入库。
- 正式模式 `IMAGE_SOURCE=main_control`；主控未接入时视觉任务与行走查询明确失败，不启用本地相机补位。
- 独立实机测试才设 `IMAGE_SOURCE=standalone-device-test`，证据单独标记，不算主控联调通过。
- `MAIN_CONTROL_ALLOW_FIXTURE` 生产必须为 0；替身只用于协议联调。
- ALSA 用稳定卡名 `CARD=Array`，不依赖会变化的 card 数字。

## 6. 启动、停止与检查

```bash
systemctl --user daemon-reload
systemctl --user start cm5-ai-vlm
# 等 /health 200，再启动 web
systemctl --user start cm5-ai-web

systemctl --user status cm5-ai-vlm cm5-ai-web --no-pager
curl -s http://127.0.0.1:8090/api/health
curl -s http://127.0.0.1:8090/api/main-control/status
curl -s http://127.0.0.1:8090/api/interaction/status | python3 -m json.tool | head -60
```

停止：

```bash
systemctl --user stop cm5-ai-web cm5-ai-vlm
```

日志：

```bash
journalctl --user -u cm5-ai-web -n 100 --no-pager
tail -f ~/cm5-ai/logs/app.log
tail -f ~/cm5-ai/logs/traces.jsonl
```

首次启动前预检：

```bash
set -a; . ~/cm5-ai/env/cm5-ai.env; set +a
~/cm5-ai/current/app/.venv/bin/python ~/cm5-ai/current/app/precheck.py --allow-busy
```

## 7. 资源与线程

| 参数 | 默认 | 说明 |
|---|---|---|
| `VLM_THREADS` | 3 | 4 线程会抢满四核；3 线程保留一核给音频与相机 |
| `VLM_CTX` | 1024 | 3.6 GB 预算下够用；加大增加内存 |
| `VLM_IMAGE_MAX_SIDE` | 512 | 按 CM5 实测调整；OCR 另走高清图 |
| `ASR_THREADS` | 1 | 按需加载，避免与 VLM 抢核 |
| `OCR_INTRA_THREADS` | 2 | OCR 并发时留余量 |
| `KWS_THREADS` | 1 | 3.3M 模型足够 |
| `MEM_BUDGET_GB` | 3.6 | 4 GB 整机减系统/桌面；超出时请求返回 503 |

VLM 线程与绑核按实测调整，不盲目绑核；Web 不绑核。

## 8. 摄像头与音频

```bash
rpicam-hello --list-cameras
arecord -l
aplay -l
```

- 正式取图由主控 FrameProvider 提供；本仓库不再独占相机做引导。
- 独立实机测试时 Picamera2 主码流 1280×720，高清帧供 OCR/VLM；不默认调用本机 `getUserMedia`。
- 麦克风：`plughw:CARD=Array,DEV=0`，16 kHz、单声道、S16_LE；设备为 reSpeaker XVF3800。
- 扬声器：`aplay -D plughw:CARD=Array,DEV=0`；Piper 输出 22.05 kHz WAV，`AlsaPlayer` 写入持续 `aplay -t raw`。
- 音量：实测 `PCM Playback Volume` 37/60 偏小，可调到 55/60；重启后需重新设置（未持久化）。
- 设备端默认半双工：播报前暂停采集，播完恢复；AEC 是否生效未确认。

## 9. 本机远程测试台

```powershell
cd D:\Desktop\AI-Deployment\CM5-AI
.\tools\remote_bench.ps1 start
.\tools\remote_bench.ps1 check
.\tools\remote_bench.ps1 stop
```

隧道：`127.0.0.1:18090 -> CM5 127.0.0.1:8090`。脚本只管理自己创建的 ssh 进程，不关闭主机密钥校验，不暴露 8091。页面用相对 API 地址；树莓派扬声器播报，浏览器不重复播放。真实图像与音频来自树莓派。

## 10. 温度、欠压与重启

- `vcgencmd measure_temp`、`get_throttled` 已接入 `/api/status` 的 `system` 字段。
- 欠压先查电源/线缆；降频先查散热/风道。
- systemd 单元 `Restart=on-failure`、`RestartSec=5`、`StartLimitIntervalSec=300`、`StartLimitBurst=5`。

## 11. 安全

- Web 与 VLM 只监听 `127.0.0.1`；不要恢复 `0.0.0.0:8090`。
- 跨机用 SSH 隧道，不把 8091 暴露到局域网。
- `env/cm5-ai.env` 权限 600，不入库；测试包只保存脱敏 env。
- VLM 启动带随机 API key，CORS 只允许本机页面。

## 12. 回滚

```bash
ln -sfn releases/v1.3.0-cm5 ~/cm5-ai/current
systemctl --user restart cm5-ai-vlm cm5-ai-web
```

但 v1.3.0 依赖已移除的 Qwen/SegFormer 与旧 `GUIDE_*` 配置；回滚只作历史复现，不是本轮正式路径。新发布失败时优先保留 `journalctl` 与 `logs/session-*`，不要删除历史 test-runs。

## 13. 测试

```bash
cd ~/cm5-ai/current
app/.venv/bin/python tools/check_cm5.py --run-id <id> --hash
app/.venv/bin/python tools/cm5_accept.py --run-id <id> --section device
app/.venv/bin/python tools/cm5_accept.py --run-id <id> --section api
app/.venv/bin/python tools/cm5_soak.py --run-id <id> --out-dir ~/cm5-ai/test-runs/<id> --duration 1800 --interval 5 --scenario idle
```

合同回归在 Windows 本机执行：

```powershell
cd D:\Desktop\AI-Deployment\CM5-AI\app
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\python.exe -m unittest discover -s . -p "test_*.py"
```

## 14. 未验证项

- 真实主控服务联调；当前只有 `test_fixture` 协议验证。
- 真人“小冰小冰”唤醒率/误唤醒率；合成 TTS 不能代替。
- 扬声器人耳听感、音量、爆音、长句连续性。
- XVF3800 固件/DSP/AEC 实际开关与双讲。
- 真实主控取图资产、真实台阶/路缘与整机联合验收。
- 真实药品/食品标签日期 OCR。
- CSI 拔插/重启后的摄像头恢复。
- 8 小时组合运行。
- Flask SSE 在 waitress/gunicorn 下的缓冲行为。
