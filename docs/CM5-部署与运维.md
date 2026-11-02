# CM5 ARM64 部署与运维（用户级，无需 sudo）

目标：Raspberry Pi Compute Module 5，四核 Cortex-A76，4 GB 内存，32 GB eMMC，纯 CPU 推理。本文以 2026-11-02 实机验证的 Debian 13（trixie）+ Python 3.13.5 为准；其它镜像需自行复核包名与设备节点。

实测报告见 [CM5 实机测试报告](CM5-实机测试报告-2026-11-02.md)。本文只写部署与运维；性能数字以报告为准。

## 1. 目录、用户与权限

本机没有免密 sudo，因此采用普通用户 `mylin` 的 user systemd，不使用 `/opt` 和 `/etc`：

```text
/home/mylin/cm5-ai/
├─ releases/v1.3.0-cm5/     发布：app、venv、deploy、tools、testsets、samples、manifest
├─ current -> releases/v1.3.0-cm5
├─ models/                  共享模型（不随发布复制）
├─ runtime/
│  ├─ llama.cpp -> ~/src/llama.cpp-b11960/build
│  └─ piper     -> ~/piper
├─ env/cm5-ai.env           权限 600，含随机 VLM_API_KEY
├─ data/ logs/ bin/ test-runs/
```

- 用户需在 `audio`、`video`、`render` 组；`mylin` 已满足。
- `loginctl enable-linger mylin` 必须为 `yes`，否则退出登录后 user 服务会停。
- 不用 `chmod 777`；`env` 600，模型目录普通用户可读。

如果确实有 sudo，也可以继续用 `deploy/install_cm5.sh` 安装到 `/opt/cm5-ai` 和 `/etc/cm5-ai.env`；两条路径不要同时启用，避免抢 8090/8091 和摄像头。

## 2. 模型与运行时

| 文件 | 位置 | 说明 |
|---|---|---|
| Qwen3-VL-2B-Instruct Q4_K_M | `models/vlm/qwen3-vl-2b/Qwen3-VL-2B-Instruct-Q4_K_M.gguf` | **CM5 默认**，约 1.10 GB |
| mmproj f16 | `models/vlm/qwen3-vl-2b/mmproj-F16.gguf` | 约 819 MB |
| RapidOCR | `models/ocr/pp-ocrv6-small/` | det/rec/cls 三个 onnx |
| Paraformer small int8 | `models/asr/sherpa-onnx-paraformer-zh-small-2024-03-09/` | 约 82 MB，按需加载 |
| Piper 中文音色 | `models/tts/piper-zh_CN-huayan-medium/` | 约 63 MB，常驻 worker |
| 引导分割 | `models/guide/segformer-b0-ade/model.onnx` | 约 15 MB |
| KWS | `models/kws/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/` | 约 36 MB |
| llama-server | `runtime/llama.cpp/bin/llama-server` | b11960 源码构建，aarch64 |
| Piper | `runtime/piper/piper` | aarch64 原生，不能复制 Windows 的 `piper.exe` |

**MiniCPM-V 4.7 不作为 CM5 默认**：实测视觉编码固定约 118 s/帧，与输入尺寸无关，无法实时。模型文件可保留作候选，但 `VLM_MODEL` 必须指向 Qwen。

许可：SegFormer-B0 为 NVIDIA 非商业研究/评估许可；Piper 引擎 GPL-3.0、中文音色许可未知；商用前必须复核或替换。

发布清单 `releases/<id>/release-manifest.json` 记录 78 个文件与 8 个关键模型的 SHA256；`release/pip-freeze.txt` 记录固定依赖。

## 3. 安装

在仓库 `CM5-AI` 目录内：

```bash
cd ~/cm5-ai/staging          # 代码包解压目录
GIT_COMMIT=844103c GIT_DIRTY=9 bash deploy/install_cm5_user.sh
```

脚本只做：

1. 复制 `app/ deploy/ tools/ testsets/ samples/` 到 `releases/<id>`；
2. 建共享目录相对链接 `models/runtime/logs/data`；
3. `python3 -m venv --system-site-packages`，按 `deploy/requirements-cm5.txt` 固定版本安装；
4. 生成 `env/cm5-ai.env`（已存在则保留，不覆盖密钥）；
5. 链接 `runtime/llama.cpp`、`runtime/piper`（若存在）；
6. 写 user systemd 单元与健康检查脚本；
7. `current` 指向新发布，enable 但不启动；
8. 跑只读预检。

不下载模型、不升级系统包、不启动服务。

## 4. 配置

`~/cm5-ai/env/cm5-ai.env` 关键项（`600`）：

```bash
WEB_HOST=127.0.0.1
WEB_PORT=8090
VLM_URL=http://127.0.0.1:8091
VLM_MODEL=/home/mylin/cm5-ai/models/vlm/qwen3-vl-2b/Qwen3-VL-2B-Instruct-Q4_K_M.gguf
VLM_MMPROJ=/home/mylin/cm5-ai/models/vlm/qwen3-vl-2b/mmproj-F16.gguf
VLM_API_KEY=<openssl rand -hex 16>
VLM_THREADS=3
VLM_CTX=1024
VLM_IMAGE_MAX_SIDE=256
MEM_BUDGET_GB=3.6
GUIDE_THREADS=2
GUIDE_FPS=4.0
GUIDE_FRAME_MAX_AGE=0.6
GUIDE_WATCHDOG_SECONDS=2.5
ASR_THREADS=1
ASR_IDLE_SECONDS=180
OCR_INTRA_THREADS=2
OCR_INTER_THREADS=1
KWS_THREADS=1
VOICE_SOURCE=device
GUIDE_SOURCE=device
OUTPUT_SINK=alsa
ALSA_INPUT=plughw:CARD=Array,DEV=0
ALSA_OUTPUT=plughw:CARD=Array,DEV=0
```

- `VLM_API_KEY` 用 `openssl rand -hex 16`，vlm 与 web 必须一致；不要把 key 写进日志或测试包。
- ALSA 用稳定卡名 `CARD=Array`，不依赖会变化的 card 数字。
- `MEM_BUDGET_GB=3.6` 是模型侧 PSS 预算，不是整机 4 GB；超出时 `/api/eyes`、`/api/vlm`、`/api/voice` 返回 503。

## 5. 启动、停止与检查

```bash
systemctl --user daemon-reload
systemctl --user start cm5-ai-vlm
# 等 /health 200，再启动 web
systemctl --user start cm5-ai-web

systemctl --user status cm5-ai-vlm cm5-ai-web --no-pager
curl -s http://127.0.0.1:8090/api/health
curl -s http://127.0.0.1:8090/api/status | python3 -m json.tool | head -60
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
cat ~/cm5-ai/logs/health.log
```

首次启动前预检：

```bash
set -a; . ~/cm5-ai/env/cm5-ai.env; set +a
~/cm5-ai/current/app/.venv/bin/python ~/cm5-ai/current/app/precheck.py --allow-busy
```

## 6. 资源与线程（实测结论）

| 参数 | 默认 | 依据 |
|---|---|---|
| `VLM_THREADS` | 3 | 4 线程会抢满四核，引导出现 `model_stuck`；3 线程 + 绑 0–2 核后引导 P95 449 ms、无看门狗 |
| `VLM_CTX` | 1024 | 3.6 GB 预算下够用；增加上下文会增加内存 |
| `VLM_IMAGE_MAX_SIDE` | 256 | 256 比 512 快约 2.4 倍，答案质量在走廊/楼梯/货架对照中可接受 |
| `GUIDE_THREADS` | 2 | 引导推理 P50 243–316 ms；不要盲目加到 4 |
| `ASR_THREADS` | 1 | 按需加载，避免与引导抢核 |
| `OCR_INTRA_THREADS` | 2 | OCR 与引导并发时留余量 |
| `KWS_THREADS` | 1 | 3.3M 模型足够 |
| `MEM_BUDGET_GB` | 3.6 | 4 GB 整机减系统/桌面约 0.4–1.1 GB |

systemd 单元已把 llama-server 绑到 0–2 核：

```text
ExecStart=... taskset -c 0-2 .../llama-server ... -t 3 ...
```

**不要**把 Web 也绑到 3 核：实测引导和 VLM HTTP/ORT 会抢同一核，出现 `model_stuck`。Web 不绑核，由调度器把引导放到空闲核。

引导关闭后会卸载 ONNX 会话，Web PSS 从约 392 MB 降到 238 MB；下次开始按需重载。

## 7. 摄像头与音频

```bash
rpicam-hello --list-cameras
arecord -l
aplay -l
```

- 摄像头：Picamera2 双流，主码流 1280×720，lores 512×384；引导读 lores，OCR/VLM 用 `/api/camera/still` 高清图。实测 lores 可用，不需要回退 main。
- 麦克风：`plughw:CARD=Array,DEV=0`，16 kHz、单声道、S16_LE；设备为 reSpeaker XVF3800（USB bcdDevice 2.06）。
- 扬声器：`aplay -D plughw:CARD=Array,DEV=0`；Piper 输出 22.05 kHz WAV，`AlsaPlayer` 写入持续 `aplay -t raw`。
- 音量：实测 `PCM Playback Volume` 37/60 偏小，已调到 55/60；可用 `amixer -c 2 cset numid=5 55,55` 调整，重启后需重新设置（未做持久化）。
- 设备端默认半双工：播报前暂停采集，播完恢复；不支持播报中可靠打断。AEC 是否生效未确认。

## 8. 温度、欠压与重启

- `vcgencmd measure_temp`、`get_throttled` 已接入 `/api/status` 的 `system` 字段。
- 实测空闲 43–53 °C、引导 60–66 °C、VLM 冷图最高 67.5 °C；`throttled=0x0`，2400 MHz 未降频。
- 欠压先查电源/线缆；降频先查散热/风道。
- systemd 单元 `Restart=on-failure`、`RestartSec=5`、`StartLimitIntervalSec=300`、`StartLimitBurst=5`。

## 9. 安全

- Web 与 VLM 只监听 `127.0.0.1`；旧部署的 `0.0.0.0:8090` 已停止，不要再恢复。
- 跨机浏览器调试用 SSH 隧道：

```bash
ssh -i C:\Users\DJH08\.ssh\trae_com5linux_ed25519 -L 8090:127.0.0.1:8090 mylin@192.168.2.63
# 本机打开 http://127.0.0.1:8090
```

- `logs/vlm.key`、`env/cm5-ai.env` 权限 600，不入库；测试包只保存脱敏 env。
- VLM 启动带随机 API key，CORS 只允许 `http://127.0.0.1:8090`。

## 10. 回滚

### 10.1 回滚到旧部署（2026-11-01）

```bash
systemctl --user disable --now cm5-ai-web cm5-ai-vlm cm5-ai-healthcheck.timer
cp ~/cm5-ai/test-runs/20261102T1430+0800_preflight/old-config/llama-vlm.service ~/.config/systemd/user/
cp ~/cm5-ai/test-runs/20261102T1430+0800_preflight/old-config/edge-ai.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now llama-vlm edge-ai
```

旧目录 `~/cm5-edge-ai`、旧模型和旧配置均保留。

### 10.2 回滚到另一个新发布

```bash
ln -sfn releases/<旧版本> ~/cm5-ai/current
systemctl --user restart cm5-ai-vlm cm5-ai-web
```

模型与 `env` 不改；只换代码。失败时保留 `journalctl` 与 `logs/session-*`。

## 11. 测试与回放

实机：

```bash
cd ~/cm5-ai/current
app/.venv/bin/python tools/check_cm5.py --run-id <id> --hash
app/.venv/bin/python tools/cm5_accept.py --run-id <id> --section device
app/.venv/bin/python tools/cm5_accept.py --run-id <id> --section api
app/.venv/bin/python tools/test_voice_scenarios.py --run-id <id> --out-dir ~/cm5-ai/test-runs/<id>
app/.venv/bin/python tools/test_concurrency.py --image samples/scene-elevator.jpg --out ~/cm5-ai/test-runs/<id>/results/concurrency.json
app/.venv/bin/python tools/test_guide_variants.py --image samples/scene-tactile-straight.jpg --out ~/cm5-ai/test-runs/<id>/results/guide-variants.json
app/.venv/bin/python tools/cm5_soak.py --run-id <id> --out-dir ~/cm5-ai/test-runs/<id> --duration 1800 --interval 2 --scenario guide
app/.venv/bin/python tools/soak_analysis.py --run-dir ~/cm5-ai/test-runs/<id>
```

Windows 回放（离线，不连生产服务）：

```powershell
cd D:\Desktop\AI-Deployment\CM5-AI
.\app\.venv\Scripts\python.exe tools\replay_run.py --run-dir test-runs\<id>
.\app\.venv\Scripts\python.exe tools\verify_run.py test-runs\<id>
```

## 12. 未验证项

- 真人“小冰小冰”唤醒率/误唤醒率；合成 TTS 不能代替。
- 扬声器人耳听感、音量、爆音、长句连续性；回环录音未证明 1 kHz 到达麦克风（可能被 AEC 抵消）。
- XVF3800 固件/DSP/AEC 实际开关。
- 真实台阶/路缘/玻璃/悬空物、暗光、逆光、遮挡下的引导。
- 真实药品/食品标签日期 OCR。
- CSI 拔插/重启后的摄像头恢复。
- 8 小时组合运行；本轮只完成 30 分钟连续引导。
- Flask SSE 在 waitress/gunicorn 下的缓冲行为。
