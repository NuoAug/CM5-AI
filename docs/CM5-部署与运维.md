# CM5 ARM64 部署与运维

目标：Raspberry Pi Compute Module 5，四核 Cortex-A76，4GB 内存，32GB eMMC，纯 CPU 推理。本文中的命令针对 Raspberry Pi OS 64-bit（Bookworm 或更新版本）；具体镜像对 CM5 的支持请以 Raspberry Pi 官方文档为准。

## 1. 目录与用户

```text
/opt/cm5-ai/
├─ app/                代码与 .venv
├─ models/             模型文件（不入库）
├─ runtime/
│  ├─ llama.cpp/       llama-server
│  └─ piper/           aarch64 piper 可执行文件
├─ logs/
└─ data/
```

服务用户默认 `cm5`，需要加入 `audio`、`video`、`render` 组。安装脚本会创建用户并设置权限。

## 2. 模型与运行时

| 文件 | 位置 | 说明 |
|---|---|---|
| MiniCPM-V 4.7 1B Q4_K_M | `models/vlm/minicpm-v-4.7-1b/MiniCPM-V-4_7-1B-Q4_K_M.gguf` | 约 640 MB |
| mmproj f16 | `models/vlm/minicpm-v-4.7-1b/mmproj-model-f16.gguf` | 约 1.0 GB |
| RapidOCR | `models/ocr/pp-ocrv6-small/` | det/rec/cls 三个 onnx |
| Paraformer small int8 | `models/asr/sherpa-onnx-paraformer-zh-small-2024-03-09/model.int8.onnx` | 约 78 MB |
| Piper 中文音色 | `models/tts/piper-zh_CN-huayan-medium/zh_CN-huayan-medium.onnx` | 约 60 MB |
| 引导分割 | `models/guide/segformer-b0-ade/model.onnx` | 约 15 MB |
| KWS | `models/kws/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/` | 约 33 MB 压缩包 |

KWS 下载：

```bash
cd /opt/cm5-ai/models/kws
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
tar xf sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
```

Piper：不能复制 Windows 的 `piper.exe`。需要 aarch64 可执行文件，放到 `runtime/piper/piper`，并把 `PIPER_BIN` 指向它。Piper 引擎 GPL-3.0；中文音色 `zh_CN-huayan-medium` 的 MODEL_CARD 许可状态不明确，商用/再分发前必须复核。

llama.cpp：用 `deploy/build_llama_cpp.sh` 在 CM5 上构建；默认只构建 `llama-server`，`-j2` 避免 4GB 内存 OOM。`LLAMA_CPP_REF` 可覆盖版本，但 MiniCPM-V 4.7 需要不低于 b11551，低于时回退 Qwen3-VL-2B。

SegFormer-B0 权重是 NVIDIA 非商业研究/评估许可，商用前必须替换为授权兼容的分割模型。

## 3. 安装

```bash
sudo CM5_AI_ROOT=/opt/cm5-ai bash /opt/cm5-ai/deploy/install_cm5.sh
```

脚本只做依赖安装、目录/用户、systemd 单元，不下载模型、不购买服务。`START_NOW=1` 时会在安装后启动服务。

安装后编辑：

```bash
sudo nano /etc/cm5-ai.env
```

必须改：

- `VLM_API_KEY`：`openssl rand -hex 16`，vlm 与 web 必须一致。
- `ALSA_INPUT` / `ALSA_OUTPUT`：先运行 `arecord -l`、`aplay -l`。
- 所有模型/运行时路径。
- `VOICE_SOURCE=device`、`GUIDE_SOURCE=device`、`OUTPUT_SINK=alsa`（真机）。

## 4. 摄像头与音频

```bash
libcamera-hello --list-cameras
arecord -l
aplay -l
```

- 摄像头：代码使用 Picamera2，低分辨率流给引导、主码流按需给 OCR/VLM；如果 lores RGB888 不被支持，会自动回退到 main 并缩放。CSI 线序、电源与相机型号请按官方文档核对。
- 麦克风：设备清单是 ReSpeaker XVF3800 USB 四麦克风阵列；先运行 `arecord -l` 看实际卡名/卡号，`ALSA_INPUT` 填 `plughw:CARD=<卡名或编号>,DEV=0`。代码使用 16 kHz、单声道、S16_LE 原始流。
- 喇叭：Piper 输出 22.05 kHz WAV；v1.3.0 设备 sink 解析 WAV 后把 PCM 写入持续 `aplay -t raw` 进程，减少逐段启动的空隙；`ALSA_OUTPUT` 同样按 `aplay -l` 填写。采样率/声道变化时会重建播放流，需在真机确认没有 underrun。
- 设备端默认半双工：播报前 `DeviceSink` 暂停采集，播完恢复；不支持播报中可靠打断。需要全双工必须做带播放参考信号的 AEC。
- 启动时后台预生成 17 条常用引导/确认短语，写入 `data/tts-cache/`；v1.3.0 默认使用常驻 Piper worker（`TTS_BACKEND=persistent`），音色只在第一次合成时加载。内存紧张时设 `TTS_IDLE_SECONDS=60~120` 空闲释放，或设 `TTS_BACKEND=subprocess` 回到每次启动 Piper 的旧路径。

## 5. 启动与检查

```bash
sudo systemctl daemon-reload
sudo systemctl start cm5-ai-vlm
sleep 5
sudo systemctl start cm5-ai-web
sudo systemctl status cm5-ai-vlm cm5-ai-web --no-pager
```

健康检查：

```bash
curl http://127.0.0.1:8090/api/health
curl http://127.0.0.1:8090/api/status | python3 -m json.tool | head -40
sudo /usr/local/bin/cm5-ai-healthcheck
```

首次启动前预检：

```bash
/opt/cm5-ai/app/.venv/bin/python /opt/cm5-ai/app/precheck.py --allow-busy
```

`--allow-busy` 用于 systemd 启动时模型服务已经占用 8091 的场景；仅跳过端口占用检查。

## 6. 资源与线程

建议起点（必须实测调整，不是结论）：

| 参数 | 起点 | 说明 |
|---|---|---|
| `MEM_BUDGET_GB` | 3.2 | 4GB 整机需给系统留约 0.8GB |
| `VLM_THREADS` | 4 | llama-server `-t` |
| `VLM_CTX` | 1024 | 单图短 prompt 够用；增大上下文会增加内存 |
| `GUIDE_THREADS` | 2 | 引导 ORT 线程 |
| `ASR_THREADS` | 1 | 按需加载，闲置释放 |
| `OCR_INTRA_THREADS` | 2 | OCR 与引导同时跑时不要占满四核 |
| `KWS_THREADS` | 1 | KWS 很小 |

注意：

- `OMP_NUM_THREADS` 不控制 ONNX Runtime 线程池；用 `GUIDE_THREADS`、`OCR_INTRA_THREADS` 等参数。
- 不要盲目绑核；先在 CM5 上比较 1/2/4 线程组合。
- 引导运行时 `/api/vlm`、`/api/ocr`、`/api/voice` 的 heavy 并发上限会降到 1；TTS light 并发上限 2。
- 不建议用 systemd `MemoryMax` 直接杀服务；应用会先返回 503。
- 不要依赖 swap 维持实时引导；4GB 下频繁 swap 会让延迟失控。

## 7. 温度、欠压与重启退避

- `vcgencmd measure_temp`、`vcgencmd get_throttled` 已接入 `/api/status` 的 `system` 字段。
- `cm5-ai-healthcheck.timer` 每 5 分钟记录 API、温度与欠压；`throttled != 0x0` 会返回非零。
- systemd 单元 `Restart=on-failure`、`RestartSec=5`、`StartLimitIntervalSec=300`、`StartLimitBurst=5`，避免无限快速重启。
- 欠压优先检查电源与线缆；降频优先检查散热与风道。

## 8. 安全

- Web 与 VLM 默认只监听 `127.0.0.1`；不要直接暴露到局域网。
- 浏览器调试若必须跨机，用 SSH 隧道：

```bash
ssh -L 8090:127.0.0.1:8090 cm5.local
```

- `logs/vlm.key`、`/etc/cm5-ai.env` 不入库；权限 600。
- VLM 启动带随机 API key，CORS 限定 `http://127.0.0.1:8090`。

## 9. 回滚

```bash
sudo systemctl disable --now cm5-ai-web cm5-ai-vlm cm5-ai-healthcheck.timer
git -C /opt/cm5-ai checkout v1.1.1
sudo /opt/cm5-ai/deploy/install_cm5.sh
```

只回滚代码时保留模型与配置；回滚 systemd 单元后记得 `daemon-reload`。

## 10. 未验证项

- Picamera2 双流配置、CSI 取帧延迟、摄像头断连恢复。
- ALSA 设备号、采样率、播放/采集切换、播报回声。
- Piper aarch64 二进制与音色加载。
- CM5 内存峰值、温度、降频、欠压、8 小时压力。
- Flask SSE 在 waitress/gunicorn 下的缓冲行为。

以上必须在真机上实测后才能写进性能结论。
