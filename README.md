# CM5 边缘 AI 测试台

面向视障辅助场景的离线多模态服务，模型全部本地推理，不联网。当前在 Windows 本机完成开发与调优，目标运行环境为下列硬件。

## 目标硬件

| 项目 | 规格 |
|---|---|
| 核心模块 | Raspberry Pi Compute Module 5（CM5） |
| SoC | Broadcom BCM2712，四核 ARM Cortex-A76 |
| 内存 | 4GB |
| 存储 | 32GB eMMC |
| 载板 | 官方 Compute Module 5 IO Board R2 |
| 来源 | SC1758 开发套件 |
| 架构 | aarch64，推理全部走 CPU，无 CUDA / NPU 加速 |

这台机器只有 4GB 内存，四个模型并行常驻接近 3GB，是本项目所有内存优化的出发点。具体预算与实测见下文「内存预算」。

## 目录结构

```
AI-Deployment/
├─ README.md
├─ .gitignore
├─ docs/
│  └─ CM5-边缘AI模型选型报告-2026-10-31.md   选型依据与对比结论
└─ CM5-AI/
   ├─ 启动测试台.bat      双击启动 VLM + 网页服务
   ├─ 停止测试台.bat      双击停止
   ├─ app/
   │  ├─ server.py        Flask 服务，端口 8090
   │  ├─ static/index.html 测试页面
   │  ├─ start.ps1 / stop.ps1
   │  └─ requirements.txt
   ├─ models/             四个模型，共约 2.1GB
   ├─ runtime/            llama.cpp、Python、MinGW（约 1.24GB）
   ├─ samples/            测试图（猫、中文标牌）
   ├─ logs/               运行日志，不入库
   └─ data/               运行产物，不入库
```

`models/`、`runtime/`、`app/.venv/`、`logs/`、`data/` 已在 `.gitignore` 中排除，仓库只含代码与文档。

## 模型清单

| 用途 | 模型 | 格式 | 说明 |
|---|---|---|---|
| 场景 / 空间 / VQA | MiniCPM-V 4.7 1B | Q4_K_M + mmproj f16 | llama.cpp 运行，端口 8091 |
| 中文 OCR | RapidOCR + PP-OCRv6_small | ONNX | det / rec / cls 三个模型 |
| 中文 TTS | MeloTTS 中文 | VITS fp32 | 默认播报引擎 |
| 备选 TTS | Kokoro-82M-zh | int8 | 音色多但明显更慢 |
| 语音识别 | Paraformer-zh-small | int8 | 按需加载，闲置 60 秒释放 |

模型文件不随仓库分发，需自行下载到 `models/` 对应目录。发布或再分发前请逐个复核许可证。

## 启动

双击 `CM5-AI\启动测试台.bat`，或命令行执行：

```powershell
powershell -ExecutionPolicy Bypass -File "CM5-AI\app\start.ps1"
```

启动后访问 http://127.0.0.1:8090 。VLM 冷启动约 30–60 秒，状态灯变绿即可用。

停止：双击 `CM5-AI\停止测试台.bat`。

## 页面功能

**AI 眼睛**（默认页签，面向使用者的最短路径）

| 按钮 / 输入 | 行为 |
|---|---|
| 看前方 | 描述正前方从近到远有什么，带左/中/右与近/远 |
| 找卫生间 | 只在画面里找标识，找不到照兜底句 |
| 找出口 | 找出口标识与箭头方向 |
| 读文字 | 先走 OCR，无文字再降级 VLM |
| 找物品 | 输入物品名，报位置与遮挡 |
| 追问 | 基于上一句描述回答，不带图片 |

**语音唤醒**：点「开启语音唤醒」授权麦克风后常开监听。先说唤醒词，再说指令，例如
「小V，前面有什么」「小V，帮我找一下水杯」「小V，卫生间在哪」。
唤醒词可在输入框改，保存在浏览器本地；默认「小V」兼容小薇、小微、小威、小伟等识别结果。

**其它页签**：场景识别、空间关系、图像问答、中文 OCR、语音播报，用于单独调试各模块。

## 接口

| 接口 | 方法 | 入参 |
|---|---|---|
| `/api/status` | GET | 无，返回 VLM / OCR / ASR / TTS 状态与内存 |
| `/api/eyes` | POST | `task`、`image`、`item`、`context`、`question` |
| `/api/voice` | POST | `audio`（16kHz WAV base64）、`image`、`context`、`wake` |
| `/api/ocr` | POST | `image` |
| `/api/tts` | POST | `text`、`engine`、`sid`、`speed` |
| `/api/vlm` | POST | `mode`、`prompt`、`image`、`max_tokens` |

`task` 取值：`scene`、`toilet`、`exit`、`item`、`text`、`followup`。
图片统一用 `data:image/jpeg;base64,` 形式。

## 内存预算

默认预算 3.6GB（4GB 整机减去约 0.4GB 系统占用），可用环境变量覆盖：

```powershell
$env:MEM_BUDGET_GB = "3.2"
```

统计口径是模型侧提交内存：llama-server 进程加本服务进程。超过预算时 `/api/eyes`、`/api/vlm`、`/api/voice` 返回 503。

本机实测：

| 状态 | 占用 |
|---|---|
| 空闲（ASR 未加载） | 2.75–2.92 GB |
| 语音识别时（ASR 已加载） | 3.03 GB |
| 优化前（ASR 常驻 + 上下文 2048） | 3.58 GB |

页面右上角实时显示「模型内存 当前 / 预算」，超预算变红。

## 内存优化原则

前提是输出稳定、识别准确，省内存只能在不动精度的维度上做。

1. **先立基准再动参数**：每次改动前用同一组图片和同一句话记录输出与耗时，改完复测；输出一致才保留，不一致就回退。
2. **只压不影响精度的维度**：VLM 上下文 2048 降到 1024（单图加短 prompt 用不完）、VLM 输入长边 800 降到 512（实测 640 会触发切图、512 是延迟拐点）。OCR 输入保持 800，不跟着一起降，避免小字识别退化。
3. **只有低频模块才按需加载**：ASR 在语音请求时才加载，闲置 60 秒释放；VLM、TTS 每帧都要用，保持常驻，卸载会引入一两次冷启动延迟。
4. **不靠降精度省内存**：模型维持现有的 Q4_K_M、fp32、int8 精度，不再做更激进的量化，中文识别质量是底线。
5. **宁可拒绝，不降级输出**：超出预算直接返回 503，不让模型在缺内存时返回截断或半截答案。
6. **兜底句写死**：找不到目标时只念固定兜底句，避免小模型在低资源状态下自由发挥。
7. **输出上限按实测放宽**：出现句尾被截断时，优先放宽 `max_tokens`，不要用降分辨率来换速度。
8. **数字进文档**：每次优化后的占用与延迟写进上面的内存预算表，不靠经验估算。

## 部署到树莓派时注意

1. 只复制 `models/`、`app/`、`samples/`，不要复制 Windows 版 `runtime/llama.cpp`。
2. Pi 上需要 b11551 及以上版本的 llama.cpp 才支持 MiniCPM-V 4.7；低于此版本可回退 Qwen3-VL-2B。
3. 启动参数改为 `-t 4 -c 1024 --parallel 1`（Pi OS Lite 四核）。
4. 摄像头应在 Pi 端抓图（`/picam/shot`），浏览器端抓图在局域网 IP 下会被安全策略禁用。
5. 麦克风同理：局域网 IP 访问时浏览器不允许录音，需要改成 Pi 端 ALSA 采集。
6. 4GB 内存下模型侧已接近 3GB，上 Pi 后必须实测峰值与热降频，必要时降到 3.2GB 预算并关闭非必要模块。

## 版本记录

| 版本 | 内容 |
|---|---|
| v1.0.0 | 初始化：AI 眼睛六类任务、模块化 prompt、OCR/VLM/TTS 链路 |
| v1.0.1 | 内存按提交内存统计、预算调至 3.6GB、补齐卫生间同义词收敛 |
| v1.0.2 | 语音唤醒与语音指令：Paraformer 中文 ASR、小V 唤醒、意图路由 |
| v1.0.3 | 内存优化：ASR 按需加载并空闲释放、VLM 上下文降至 1024 |
| v1.0.4 | 唤醒词可自定义、清理无用目录、补充本文档 |
| v1.0.5 | 补充 CM5 硬件配置（SC1758 套件 / IO Board R2）并置顶 |
| v1.0.6 | 补充内存优化原则：稳定性优先，只压不影响精度的维度 |
