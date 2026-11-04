# CM5-AI-M · 树莓派 MiniCPM 实机特调版

AI_Eyes 项目的**板端主线工程**：在 Raspberry Pi CM5 上离线运行 MiniCPM-V 视觉语言模型、RapidOCR、sherpa-onnx 流式语音识别与唤醒词、Piper 语音合成，通过网页测试台完成"看图问答、读文字/日期/价格、语音聊天与播报、取消与重试"的端到端实验。与 `CM5-AI/` 下的 MiniCPM 基线和 Qwen 候选同级独立存放，不修改原工程。

- 版本：**1.1.31-pi-tune**（板端 `VERSION` + `source-commit.txt` = `3f0282a0`）
- 部署根：`/home/mylin/CM5-AI-M`（板端）；仓库内为同版本源码
- Windows 测试台：`http://127.0.0.1:18090/`，经现有 SSH 隧道访问板端 Web；只使用树莓派摄像头、麦克风与 ALSA 扬声器，不回退到电脑外设
- 完整操作步骤与提问文案见仓库根目录 [树莓派MiniCPM实机测试指南.md](../树莓派MiniCPM实机测试指南.md)

## 功能

| 能力 | 实现/说明 |
|---|---|
| 场景描述、找物品、图片追问 | MiniCPM-V 4.7 1B Q4_K_M + f16 mmproj，llama.cpp CPU（8091）；单次 scene 约 155 s |
| 读文字、读标签日期 | RapidOCR PP-OCRv6 small + 字段校验 |
| 读价格、两图比价 | 只从 OCR 证据取金额/单位/条件；归属不明确时如实说明并请用户补拍或指认 |
| 多轮聊天与视觉追问 | 规则意图解析与会话语义（话题/画面可用性分离），文字与语音入口一致 |
| 语音唤醒与识别 | KWS「小冰小冰 / 小冰」+ sherpa-onnx 流式 Paraformer（16 kHz） |
| 语音播报与取消 | 常驻 Piper + 单一 ALSA 播放队列；分段、优先级、取消代次与播放回执 |
| 唤醒确认音「我在，请说」 | `models/tts/prompts/` 预制 PCM，按事件 ID 加载并校验 sha256/采样率/时长，缺失回退 Piper |
| 近黑/低质量拦截 | 实际模型输入 RGB 极值检测，过暗返回 422 并提示重拍，不调模型 |
| 收音故障自恢复 | 监督 arecord：区分短读/EOF/退出/超时，退避重开并清理旧识别上下文，状态页可见 |
| 远程外设调试与测试台 | 抓图、录放音、音量/静音、设备启停、测试批次/题号/人工评分/JSON 导出/延迟 trace |
| 主控适配 | `interaction.v0.1`：画面、场景、控制、引导事件与播放状态；未配置真实传输时明确不可用 |

## 实现

| 模块 | 职责 |
|---|---|
| `app/server.py` | Flask 服务、REST/SSE、意图分派、VLM/OCR 调用、会话记忆、状态与视觉证据注入 |
| `app/intent.py` | 规则意图：导航/空间查询转主控；聊天、视觉任务、价格、比价、播报控制与澄清 |
| `app/device_voice.py` | 录音监督与自恢复、唤醒/会话武装、取消代次、流式 ASR 对接 |
| `app/device_io.py` | ALSA 采集/播放、短读累积、故障分类、播放代次 |
| `app/streaming_asr.py`、`app/kws.py` | sherpa-onnx 流式识别与 Zipformer 唤醒词 |
| `app/visual_bridge.py`、`app/visual_context.py` | FrameProvider 与有界图片/对象会话、槽位、OCR 行与证据关联、原图 ROI 映射 |
| `app/image_quality.py` | 近黑判定与重拍提示 |
| `app/output.py`、`app/turns.py` | 输出总线（优先级/分段/取消/PlaybackStatus）与 turn 取消 |
| `app/tts_worker.py`、`app/tts_prompts.py` | 常驻 Piper 与本地预制提示音 |
| `app/main_control.py` | `interaction.v0.1` 主控适配 |
| `app/trace.py`、`app/bench_session.py` | 分项计时与测试批次/评分/导出 |
| `static/` | 原生 HTML/JS 测试台（含设备模式与测试会话页） |
| `tools/` | 基准与检查：`bench_vlm.py`、`vlm_split.py`、`bench_streaming_asr.py`、`remote_bench.ps1` 等 |

关键约束：视觉描述只陈述画面事实，不输出障碍/可通行/实际距离；价格不调用 VLM 猜；取消后旧分段不复活；收音故障可自恢复且可手动重开。

## 树莓派配置（2026-11-04 实测）

| 项目 | 实测/记录 |
|---|---|
| 主机 / 用户 | `EYES` / `mylin` |
| 系统 | Debian GNU/Linux 13 (trixie)，aarch64，内核 6.18.50+rpt-rpi-2712，Python 3.13.5（`app/.venv`） |
| 硬件 | CM5 + IO Board（历史清单）；4 GB RAM（`MemTotal` 4045 MB）；eMMC 根分区 29 GB（已用 12 GB） |
| 摄像头 | Camera Module 3 Wide（IMX708），5 fps，lores 512×384 / high 1280×720 |
| 录音 / 播放 | ReSpeaker XVF3800 USB 四麦；ALSA `plughw:CARD=Array,DEV=0`，混音器 `Array`/`PCM,0`，实测音量 62% |
| 服务 | `cm5-ai-m-test-web.service`（127.0.0.1:8090）、`cm5-ai-m-test-vlm.service`（127.0.0.1:8091）；手工启动、不 enable，`Restart=no` |
| 环境文件 | `config/pi-test.env`（600，不入 Git）：`IMAGE_SOURCE=standalone-device-test`、`VOICE_SOURCE=device`、`OUTPUT_SINK=alsa`、`MEM_BUDGET_GB=3.0`、`AEC_REFERENCE_VERIFIED=0` |
| 模型/运行时 | MiniCPM-V 4.7 1B Q4_K_M + mmproj f16，llama.cpp b12066（ctx 2048、3 线程、`-ngl 0`）；sherpa-onnx Paraformer 16 kHz；Zipformer 3.3M；RapidOCR PP-OCRv6；Piper huayan medium |
| 资源 | 交互侧 RSS 常态 1.8–2.7 GB（预算 3.0 GB，峰值上限 3 GiB）；温度 43–69 °C，无降频 |
| 备份 | `runtime/pre-v1.1.30-app.tar.gz`、`pre-v1.1.31-app.tar.gz`；安装前状态 `/home/mylin/.cm5-ai-m-task-20261103` |

## 部署与回滚

完整步骤、审计与清理流程见 [deploy/pi-test-deployment.md](deploy/pi-test-deployment.md)。

1. 本地 `git archive` 打包 `CM5-AI-M/app`、`deploy`、`README.md`、`VERSION`；
2. SFTP 上传到板端 `runtime/`，先 tar 备份现有 `app deploy README.md VERSION`；
3. `sudo systemctl stop cm5-ai-m-test-web` → 解包覆盖 → `chown -R mylin:mylin` → 清 `app/__pycache__`；
4. 更新 `VERSION`、`source-commit.txt` → `sudo systemctl start cm5-ai-m-test-web`；
5. 核对 `/api/status` 版本、`/api/health` 与两个单元状态。

回滚：停 web → 解包 `runtime/pre-v<版本>-app.tar.gz` → 恢复 VERSION/config → 启动 web。部署只改应用目录与配置，不改网络、SSH、密钥或开机启动。

## 测试

```powershell
# Windows 软件回归（工作目录 CM5-AI-M/app）
python -X utf8 -m unittest discover -s . -p "test_*.py"

# 网页交互 Node 回归
node app/test_remote_device_ui.cjs
node app/test_test_session_ui.cjs
```

当前：449 项通过 / 1 项 Piper 条件跳过（2026-11-04）。

## 实机验证状态

已在板端验证（真机 HTTP + 真机设备/进程故障注入，输入含合成图与 WAV 回放）：

- 近黑拦截 422（0.03 s）；价格读取 1.7 s / A-B 比价 3.5 s（纯 OCR，有来源、不编归属）；
- 带缓存图继续聊天不再误判看图；视觉话题追问正常；
- 真实相机 scene 155 s，答案不含障碍/可通行/距离结论；主控缺失返回 `DEPENDENCY_UNAVAILABLE`；
- SIGKILL arecord 后 1 s 内恢复监听，状态页可见故障计数与最后错误；
- 连续 20 次在途取消全部 499 `CANCELLED`，无残留请求槽；
- 预制唤醒提示音按事件 ID 播放与取消可用（清单缺失/损坏回退 Piper）；
- ASR 端点 0.9 s 较 1.2 s 提前约 0.60 s、转写一致（候选，待真人实测）；
- VLM 拆测：纯文本预填充 1.10 s vs 带图预填充约 142 s，解码 8 token 约 0.5 s，384/448/512 输入 token 数相同。

未验证：真人唤醒/接续/插话（AEC 保护模式）、真人验听、真实主控联调、Orin NX 对照、2 小时续航与散热联合验收、Qwen 板端对照。接口可访问不等于质量或性能通过。

## 资产与证据

模型、ARM 运行时、venv、缓存、真实照片与原始测试结果**不入 Git**。本机记录与证据位于 `D:\Desktop\Project\Log-Cache\CM5-AI-M-20261103\assistant-diagnostics-20261104\`（含阶段报告、部署/验证 JSON、WAV 与复现命令）。SSH 凭据与 API key 不写入仓库、日志或交付物。

## 相关文档

- [仓库根 README](../README.md)、[CHANGELOG](../CHANGELOG.md)
- [树莓派MiniCPM实机测试指南.md](../树莓派MiniCPM实机测试指南.md)
- [docs/main-control-task/02-主控与交互接口合同.md](../docs/main-control-task/02-主控与交互接口合同.md)、[05 持续执行指令与资源限制](../docs/main-control-task/05-给开发AI的持续执行指令与资源限制-20261104.md)

## 边界

空间、距离、导航与危险判断不在本工程职责内，只通过主控接口获取；全链路离线、无云端兜底；AEC 未验证时播报期间保护性屏蔽识别；视觉描述不输出行动结论；对外报告须区分真机/回放/替身证据。
