# AI_Eyes · CM5-AI

面向视障辅助的**离线多模态交互实验工程**：在树莓派 CM5 上本地运行视觉语言模型、OCR、流式语音识别、唤醒词与语音合成，通过网页测试台完成"看图问答、读文字/日期/价格、语音聊天与播报、取消与重试"的端到端实验。空间、距离、导航与危险判断不在本工程职责内，统一通过 `interaction.v0.1` 主控接口获取。

- 当前主线：`CM5-AI-M/`，版本 **1.1.31-pi-tune**，已部署到 CM5 实机（板端 `source-commit.txt` = `3f0282a0`）
- 仓库基线：`CM5-AI/MiniCPM-V4.7-1B/`（v1.1.25 MiniCPM 基线）、`CM5-AI/Qwen3.5-0.8B/`（v1.1.26-candidate 独立候选）
- 变更记录：[CHANGELOG.md](CHANGELOG.md)；阶段文档：[docs/](docs/)；实机测试步骤：[树莓派MiniCPM实机测试指南.md](树莓派MiniCPM实机测试指南.md)

## 当前状态（2026-11-04）

| 项目 | 状态 |
|---|---|
| Windows 软件回归 | 449 项通过，1 项 Piper 条件跳过（`CM5-AI-M/app`，Python 3.12 + unittest） |
| 板端部署 | v1.1.31-pi-tune 运行中；`cm5-ai-m-test-web.service` / `cm5-ai-m-test-vlm.service` 手工启动、未 enable |
| 已完成实机验证 | 真实相机抓图、麦克风采集、ALSA 播放/音量、ASR/KWS/OCR/Piper 加载；近黑拦截；价格读取与 A/B 比价（纯 OCR）；入口会话上下文；场景描述边界；主控缺失失败路径；收音故障自恢复；连续 20 次在途取消；预制唤醒提示音播放/取消；ASR 端点 0.9 s 对照；VLM 视觉/预填充/解码拆测 |
| 未验证 | 真人唤醒与接续、真人插话（AEC）、真人验听提示音、真实主控联调、Orin NX 对照、2 小时续航与腰间散热联合验收、Qwen 板端对照 |

所有实验结论区分证据来源：`live`（真机）、`replay`（文件回放）、`synthetic`（测试替身）。接口可访问不等于质量或性能通过。

## 功能

| 能力 | 输入 | 实现 | 板端状态 |
|---|---|---|---|
| 场景描述、找出口/卫生间/物品、图片追问 | 相机图 / 上传图 | MiniCPM-V 4.7 1B Q4_K_M + f16 mmproj，llama.cpp CPU 推理（8091） | 已验证；单次 scene 约 155 s |
| 读文字、读标签日期 | 相机高清图 | RapidOCR PP-OCRv6 small + 字段校验（日期/单位） | 已验证 |
| 读价格、两图比价 | 相机图 / A-B 两张图 | 只从 OCR 证据取金额/单位/条件；归属不明返回 ambiguous 并向用户澄清 | 已验证 |
| 多轮聊天与视觉追问 | 文字 / 语音 | 意图解析 `intent.py`（会话话题 + 画面可用性分离）、流式文本分段播报 | 已验证 |
| 中文语音指令与唤醒 | 麦克风 16 kHz | 设备端 KWS（"小冰小冰 / 小冰"）+ sherpa-onnx 流式 Paraformer | 已验证（合成/回放） |
| 语音播报 | 文本/事件 | 常驻 Piper huayan medium + 单一 ALSA 播放队列 + 语义分段 | 已验证 |
| 唤醒确认音「我在，请说」 | 预制 PCM | `models/tts/prompts/` 按事件 ID 加载，sha256/16 kHz/时长校验，缺失回退 Piper | 播放/取消已验证；真人验听待做 |
| 播报取消与打断 | 按钮 / 语音 / 主控事件 | turn token + 输出代次 + 流关闭；旧分段不会复活 | 20/20 在途取消失败 0 |
| 近黑/低质量画面拦截 | 相机图 | RGB 极值检测实际模型输入，过暗返回 422 并提示重拍，不调模型 | 已验证 |
| 收音故障自恢复 | arecord 异常 | 监督器区分短读/EOF/退出/超时，退避重开并清理旧识别上下文 | 已验证（SIGKILL 注入） |
| 远程外设调试 | 网页 | Picamera2 抓图、ALSA 录放音、音量/静音、设备启停 | 已验证 |
| 主控适配接口 | 主控消息 | `interaction.v0.1`：画面、场景、控制、引导事件、播放回执 | 未配置真实传输，仅失败路径 |
| 测试批次与证据 | 网页 | 批次/题号/人工评分、JSON 导出、有界原图、延迟 trace | 已具备 |

## 系统架构

```text
┌────────────────────────────── 网页测试台（static/，SSE + REST）──────────────────────────────┐
│ 拍照 / 上传 / 语音按钮 / 结果与 trace / 测试批次                                            │
└───────────────▲───────────────────────────────────────────────────────────▲─────────────────┘
                │ HTTP 127.0.0.1:8090（SSH 隧道 18090→8090）                    │
┌───────────────┴──────────────────────── app/server.py（Flask）─────────────┴─────────────────┐
│ 意图路由 intent.py ──► chat / 视觉任务 / 主控 / 播报控制                                    │
│ 视觉任务 ──► visual_bridge.py（FrameProvider + 有界图片/对象会话）──► visual_context.py     │
│              └► VLM 客户端（8091 llama-server）/ RapidOCR / 质量校验 image_quality.py        │
│ 语音输入 ──► device_voice.py（监督 + 会话武装）──► kws.py / streaming_asr.py                │
│ 输出 ──► output.py（优先级/分段/取消/播放状态）──► tts_worker.py + tts_prompts.py ──► ALSA   │
│ 生命周期 ──► turns.py（turn token/取消）  trace.py（分项计时）  main_control.py（主控适配）  │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

| 模块 | 职责 |
|---|---|
| `app/server.py` | Flask 服务、REST/SSE 接口、意图分派、VLM/OCR 调用、会话记忆、状态页 |
| `app/intent.py` | 规则意图解析：导航/空间查询转主控；聊天、视觉任务、价格、比价、播报控制与澄清 |
| `app/device_voice.py` | 录音监督与自恢复、唤醒/会话武装、取消代次、流式 ASR 对接 |
| `app/device_io.py` | ALSA 采集/播放（arecord/aplay）、短读累积、故障分类、播放代次 |
| `app/streaming_asr.py` / `app/kws.py` | sherpa-onnx 流式 Paraformer 与 Zipformer 唤醒词 |
| `app/visual_bridge.py` / `app/visual_context.py` | 帧获取与缓存、对象/槽位、OCR 行与证据关联、原始 ROI 映射 |
| `app/standalone_frame_provider.py` | 独立设备调试用 FrameProvider（仅显式选择时使用） |
| `app/image_quality.py` | 近黑/低质量画面判定与重拍提示 |
| `app/output.py` / `app/turns.py` | 输出总线（优先级、分段、取消、PlaybackStatus）与 turn 取消 |
| `app/tts_worker.py` / `app/tts_prompts.py` | 常驻 Piper 合成与本地预制提示音（校验、按 ID 取用、回退） |
| `app/main_control.py` | `interaction.v0.1` 主控适配（无真实传输时明确不可用） |
| `app/trace.py` / `app/bench_session.py` | 分项计时与测试批次/评分/导出 |
| `static/` | 原生 HTML/JS 网页测试台（`index.html`、`app-core.js`、`device-mode.js`、`test-*.js`） |
| `tools/` | 基准与检查：`bench_vlm.py`、`vlm_split.py`、`bench_streaming_asr.py`、`bench_kws.py`、`bench_tts.py`、`remote_bench.ps1` 等 |

### 实现要点

1. **会话语义与画面分离**：`SessionContext(topic, has_frame)` 独立传入文字/语音入口；网页每个请求都可能附带缓存画面，因此"有历史"或"有图"不直接等于"在看图"。
2. **价格只读 OCR 证据**：OCR 行保留原文、位置、分数、帧来源；金额/单位/促销条件按原文输出；商品与价签归属不明确时返回 `ambiguous` 并请用户补拍或指认，不调用 VLM 猜价格。
3. **开放描述边界**：视觉任务提示词只描述画面事实；流式与最终答案均拦截"无障碍/可通行/距离数字"等行动结论，替换为"照片只能用来描述画面，当前距离和通行情况需要主控确认"。
4. **近黑拦截**：以实际送入模型的编码图做 RGB 极值检测，过暗直接 422 + 语音提示重拍，避免模型对近黑画面编造物体。
5. **收音自恢复**：区分正常短读、EOF、子进程退出与读超时；监督器退避重开 arecord，重置 KWS/ASR 上下文与待处理话语，重试上限与最后错误在状态页可见；状态页分开显示"期望监听"与"实际采集"。
6. **取消一致性**：turn token + 输出 generation + 流关闭竞争修复；`/api/ask`、`/api/voice/final` 统一 499 `CANCELLED`，SSE 返回 `cancelled + done`，旧代次文字/音频回调不得复活。
7. **预制提示音**：`manifest.json` 记录 sha256、采样率、时长范围；启动时校验并加载，唤醒确认走事件 ID → 唯一 ALSA 队列，优先级/取消/主控危险提示沿用现有机制，缺失或损坏回退 Piper。
8. **ASR 端点可配**：`ASR_ENDPOINT_SILENCE` / `ASR_ENDPOINT_SILENCE_ONLY` 覆盖尾部静音与仅静音阈值（默认 1.2 s / 2.4 s）；板端回放对照显示 0.9 s 较 1.2 s 提前约 0.60 s、转写一致，作为候选待真人实测。
9. **主控职责分离**：行走、距离、空间与危险判断只走 `interaction.v0.1`；主控缺失时返回 `DEPENDENCY_UNAVAILABLE`，不用 VLM 顶替，不伪造现场。

## 树莓派配置（2026-11-04 实机核对）

| 项目 | 实测/记录 |
|---|---|
| 主机 / 用户 | `EYES` / `mylin` |
| 硬件 | Raspberry Pi CM5 Rev 1.0 + 官方 IO Board R2（历史设备清单）；四核 Cortex-A76；4 GB RAM（实测 `MemTotal` 4045 MB）；32 GB eMMC（根分区 29 GB，已用 12 GB） |
| 系统 | Debian GNU/Linux 13 (trixie)，aarch64；内核 6.18.50+rpt-rpi-2712；Python 3.13.5（工程内 `app/.venv`） |
| 摄像头 | Camera Module 3 Wide（IMX708），CSI；Picamera2 采集，5 fps；lores 512×384 / high 1280×720 |
| 录音 | ReSpeaker XVF3800 USB 四麦阵列（ALSA card `Array`）；16 kHz 单声道 S16LE |
| 播放 | ALSA `plughw:CARD=Array,DEV=0`，混音器 `Array` / `PCM,0`；实测音量 62% |
| 设备模式 | `IMAGE_SOURCE=standalone-device-test`、`VOICE_SOURCE=device`、`OUTPUT_SINK=alsa` |
| 部署根 | `/home/mylin/CM5-AI-M`；版本由 `VERSION` + `source-commit.txt` 记录（板端当前 1.1.31-pi-tune / `3f0282a0`） |
| 服务 | `cm5-ai-m-test-web.service`（Flask，127.0.0.1:8090）、`cm5-ai-m-test-vlm.service`（llama-server，127.0.0.1:8091）；**手工启动、禁止 enable**，`Restart=no` |
| 环境文件 | `config/pi-test.env`（权限 600，不入 Git）；`MEM_BUDGET_GB=3.0`、`AEC_REFERENCE_VERIFIED=0` |
| VLM | MiniCPM-V 4.7 1B `Q4_K_M` + `mmproj-model-f16.gguf`；llama.cpp b12066 ARM64；`-c 2048 -t 3 -ngl 0 --parallel 1` |
| ASR / KWS / OCR / TTS | sherpa-onnx streaming Paraformer bilingual zh-en（16 kHz，max 2 会话）；Zipformer 3.3M（阈值 0.1，尾部 blank 8）；RapidOCR PP-OCRv6 small；Piper `zh_CN-huayan-medium` 常驻 + 预制提示音 |
| 资源实测 | 交互侧 RSS 合计常态 1.8–2.7 GB（预算 3.0 GB，峰值接入上限 3 GiB）；可用内存 0.6–1.15 GB；温度 43–69 °C；`throttled=0x0` |
| 供电/散热 | 27 W USB-C 5.1 V/5 A，散热片+风扇+金属机壳（历史清单，本轮未复核） |
| 远程访问 | 现有 SSH 隧道：本机 `127.0.0.1:18090` → 板端 `127.0.0.1:8090`；8091 不对外转发 |
| 备份 | 板端 `runtime/pre-v1.1.30-app.tar.gz`、`pre-v1.1.31-app.tar.gz`；安装前状态 `/home/mylin/.cm5-ai-m-task-20261103` |

> 表中标注"实测"的项来自本机 SSH 只读审计与板端状态页；历史设备清单项来自 2026-11-02 记录，本轮未逐项复核。设备配置、环境文件与音量保持原样；未修改网络、SSH 配置、既有密钥或开机启动。

## 目录结构

```text
AI-Deployment/
├── CM5-AI-M/                  当前主线：树莓派 MiniCPM 实机特调版（v1.1.31-pi-tune）
│   ├── app/                   服务、意图、设备 I/O、语音、视觉桥接、输出与测试
│   ├── deploy/                部署脚本、systemd 单元、env 示例、清理审计
│   ├── tools/                 板端/本机基准与检查工具
│   ├── samples/ testsets/     示例图与固定测试集
│   ├── models/ runtime/ logs/ data/ test-runs/   本地资产（不入 Git）
│   ├── VERSION  README.md  启动测试台.bat
├── CM5-AI/
│   ├── MiniCPM-V4.7-1B/       MiniCPM 基线工程（v1.1.25，完整保留）
│   └── Qwen3.5-0.8B/          Qwen 独立候选工程（v1.1.26-candidate）
├── docs/                      任务说明、阶段报告、接口合同与部署运维
├── CHANGELOG.md               所有版本变更记录
└── 树莓派MiniCPM实机测试指南.md
```

Git 只分发代码、文档、示例与测试集；模型、运行时、虚拟环境、缓存、真实照片与原始测试结果不入库。

## 部署与运行

**Windows 测试台**（在仓库根目录，使用本机 SSH 隧道访问板端网页）：

```powershell
powershell -ExecutionPolicy Bypass -File CM5-AI-M/tools/remote_bench.ps1 start
# 打开 http://127.0.0.1:18090/
powershell -ExecutionPolicy Bypass -File CM5-AI-M/tools/remote_bench.ps1 stop
```

**板端部署**（详细步骤、备份与回滚见 [deploy/pi-test-deployment.md](CM5-AI-M/deploy/pi-test-deployment.md)）：

1. 本地 `git archive` 打包 `CM5-AI-M/app`、`deploy`、`README.md`、`VERSION`；
2. SFTP 上传到 `/home/mylin/CM5-AI-M/runtime/`，先 `tar` 备份现有 `app deploy README.md VERSION`；
3. `sudo systemctl stop cm5-ai-m-test-web` → 解包覆盖 → `chown -R mylin:mylin` → 清 `app/__pycache__`；
4. 更新 `VERSION` 与 `source-commit.txt` → `sudo systemctl start cm5-ai-m-test-web`；
5. 核对 `/api/status` 版本、`/api/health` 与单元状态。

回滚：停 web → 解包 `runtime/pre-v<版本>-app.tar.gz` → 恢复 VERSION/config → 启动 web。

## 接口

网页与内部调用（Flask，全部只绑 `127.0.0.1:8090`）：

| 接口 | 用途 |
|---|---|
| `POST /api/ask` | 文字提问（可附图片/帧号），返回 intent/task/source/answer/延迟与视觉证据 |
| `POST /api/eyes` | 指定视觉任务（scene/detail/text/date/find_object/compare/price 等） |
| `POST /api/voice/final` / `/api/voice` | 语音最终话语/分块流式识别 |
| `POST /api/output/cancel` | 取消普通播报（低于主控危险提示优先级时拒绝） |
| `POST /api/device/voice`、`/api/device/camera`、`/api/device/volume` | 设备启停、抓图、音量 |
| `POST /api/device/speak` | 文本或 `prompt_id` 播报（预制提示音） |
| `GET /api/status`、`/api/health`、`/api/traces` | 状态、健康、分项延迟 |
| `POST /api/session/visual/action` | 对象绑定/修正、A/B 槽位分配与交换、OCR 行关联 |
| `/api/test/*` | 测试批次、题号、评分、结果导出 |

主控接口为独立逻辑协议 `interaction.v0.1`（`capabilities.get`、`frame.acquire`、`scene.query`、`control.intent`、`target.observation`、`guidance.event`、`output.cancel`、`playback.status`、`interaction.status`），字段与错误码见 [docs/main-control-task/02-主控与交互接口合同.md](docs/main-control-task/02-主控与交互接口合同.md)。

## 测试

```powershell
# Windows 软件回归（工作目录 CM5-AI-M/app；venv 按本机实际路径）
python -X utf8 -m unittest discover -s . -p "test_*.py"

# 网页交互 Node 回归
node CM5-AI-M/app/test_remote_device_ui.cjs
node CM5-AI-M/app/test_test_session_ui.cjs
```

当前结果：449 项通过 / 1 项 Piper 条件跳过（2026-11-04）。板端验证记录按证据类别区分真机 HTTP、真机设备/进程故障注入、文件回放与测试替身；原始证据不入仓库，保存在 `D:\Desktop\Project\Log-Cache\CM5-AI-M-20261103\`。

## 资源与性能基线（CM5 实测）

| 指标 | 实测 |
|---|---|
| 交互侧内存 | RSS 合计常态 1.8–2.7 GB；本地准入预算 3.0 GB；完整子系统峰值上限 3 GiB |
| VLM 预填充 | 纯文本 72 token ≈ 1.10 s；带图 140 token ≈ 142.1–142.5 s（384/448/512 输入 token 数相同，尺寸不改变成本） |
| VLM 解码 | 8 token ≈ 0.51 s（约 64 ms/token） |
| 场景端到端 | 真实相机 scene ≈ 154.6 s（含质量检查、会话与更长 app prompt） |
| 价格/比价 | 单图价格 ≈ 1.7 s；A/B 比价 ≈ 3.5 s（纯 OCR） |
| 温度/降频 | 43–69 °C，`throttled=0x0`；v1.1.28 曾出现交换几乎用尽，重启部署后恢复 |

结论：时延瓶颈在视觉编码/图片 token 预填充，而非解码或 JPEG 编码；缩小输入尺寸不能降低耗时。优化方向（mmproj 精度、图像 token 预算、模型选择）需在内存预算与主控协调下评估。

## 边界与安全

- 本工程**不做**深度、导航、空间、体型或危险判断；这些只通过主控接口获取，缺主控时明确不可用，不用模型顶替。
- 全链路离线，无云端兜底、无新闻/联网功能；不采集、不上传用户数据。
- AEC 未验证：播报期间保护性屏蔽识别，不能据此宣称真人插话通过。
- 视觉描述只陈述画面事实；不允许输出障碍、可通行性、实际距离等行动结论。
- 对外报告须区分真机/回放/替身证据，接口可访问不等于质量或性能验收通过。

## 已知限制与待验证

1. 真人唤醒、唤醒后接续、慢语速与停顿下的 0.9 s 端点、真人插话（AEC）、预制提示音人工验听。
2. 真实主控服务与硬件联调（当前仅验证 `DEPENDENCY_UNAVAILABLE` 失败路径与协议字段）。
3. Orin NX 平台对照由主控侧核查；Qwen 板端同口径对照未部署（需额外模型传输与内存预算）。
4. 2 小时续航与腰间散热联合验收、30 分钟以上连续会话中的真人部分。
5. 近黑/低质量拦截对真实相机故障（掉线、镜头遮挡）的完整覆盖仍待补充。

## 文档

- [CHANGELOG.md](CHANGELOG.md)：版本变更记录
- [树莓派MiniCPM实机测试指南.md](树莓派MiniCPM实机测试指南.md)：测试台操作、提问文案与记录方式
- [docs/README.md](docs/README.md)：文档索引、阅读顺序与命名/维护规范
- [docs/01-范围与需求/](docs/01-范围与需求/)、[docs/03-实施记录/](docs/03-实施记录/)：职责范围与 M0–M3 阶段实施记录
- [docs/02-架构与接口/](docs/02-架构与接口/)：功能架构、设备与模型、主控接口合同
- [docs/04-部署与运维/](docs/04-部署与运维/)、[docs/05-测试与验证/](docs/05-测试与验证/)：部署运维、测试日志与报告模板
- [docs/06-计划与进度/](docs/06-计划与进度/)：工作进度与下一阶段优化方案
- [docs/main-control-task/](docs/main-control-task/)：主控下达任务包（任务书、接口合同、持续执行指令，路径按流程固定）
- [CM5-AI-M/deploy/pi-test-deployment.md](CM5-AI-M/deploy/pi-test-deployment.md)：板端部署、审计、回滚与清理
- [CM5-AI-M/README.md](CM5-AI-M/README.md)、[CM5-AI/Qwen3.5-0.8B/README.md](CM5-AI/Qwen3.5-0.8B/README.md)：子工程说明

## Git 提交规范

功能更新的提交简述使用 `vX.Y.Z 修改内容`，版本号按序延续；纯文档更新使用 `docs: 修改内容`，不增加功能版本号。简短写明实际改动，仅在 `main` 按任务阶段提交，不自动创建支线。板端部署版本以 `VERSION`、`source-commit.txt` 与该提交记录为准。

## 备份与资产

- 代码、文档、示例与测试集随 Git 分发；模型、运行时、venv、缓存、真实照片与原始结果不入库。
- 备份统一位于 `D:\Desktop\Project\Log-Cache\`；实机证据与交接草稿在同目录的 `CM5-AI-M-20261103\assistant-diagnostics-20261104\`。
- SSH 凭据、API key 不写入仓库、日志或交付物。

## 许可

内部实验工程，未附带开源许可证；未经授权不得用于生产或对外发布。
