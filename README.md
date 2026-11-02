# CM5 边缘 AI 测试台

面向视障辅助场景的离线多模态服务，模型全部本地推理，不联网。Windows 本机用于开发与调优，CM5 实机用于部署与验收。

当前代码版本：v1.4.0（M0：移除独立 SegFormer 引导与本地通行判断，新增 interaction.v0.1 主控适配器与明确标记的主控替身）。本机模型与工具清单见 [模型与工具发布清单](docs/模型与工具发布清单.md)，M0 改动与验证见 [M0 职责清理与主控适配](docs/M0-职责清理与主控适配-2026-11-02.md)。早期阶段说明、CM5 实机报告与性能模板保留为历史证据，不再代表当前默认路径。

## 目标硬件

核心设备（按 2026-11-02 设备清单整理）：

| 部件 | 型号 / 规格 | 说明 |
|---|---|---|
| 核心模块 | Raspberry Pi Compute Module 5（CM5），4GB 内存、32GB eMMC | Broadcom BCM2712 四核 Cortex-A76，aarch64，推理全部走 CPU，无 CUDA / NPU 加速 |
| 载板 | 官方 Compute Module 5 IO Board R2 | 提供 USB、摄像头排线接口、网口等 |
| 摄像头 | Camera Module 3 Wide（SC0874），IMX708 传感器，约 1200 万像素 | 广角视野，用于场景描述与读字取图 |
| 麦克风 / 音频板 | ReSpeaker XVF3800 USB 四麦克风阵列 | 实物记录为 4-Mic Array V1.1，实际型号需按实物确认 |
| 扬声器 | 4Ω、5W，两线连接 | 接 XVF3800 的 SPK 接口，播放播报 |
| 套件来源 | SC1758 开发套件 | 与 IO Board R2 配套 |

配套附件：

| 附件 | 规格 | 用途 |
|---|---|---|
| 电源 | 树莓派 27W USB-C，5.1V / 5A | 为 CM5 开发套件供电 |
| 散热与机壳 | CM5 散热片、机壳风扇、黑色金属机壳 | 散热、固定并保护开发板 |
| 外置无线天线 | 规格待补 | 配合 CM5 无线连接使用；清单截图中该条说明被截断 |
| 摄像头排线 | SC1129，30 厘米，Standard-Mini | 连接摄像头与 CM5 载板 |
| USB 线 | — | 连接 ReSpeaker 音频板与 CM5 |

摄像头「支持…」、麦克风「收集声…」等描述在截图中被截断，未写入上表，需要时请补全原清单。

这台机器只有 4GB 内存，是项目所有内存优化的出发点。具体预算与实测见下文「内存预算」。

## CM5 实机部署（2026-11-02）

已完成用户级可回滚部署（`~/cm5-ai/releases/v1.3.0-cm5`，`current` 符号链接），真实 CSI 摄像头、XVF3800 麦克风、ALSA 播放、KWS/ASR/OCR/TTS/VLM 全链路联调。详细数据与人工待确认项见 [CM5 实机测试报告](docs/CM5-实机测试报告-2026-11-02.md)。

关键结论（均为 CM5 实测，不是 Windows）：

以上 v1.3.x 实机结论保留为历史记录。按主控 v1.1 任务，本轮已从当前交付移除设备上的 Qwen3-VL-2B 与 SegFormer 引导；本机唯一 VLM 基线为 MiniCPM-V 4.7 1B。
- **VLM 现状**：MiniCPM-V 4.7 视觉编码约 118 s/帧（256/320/384/512 相同），不能实时；M3 前只作质量/兼容基线，不以它宣称实时视觉能力。
- **行走判断**：全部改为主控 `scene.query` 与 `control.intent`，交互侧不再维护第二套通行算法。
- **取图**：正式模式只走主控 FrameProvider；独立设备测试必须显式 `IMAGE_SOURCE=standalone-device-test`，并单独记录，不算主控联调通过。
- **待人工确认**：扬声器实际出声、真人唤醒、真实标签、AEC 开关。

## 目录结构

```
AI-Deployment/
├─ README.md
├─ .gitignore
├─ docs/
│  ├─ CM5-边缘AI模型选型报告-2026-10-31.md   选型依据与对比结论
│  ├─ 第一阶段-视障辅助系统改造说明.md       第一阶段改动、测试与未验证项
│  ├─ 第二阶段-语音与视觉优化说明.md         流式分段、常驻 TTS、候选实测与回滚
│  ├─ CM5-部署与运维.md                     ARM64 用户级部署、systemd、健康检查、回滚
│  ├─ CM5-实机测试报告-2026-11-02.md        CM5 实测环境、优化、外设、性能与限制
│  ├─ CM5-AI-功能清单-2026-11-02.md         功能与验证口径
│  └─ CM5-性能实测报告模板.md               实测记录模板
└─ CM5-AI/
   ├─ 启动测试台.bat      双击启动；关闭窗口即停止
   ├─ app/
   │  ├─ server.py         Flask 服务，端口 8090
   │  ├─ stream_text.py    朗读规范化 + 流式语义分段
   │  ├─ tts_worker.py     常驻 Piper worker
   │  ├─ trace.py          请求时间线与 P50/P95/P99
   │  ├─ main_control.py   interaction.v0.1 适配器（握手/取图/场景/控制/事件/播放反馈）
   │  ├─ output.py         统一输出仲裁（危险抢占/去重/取消/引导事件撤销）
   │  ├─ intent.py         规则意图与结构化任务
   │  ├─ kws.py            sherpa-onnx KWS + 环形缓冲
   │  ├─ device_io.py      Picamera2 / ALSA 适配
   │  ├─ device_voice.py   设备端常开语音闭环
   │  ├─ hardware.py       Live/Replay 硬件抽象与测试接收器
   │  ├─ config/keywords.txt 唤醒词拼音 token
   │  ├─ static/index.html 测试页面
   │  ├─ start.ps1 / stop.ps1
   │  ├─ precheck.py       启动前只读预检
   │  ├─ test_*.py         单元/接口测试
   │  └─ requirements.txt
   ├─ tools/              CM5 环境快照、外设验收、并发/稳定性、VLM/SSE 探测、回放与校验
   ├─ testsets/           意图金标准与合同回归样例
   ├─ deploy/             用户级/ARM64 安装脚本、systemd、健康检查、固定依赖
   ├─ models/             模型文件（VLM / OCR / ASR / TTS / KWS）
   ├─ runtime/            llama.cpp、Piper、Python、MinGW
   ├─ samples/            测试图（猫、中文标牌、场景与盲道示例）
   ├─ test-runs/          CM5 实机原始数据（不入库）
   ├─ logs/               运行日志，不入库
   └─ data/               运行产物，不入库
```

`models/`、`runtime/`、`app/.venv/`、`logs/`、`data/` 已在 `.gitignore` 中排除，仓库只含代码与文档。

## 模型清单

| 用途 | 模型 | 格式 | 说明 |
|---|---|---|---|
| 场景 / VQA / 聊天（唯一 VLM 基线） | MiniCPM-V 4.7 1B | Q4_K_M + mmproj f16 | llama.cpp 运行，端口 8091；CM5 视觉编码约 118 s/帧，不能实时 |
| 中文 OCR | RapidOCR + PP-OCRv6_small | ONNX | det / rec / cls 三个模型 |
| 中文 TTS | Piper 中文 zh_CN-huayan-medium | ONNX VITS | 默认播报引擎；常驻 worker 一次加载，约 63 MB；音色许可未闭合 |
| 语音识别 | Paraformer-zh-small | int8 | 按需加载，闲置释放；不是原生流式 |
| 语音唤醒 | sherpa-onnx-kws-zipformer-wenetspeech-3.3M | ONNX | 中文拼音 KWS，Apache-2.0，关键词见 `app/config/keywords.txt` |

行走判断、测距、空间检查和引导不再由本仓库提供。独立 SegFormer 引导、Qwen3-VL 设备残留与相关工具已从当前交付移除，历史代码由 `archive/pre-interaction-v11-20261102` 分支保留。SmolVLM-256M 仅作 M3 对照，不进入发布清单。

模型文件不随仓库分发，需自行下载到 `models/` 对应目录。Piper 中文音色的 MODEL_CARD 标注 `License: Unknown`，商用或再分发前需自行确认授权；Piper 二进制本身的许可证也需复核。

KWS 模型下载（Apache-2.0）：

```powershell
cd CM5-AI\models\kws
Invoke-WebRequest -Uri "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2" -OutFile kws.tar.bz2
tar -xf kws.tar.bz2
Remove-Item kws.tar.bz2
```

关键词文件是拼音 token，不是普通中文文本；当前文件内容：

```text
x iǎo b īng x iǎo b īng @小冰小冰
x iǎo b īng @小冰
```

`sherpa-onnx-kws` 阈值 `KWS_SCORE=2.0`、`KWS_THRESHOLD=0.10` 只在官方真人测试音频上验证过；「小冰小冰」必须用真人录音重新标定，不能用 TTS 合成音频冒充唤醒样本。

## 处理链（v1.4.0）

```text
主控事件：主控 guidance.event / output.cancel
  → MainControlAdapter 校验版本、来源、时钟、event_id 与有效期
  → 输出仲裁（紧急 priority 0，导航 priority 1）→ 唯一播放器
  → playback.status 回报 queued/started/completed/cancelled/failed

语音交互：麦克风 → KWS「小冰小冰」→ VAD → ASR
  → 规则意图（控制动作 / 视觉任务 / 聊天 / 未知）
  → control.intent / scene.query / FrameProvider 取图
  → OCR 快路径 / VLM 慢路径 → 语义片段 → 事实守卫
  → SSE 分段 → 常驻 Piper → 连续音频播放
```

- 行走控制只提交 `start/pause/resume/end/repeat_guidance`，是否执行由主控决定；“收到请求”不等于“已完成”。
- 主控未配置或不可用时，行走查询与正式取图明确返回不可用，不启动本地旧引导补位。
- 同一结构化结果生成文字与语音，避免两者冲突；VLM/OCR 不能生成通行授权或虚构距离。
- 所有队列有界；过期帧、晚到结果、取消的旧音频、主控重启前的旧引导会被丢弃。
- 同一回答的多个片段追加播放，不重新启动整段；新回答/紧急事件才抢占。
- `done` 不再重播全文；SSE 重连按 `Last-Event-ID` 回放并用 `utterance_id|segment_id` 去重。

## 第二阶段实测摘要（历史）

以下为 Windows AMD64 开发机在 v1.3.x 的数据，不是 CM5 实测；含已移除的引导模块，只作为历史证据保留。

| 项目 | 结果 | CM5 状态 |
|---|---|---|
| Piper 常驻 vs 逐次启动（5 字短句） | 136 ms vs 522 ms | 未实测 |
| Piper 常驻（25 字 / 121 字） | 375 ms / 1747 ms | 未实测 |
| 语义分段首段 | 12–25 字，实测 24 字先播 | 逻辑已验证 |
| `/api/talk` 视觉首段 / 全文 | 3.6 s / 4.2 s（n=1；prefill 约 3.0 s） | 未实测 |
| 引导状态快路径（服务端） | 16–31 ms（ASR 已热）；ASR 冷加载后首句约 2.6 s | 未实测 |
| OCR 文本快路径（服务端） | 296–312 ms（热），首次 750 ms | 未实测 |
| MiniCPM-V 512 prefill P50 | 2.6 s（仅任务 prompt）/ 2.88–3.05 s（实际含 EYES_RULE，234 tokens） | 未实测 |
| SmolVLM-256M wall P50 | 2.37–2.71 s，质量不合格 | 未实测 |
| Matcha 中文 RTF | 0.04–0.06，但会丢 A/B 字母 | 未实测 |
| Kokoro INT8 RTF | 1.24–1.48，合成慢于播放 | 未实测 |

语音样本在 `CM5-AI/data/tts-samples/`，可直接试听；ASR 回听记录在 `CM5-AI/data/tts-asr-check.json`。ASR 只用于筛查漏字，不能替代人工音质评价。

## 启动

双击 `CM5-AI\启动测试台.bat`，或命令行执行：

```powershell
powershell -ExecutionPolicy Bypass -File "CM5-AI\app\start.ps1"
```

启动后访问 http://127.0.0.1:8090 。VLM 冷启动约 30–60 秒，状态灯变绿即可用。

停止：直接关闭启动窗口，守护进程会自动结束 VLM 与网页服务；也可运行 `CM5-AI\app\stop.ps1`。

## 页面功能

**语音交互**（默认页签，面向使用者的最短路径）

| 按钮 / 输入 | 行为 |
|---|---|
| 看前方 | 描述正前方从近到远有什么，带左/中/右与近/远；画面主要是标牌或文字时直接读字，不套用人物车辆模板 |
| 找卫生间 | 只在画面里找标识，找不到照兜底句 |
| 找出口 | 找出口标识与箭头方向 |
| 读文字 | 走 OCR，使用高清图；无文字再降级 VLM |
| 读日期 | 按生产日期 / 包装日期 / 到期日 / 保质期 / 批号定位，值非法或同字段冲突就拒猜 |
| 详细描述 | 三到五句，比「看前方」更细 |
| 找物品 | 输入物品名，报位置与遮挡 |
| 对话（聊天 / 追问） | 单一输入框：勾选「结合画面」且有画面时按追问处理，带画面与有界历史；不勾选按聊天处理，不传图、流式生成并逐句播报 |
| 停止播报 | 立即停止播放，丢弃未播放的句子，旧请求结果不再写回 |

视觉按钮在相机开启时会即时抓拍，并带 `frame_id` / `captured_at`；超过 `FRAME_MAX_AGE` 秒的画面会被拒绝并要求重拍。手动上传的静态图标记为「上传」，不会被称为当前视野。
追问没有当前帧、或帧已过期时，服务端直接返回「当前没有画面，请先拍一张或打开相机」，不调用 VLM，避免小模型拿历史回答编造画面内容。
会话历史保留最近 3 轮、最多 360 字，送进模型的上下文再截到 320 字；每次 VLM 调用记录 `prompt_tokens` / `completion_tokens` 与 `finish_reason`，方便确认是否被截断。

**语音唤醒**：点「开启语音唤醒」授权麦克风后常开监听。浏览器调试链路先跑专用 KWS（`/api/kws`），命中「小冰小冰」才送 ASR；KWS 模型不可用时退回 ASR 文本匹配。CM5 真机由 `device_voice.py` 用 arecord 常开，不经过浏览器。
先说唤醒词「小冰小冰」，再说指令，例如「小冰小冰，前面有什么」「小冰小冰，帮我找一下水杯」「小冰小冰，卫生间在哪」。
唤醒词固定为「小冰小冰」，单说「小冰」也能唤醒；同音兼容由 KWS 拼音和 ASR 文本规则共同处理，字间可带空格或标点。
只听到唤醒词时回答「我在，请说」，随后 15 秒内直接说指令即可，不必重复唤醒词；超时未说指令则回到需要唤醒词的状态。
「免唤醒窗口」由服务端按 `session_id` 计时；只说「嗯」「对对对」这类确认词不会播报、不会延长窗口，只有真正的指令才消耗窗口。
播报期间暂停采集，播放结束后再静默 0.5 秒；VAD 状态在播报前后清空，ASR 结果与最近一次播报文本相似度超过 0.7 时按回声丢弃并提示「已忽略播报回声」。
输入不足 0.6 秒、平均音量过低或 ASR 返回空文本时直接判为无效语音；同一句指令 2 秒内重复出现只按确认处理。
「停止播报」只取消普通输出队列，不会关闭主控引导，也不会永久关闭紧急入口；危险提示仍可抢占。「停一下」会结合当前状态解释：只在引导时提交结束意图，只在播报时停播报，两者同时进行时要求澄清。
旁边的「语音试听」按钮只播一句测试语，用来确认服务端 Piper 是否真的发声。

**调试工具**（二级页签）：场景识别、空间关系、图像问答、中文 OCR、语音播报，用于单独调试各模块。

**主控接口**（调试页签）：显示主控连接、实例、时钟与场景状态，可发起 `scene.query` 和五种控制意图。主控未配置时明确显示“未配置”，不会启用本地旧引导；替身连接单独标记 `fixture_allowed`。

「选择图片」内置 14 个示例：猫、中文标牌、楼梯、走廊、斑马线、桌面物品、电梯、超市货架，以及直行、圆点转向、过街、被占用、站内、破损 6 种盲道场景；示例图均为 AI 生成，仅用于界面与模型测试。下拉选择即载入，「清除图片」清空预览与已载入画面；摄像头按钮为开/关切换，载入示例或上传图片会自动关闭摄像头，避免取帧时仍用摄像头画面。

## 接口

| 接口 | 方法 | 入参 |
|---|---|---|
| `/api/status` | GET | 无；返回 VLM / OCR / ASR / TTS / KWS / 主控 / 输出 / 设备 / 内存 / CPU / 温度状态 |
| `/api/health` | GET | 无；systemd 健康检查，不做模型加载 |
| `/api/eyes` | POST | `task`、`image`、`item`、`context`、`question`、`frame_id`、`captured_at`、`request_id` |
| `/api/talk` | POST | 同上；SSE 逐段返回 `segment`，最后 `done`；视觉任务流式输出 |
| `/api/voice` | POST | `audio`（16kHz WAV base64）、`image`、`context`、`session_id`、`wake_detected`、`request_id`、`speech_end_ms`、`vad_end_ms`；免唤醒窗口由服务端维护 |
| `/api/kws` | POST | `audio`（WAV base64）；返回 `keyword` / `hits` / `latency_ms` |
| `/api/ocr` | POST | `image` |
| `/api/tts` | POST | `text`、`engine`、`speed`、`trace_id` |
| `/api/vlm` | POST | `mode`、`prompt`、`image`、`max_tokens` |
| `/api/main-control/status` | GET | 主控连接、实例、时钟、场景与最后一次错误 |
| `/api/main-control/scene` | GET | 调用 `scene.query`；只复述主控事实与有效期 |
| `/api/main-control/control` | POST | 调试用控制意图；正式控制只来自最终语音话语 |
| `/api/interaction/status` | GET | interaction.v0.1 `interaction.status`；不触发模型加载 |
| `/api/interaction/cancel` | POST | 主控 `output.cancel` 入口；返回停止/清队列/计算退出状态 |
| `/api/metrics` | GET | 最近 trace 的原始记录、P50/P95/P99、失败数、TTS worker 状态 |
| `/api/trace/event` | POST | 客户端补记 `play_start` / `underrun` / `tts_segment_failed` |
| `/api/output/stream` | GET | SSE；推送 `speech` / `cancel`；支持 `Last-Event-ID` 回放 |
| `/api/output/cancel` | POST | 浏览器调试用；停止普通播报，保留紧急输出 |
| `/api/device/status` | GET | 摄像头、ALSA、设备语音、取图来源与 sink 状态 |

`task` 取值：`scene`、`toilet`、`exit`、`item`、`text`、`date`、`detail`、`compare`、`followup`、`chat`。
图片统一用 `data:image/jpeg;base64,` 形式。

## 主控接口 interaction.v0.1

交互侧只实现接口合同与表达，不计算通行、距离或空间结论。

- 适配器：`app/main_control.py`，唯一接入口；HTTP 请求/响应 + 常驻 SSE 事件通道。
- 配置：`MAIN_CONTROL_URL` 为空时明确显示未配置；`MAIN_CONTROL_ALLOW_FIXTURE=1` 才接受 `test_fixture` 来源，生产必须为 0。
- 消息：`capabilities.get`、`frame.acquire`、`scene.query`、`control.intent`、`target.observation`、`guidance.event`、`output.cancel`、`playback.status`、`interaction.status`。
- 时效：`valid_until_mono_ms` 过期即撤销；主控实例或 `clock_id` 变化时丢弃旧场景和待播引导。
- 取图：正式模式只走 `frame.acquire`，校验 frame_id 稳定性与 asset sha256；`standalone-device-test` 只用于本轮独立实机测试，并单独记录。
- 播放：紧急事件 priority 0，导航提示 priority 1；`playback.status` 区分 queued/started/completed/cancelled/failed，设备写入基准为 `device_write`，浏览器不能报成已出声。
- 取消：`output.cancel` 分 `ordinary_turn` / `ordinary_session` / `guidance_event`；普通取消保留紧急入口，后台模型计算是否真正退出单独记录。

**主控替身**（仅协议验证）

```powershell
cd CM5-AI
app\.venv\Scripts\python.exe tools\main_control_fixture.py --port 18091 --emit-guidance
```

替身固定返回 `producer.role=test_fixture`、`evidence_mode=synthetic`，不计算真实通路或距离。启动服务时设置 `MAIN_CONTROL_URL=http://127.0.0.1:18091` 与 `MAIN_CONTROL_ALLOW_FIXTURE=1` 才能联调；它不能作为真实主控验收证据。

## 内存预算

默认预算 3.6GB（4GB 整机减去约 0.4GB 系统占用），可用环境变量覆盖：

```powershell
$env:MEM_BUDGET_GB = "3.2"
```

统计口径是模型侧提交内存：llama-server 进程、本服务进程，以及短暂运行的 Piper/aplay/arecord 子进程。超过预算时先释放按需加载的 ASR，再复查；仍超预算则 `/api/eyes`、`/api/vlm`、`/api/voice` 返回 503。

本机实测：

| 状态 | 占用 |
|---|---|
| 空闲（ASR 未加载） | 2.75–2.92 GB |
| 语音识别时（ASR 已加载） | 3.03 GB |
| 优化前（ASR 常驻 + 上下文 2048） | 3.58 GB |
| 历史：引导模型加载并推理（Windows，v1.2.0，已移除） | 进程内增量 288.5 MB（531.6 → 820.1 MB） |

前几行为 v1.0.12 及以前的实测值，当时服务端 TTS 不加载；v1.2.0 的引导模型增量只作历史记录，该模块已从当前交付移除。v1.3.0 起 TTS 默认使用常驻 Piper worker（状态页显示 `worker.alive` 与 `load_ms`）；内存紧张时可设 `TTS_IDLE_SECONDS=60~120` 空闲释放，或 `TTS_BACKEND=subprocess` 回到逐次启动。常驻 Piper 的 RSS 增量在 Windows 上约 90 MB，需在 CM5 复测。

页面右上角实时显示「模型内存 当前 / 预算」，超预算变红；旁边的「模型 CPU」显示模型侧进程占整机 CPU 的百分比，只统计 llama-server 与本服务进程（内含 OCR / ASR），按整机逻辑核心归一化，不含系统其它进程。Piper/aplay 只计入内存，不计入 CPU 读数。首次采样显示 `--`，第二次轮询起才有读数。

## 内存优化原则

前提是输出稳定、识别准确，省内存只能在不动精度的维度上做。

1. **先立基准再动参数**：每次改动前用同一组图片和同一句话记录输出与耗时，改完复测；输出一致才保留，不一致就回退。
2. **只压不影响精度的维度**：VLM 上下文 2048 降到 1024（单图加短 prompt 用不完）、VLM 输入长边 800 降到 512（实测 640 会触发切图、512 是延迟拐点）。OCR 输入保持 800，不跟着一起降，避免小字识别退化。
3. **只有低频模块才按需加载**：ASR 在语音请求时才加载，闲置 60 秒释放；VLM 每帧都要用，保持常驻；TTS 常驻 worker 可用 `TTS_IDLE_SECONDS` 控制释放，默认长驻以降低首声延迟。
4. **不靠降精度省内存**：模型维持现有的 Q4_K_M、fp32、int8 精度，不再做更激进的量化，中文识别质量是底线。
5. **宁可拒绝，不降级输出**：超出预算直接返回 503，不让模型在缺内存时返回截断或半截答案。
6. **兜底句写死**：找不到目标时只念固定兜底句，避免小模型在低资源状态下自由发挥。
7. **输出上限按实测放宽**：出现句尾被截断时，优先放宽 `max_tokens`，不要用降分辨率来换速度。
8. **数字进文档**：每次优化后的占用与延迟写进上面的内存预算表，不靠经验估算。

## 本地服务安全

- VLM 只监听 `127.0.0.1:8091`，启动时生成随机 API key（`logs/vlm.key`），服务端请求统一带 `Authorization`；未带 key 的生成接口返回 401（`/health` 保持开放）。
- llama-server 的 CORS 限定为 `http://127.0.0.1:8090`，其它网页不能直接调用本机模型。
- Flask 仍是开发服务器，只监听 `127.0.0.1:8090`；不要直接暴露到局域网。需要长期对外服务时先验证 `/api/talk` 的 SSE 在 waitress 下不被缓冲，再切换生产服务器。
- `find_slot: non-consecutive token position` 告警已实测：加 `--no-cache-prompt` 与 `--slot-prompt-similarity 0` 后仍会出现，说明与提示缓存复用无关；当前不影响输出，成因待上游确认，启动参数保持默认。
- `logs/` 与 `models/` 已在 `.gitignore` 中排除，API key 不会入库。

## 部署到树莓派时注意

完整步骤见 [CM5 部署与运维](docs/CM5-部署与运维.md)，实测见 [CM5 实机测试报告](docs/CM5-实机测试报告-2026-11-02.md)。要点：

1. **用户级部署优先**：`deploy/install_cm5_user.sh` 不需要 sudo，发布到 `~/cm5-ai/releases/<id>`，`current` 符号链接回滚；`models/`、`runtime/`、`data/`、`logs/` 共享，不随发布复制。有 sudo 时才用 `deploy/install_cm5.sh` 的 `/opt` 路径，两条路径不要同时启用。
2. **VLM 只部署本机发布清单中的 MiniCPM-V 4.7 1B**；设备上旧的 Qwen3-VL-2B 已移除。MiniCPM 视觉编码约 118 s/帧，不能实时，M3 前只作质量/兼容基线。
3. **取图来源必须显式配置**：正式 `IMAGE_SOURCE=main_control`，主控未接入时视觉任务明确失败；只有独立实机测试才用 `standalone-device-test`，并单独记录证据。
4. **输入尺寸**：开发默认 `VLM_IMAGE_MAX_SIDE=512`，按 CM5 实测调整；OCR 另走高清图，不跟着降。
5. 主控事件通道使用 `MAIN_CONTROL_URL` + SSE；`MAIN_CONTROL_ALLOW_FIXTURE=0` 为生产默认，替身只能在显式测试模式启用。
6. 麦克风与喇叭使用 ALSA：先运行 `arecord -l`、`aplay -l`；CM5 实测用 `plughw:CARD=Array,DEV=0`（reSpeaker XVF3800），不要依赖 card 数字。
7. 4GB 内存下模型侧余量有限：预算 3.6 GB，VLM 常驻后 Web 与 TTS 仍需单独核算；内存统计口径见 `interaction.status.memory`。
8. Piper 需要 Pi 端 aarch64 可执行文件，不能直接复制 Windows 的 `piper.exe`；用 `PIPER_BIN` 指向 Pi 上的 piper，`PIPER_MODEL` 指向中文音色 onnx。
9. 真机语音默认半双工：播报期间暂停采集；全双工需要 AEC，当前没有实现，XVF3800 的 AEC 开关也未确认。
10. 服务只监听 `127.0.0.1`；跨机调试用 SSH 隧道 `ssh -L 8090:127.0.0.1:8090`。

## 本轮新增能力（v1.0.8）

**使用层面**

- 视觉任务在相机开启时即时抓拍，请求带 `frame_id` 与 `captured_at`；超过 `FRAME_MAX_AGE`（默认 20 秒）的帧会被拒绝并提示重拍。手动上传的图标记为 `upload`，不会被称作当前视野。
- 多轮会话按 `session_id` 隔离，有界（3 轮 / 360 字）并带 10 分钟 TTL；换画面只把旧视觉事实标记失效，聊天历史保留。
- 新增「读日期」：只走 OCR，按关键词与空间位置定位生产日期、包装日期、到期日、保质期、批号；占位语（见瓶底）、非法日期（如 2026-19-45）、同字段冲突、置信度缺失一律拒猜，不会让 VLM 猜数字。
- 新增「详细描述」「聊天」「停止播报」。聊天走 `/api/talk` 真流式分句播报；新闻/天气等实时信息直接说明无联网查询能力。
- 分句播报不再截断，按完整句子顺序播放，可随时停止；停止会作废在途请求与待播队列。

**接口**

| 接口 | 说明 |
|---|---|
| `POST /api/talk` | 流式 SSE，逐句推送 `{sentence}`，结束推 `{done, answer}` |
| `POST /api/eyes` | 新增 `session_id`、`image_vlm`、`captured_at`；`task` 支持 `detail`、`compare`、`chat`、`date` |
| `POST /api/voice` | 新增 `session_id`、`captured_at`、`image_vlm` |

**环境变量**

| 变量 | 默认 | 说明 |
|---|---|---|
| `DATA_DIR` | `CM5-AI/data` | 覆盖数据目录，Windows 默认路径不变 |
| `DEBUG_SNAPSHOT` | `0` | 设为 `1` 才写调试快照，文件名带时间戳与线程号，避免并发串图 |
| `ASR_THREADS` | 4 | sherpa 语音识别线程数 |
| `ASR_IDLE_SECONDS` | 180 | ASR 空闲释放秒数；调大可减少反复加载，占用更多内存 |
| `MIN_VOICE_SECONDS` | 0.6 | 服务端最短有效语音时长，低于此值不送 ASR |
| `MIN_VOICE_RMS` | 0.008 | 服务端最低平均音量，低于此值判无效语音 |
| `ARMED_SECONDS` | 15 | 唤醒后免唤醒窗口秒数 |
| `VLM_THREADS` | 4（≤4 核）/ 8 | 传给 llama-server 的 `-t`，start.ps1 按逻辑核数选择；CM5 实测用 3 + 绑 0–2 核 |
| `VLM_CTX` | 1024 | 传给 llama-server 的 `-c` |
| `VLM_IMAGE_MAX_SIDE` | 512 | 服务端 VLM 输入长边；按 CM5 实测调整 |
| `VLM_API_KEY` | 每次启动随机生成 | llama-server 与本服务共用的 API key；不设置时由 start.ps1 生成并写入 `logs/vlm.key` |
| `OCR_INTRA_THREADS` / `OCR_INTER_THREADS` | 4 / 1 | RapidOCR 的 ORT 线程，注意 `OMP_NUM_THREADS` 不控制 ORT 线程池 |
| `FRAME_MAX_AGE` | 20 | 画面过期秒数 |
| `MEM_BUDGET_GB` | 3.6 | 模型侧内存预算 |
| `MEM_SAMPLE_TTL` | 10 | PSS 采样缓存秒数；读 2.5 GB 进程的 smaps_rollup 开销明显，轮询接口不应每次触发 |
| `PIPER_BIN` | `runtime/piper/piper/piper.exe` | Piper 可执行文件路径，Pi 端指向 aarch64 版 |
| `PIPER_MODEL` | `models/tts/piper-zh_CN-huayan-medium/zh_CN-huayan-medium.onnx` | Piper 中文音色模型路径 |
| `IMAGE_SOURCE` | `main_control` | `main_control` 正式 / `standalone-device-test` 独立实机测试 / `browser` 调试 |
| `MAIN_CONTROL_URL` | 空 | 主控服务基地址；为空时行走查询与正式取图明确不可用 |
| `MAIN_CONTROL_RPC_PATH` / `MAIN_CONTROL_EVENTS_PATH` | `/interaction/rpc` / `/interaction/events` | 主控请求与事件通道路径 |
| `MAIN_CONTROL_ASSET_ROOT` | 空 | 帧资产相对路径根目录；只读文件传输 |
| `MAIN_CONTROL_ALLOW_FIXTURE` | `0` | 生产必须为 0；仅显式测试替身联调时设 1 |
| `MAIN_CONTROL_TOKEN` | 空 | 主控鉴权 token；不写入仓库 |
| `KWS_MODEL_DIR` | `models/kws/sherpa-onnx-kws-...` | sherpa-onnx KWS 模型目录 |
| `KWS_KEYWORDS` | `app/config/keywords.txt` | 拼音 token 关键词文件 |
| `KWS_THREADS` | 1 | KWS ORT 线程数 |
| `KWS_SCORE` / `KWS_THRESHOLD` | 2.0 / 0.10 | KWS 提升分数/触发阈值；仅官方测试音频实验值，需真人标定 |
| `KWS_TRAILING_BLANKS` / `KWS_INT8` | 8 / 0 | 重叠关键词尾随 blank 数；是否用 int8 模型 |
| `VOICE_SOURCE` | `browser` | `browser` 调试 / `device` 真机常开语音 |
| `OUTPUT_SINK` | `browser` | `browser` SSE / `alsa` 设备端 Piper+aplay |
| `ALSA_INPUT` / `ALSA_OUTPUT` | `plughw:0,0` | 先用 `arecord -l` / `aplay -l` 确认 |
| `WEB_HOST` / `WEB_PORT` | `127.0.0.1` / `8090` | 默认只回环，跨机用 SSH 隧道 |

**播报方式（v1.0.24）**

所有播报统一走服务端 Piper（`/api/tts`，`engine=piper`）。每次合成启动一次 `piper.exe` 子进程，输出 22.05 kHz WAV 后进程退出，不在服务进程常驻模型。Piper 可执行或音色文件缺失时接口返回 503，页面仍显示文字但没有声音。语音播报期间仍然暂停麦克风采集。

合成结果写入 `data/tts-cache/`（上限 50 MB，超出按时间删最旧），固定短语和重复播报第二次为 0 ms；启动时后台预生成常用确认短语，避免第一次提示还要等 Piper 启动。前端把整段回答按 180 字分块合成，当前块播放时提前合成下一块，聊天也等回答生成完再整段播报，避免逐句请求造成的句间停顿。Piper 输出里超过 0.28 s 的停顿会压到 0.15 s，句间静音默认 0.12 s。首句仍有约 0.6–1.2 s 合成等待。
超过 60 字的回答在前端自动提到 1.15 倍速（用户手动设置的语速更高时以手动值优先）；「详细描述」提示词限三到五句，避免一次生成过长内容。

**真机 A/B 候选（均未执行，待 CM5 实测）**

tmpfs 放 `data/`（占用 RAM）、zram（CPU 换内存）、swap、限制 journald、改 fstab / gpu_mem、systemd `MemoryMax`（超限会直接杀服务）、线程数 1/2/4 组合、CPU 亲和性绑核。所有提速百分比、内存预算与预计耗时都待实测，本轮不提供数字。

**启动前预检**

```powershell
CM5-AI\app\.venv\Scripts\python.exe CM5-AI\app\precheck.py
```

只读检查 Python 版本、依赖、VLM/OCR/ASR/TTS 模型、KWS 模型、llama-server、Piper、Picamera2、arecord/aplay、主控配置、8090/8091 端口和数据目录可写，不启动服务、不加载模型、不开相机麦克风。有任何失败返回非零退出码；systemd 启动时用 `--allow-busy` 跳过端口占用检查。

**意图否定与改口（v1.0.9）**

语音说「不要打开摄像头」「换个话题」「只聊天」会直接走聊天，不触发取图；「这个是什么意思」这类词义问题同样不取景，但「眼前这个是什么意思」仍按视觉问题处理。读字任务的提示词也加了「孤立数字和年份不要扩写成生产日期、保质期或型号」。

**已知限制**

- 语音入口仍是整段等待 ASR 与推理完成再返回，只有文字聊天是真流式，不能称为完整实时语音；播报期间半双工暂停采集，不支持可靠打断。
- KWS 的「小冰小冰」尚未用真人录音标定；官方真人测试音频可检出，但 TTS 合成音频不能代替唤醒测试。
- 未在 CM5 上验证硬件音频延迟、采样率、回声消除、Picamera2 双流与麦克风边界。
- 日期标签只用构造的 OCR 行与样例图验证，未用真实药品/食品标签拍照验证。
- 播报已完全依赖服务端 Piper：可执行文件或音色缺失时没有语音，只能看页面文字。
- 行走判断依赖主控真实服务；当前只有替身协议验证，真实主控联调尚未进行。
- MiniCPM-V 在 CM5 上视觉编码约 118 s/帧，视觉问答未达到实时目标；M3 需按主控确认的来源与许可比较候选。
- Piper 中文音色训练数据许可 Unknown；商用或再分发前必须替换或取得授权。
- Flask 仍是开发服务器；SSE 在 waitress/gunicorn 下的缓冲行为未验证。

## 日志与排查

| 文件 | 内容 |
|---|---|
| `logs/app.log` | 应用事件：启动配置、eyes / voice / talk / OCR / TTS / VLM 请求的任务、意图、画面 md5、来源、延迟与回答、token 用量；主控握手/事件/取消与播放状态；KWS 命中和设备状态；按 2 MB × 3 轮转 |
| `logs/session-YYYYMMDD-HHMMSS/` | 每次启动单独一份：`web.log`、`web.err.log`、`vlm.log`、`vlm.err.log`、`stopped.txt` |
| `logs/sessions.log` | 启停事件：时间、PID、session 目录、停止原因（manual / window_closed / component_exit）与退出码 |
| `logs/vlm.key` | 本次启动生成的 llama-server API key，不入库 |
| `logs/health.log` | CM5 健康检查：API、内存预算、温度、欠压/降频 |

只保留最近 10 次会话目录。Werkzeug 访问日志已过滤 `/api/status`、`/static/*`、`/favicon.ico`、`/@vite/client`，避免轮询刷满错误输出；业务日志仍写 `app.log`。

每行请求日志都带 `frame`、`session`、`full` / `vlm` 图片 md5、`image_source`（main_control / standalone-device-test / browser）、延迟和回答预览。排查「图不对」时先比对日志里的 md5 与浏览器实际发送的图片；需要保留原图时设 `DEBUG_SNAPSHOT=1`，请求图会写到 `data/`，文件名带时间戳和线程号，默认关闭。

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
| v1.0.7 | 目录更名 edge-ai 为 CM5-AI |
| v1.0.8 | 新鲜画面与多轮会话、聊天/日期/详细描述、分句播报与打断、OCR 内存图像、线程可配置、测试与报告 |
| v1.0.9 | 意图否定与改口、读字禁止扩写、资源预检脚本 |
| v1.0.10 | 播报改为浏览器系统中文语音，服务端 TTS 默认不加载，保留降级路径 |
| v1.0.11 | 打开麦克风增加设备检测与 8 秒超时，失败时给出明确原因 |
| v1.0.12 | 唤醒修复：唤醒后 6 秒免唤醒窗口、同音词扩展、播报期间不采集、VAD 阈值下调并做音量归一、失败时回显 ASR 原文、系统语音播放加超时兜底 |
| v1.0.13 | 移除浏览器 speechSynthesis，播报统一走服务端 MeloTTS 并在启动时预加载；语音试听改为 MeloTTS 测试 |
| v1.0.14 | 测试台改为「语音交互 / 调试工具」两层页签，删除左侧说明卡与重复编号，同步页头页脚文案 |
| v1.0.15 | 聊天与追问合并为单一「对话」输入框，勾选「结合画面」走追问，否则走聊天 |
| v1.0.16 | 顶栏新增模型 CPU 占用百分比，只统计 llama-server 与本服务进程 |
| v1.0.17 | 修复 start.ps1 / stop.ps1 缺少 UTF-8 BOM 导致 Windows PowerShell 5.1 双击启动报错 |
| v1.0.18 | 唤醒同音词扩展、唤醒后等待 15 秒；语音试听改称「AI 助手」；启动窗口输入 stop 停止，移除独立停止 bat |
| v1.0.19 | 唤醒词固定为「小冰小冰」并扩展同音词，移除自定义唤醒词输入与 `wake` 参数 |
| v1.0.20 | 模型人设提示词由「AI 眼睛」改为「AI 助手」，与语音试听文案统一 |
| v1.0.21 | 重写「看前方」场景提示词：移除床猫示例，标牌/文字画面改为直接读字；start.ps1 在非交互环境改为等待进程 |
| v1.0.22 | 停止方式改为直接关闭启动窗口，由守护进程自动结束模型与网页服务，移除 stop 输入 |
| v1.0.23 | 新增 `logs/app.log`：请求日志带任务/意图/画面 md5/来源/延迟/回答，按 2 MB × 3 轮转 |
| v1.0.24 | TTS 由 MeloTTS 换成 Piper 中文神经语音（zh_CN-huayan-medium，子进程按需合成），移除 MeloTTS 引擎 |
| v1.0.25 | Piper 播报优化：磁盘缓存固定短语（上限 50 MB）+ 前端提前合成下一句，减少句间卡顿 |
| v1.0.26 | 移除 Kokoro 引擎、音色选择与模型文件，语音播报只保留 Piper |
| v1.0.27 | 播报连贯性优化：整段回答分块合成、聊天等生成完再整段播报、压缩过长停顿 |
| v1.0.28 | 新增 6 张场景示例图（楼梯/走廊/斑马线/桌面/电梯/超市），示例入口合并为下拉框 |
| v1.0.29 | 新增 6 张盲道示例图：直行、圆点转向、过街、被占用、站内、破损 |
| v1.0.30 | 示例图下拉改为选中即载入，新增「清除图片」与摄像头开/关切换 |
| v1.0.31 | 示例下拉的「选择示例图…」占位项设为不可选，只在未载入时显示 |
| v1.1.0 | 新增无盲道引导（试验）：SegFormer-B0 分割 + 保守方向决策，测试台新增引导页签 |
| v1.1.1 | 语音闭环与稳定性修复：回声过滤、服务端免唤醒窗口、无画面追问、意图归一化、VAD/ASR 门限、上下文收紧、llama API key/CORS、会话日志 |
| v1.2.0 | 第一阶段改造：服务端持续引导循环 + 连通通道/瓶颈/前方物体 + 状态机/看门狗；KWS 唤醒；结构化意图与控制动作；统一输出仲裁与危险抢占；帧标识冲突检测；VLM 计时字段修正；设备摄像头/ALSA 适配；ARM64 部署脚本、回放/基准工具与 153 项测试 |
| v1.3.0 | 第二阶段优化：流式语义分段（首段 12–25/后续 25–50 字）+ 常驻 Piper + 浏览器 `AudioContext` 连续调度 + 设备 ALSA 连续 PCM；分片段安全守卫与输出仲裁；引导状态/OCR 快路径；`trace_id` 分阶段延迟与 `/api/metrics`；TTS/VLM 候选实测与回滚开关；176 项测试 |
| v1.3.1 | CM5 实机部署与修复：Qwen3-VL-2B 替代 MiniCPM（MiniCPM 视觉编码 118 s/帧）、256 px、VLM 3 线程绑 0–2 核、取消播报立即断流、追问统一 256、复读/截断/障碍矛盾守卫、ASR 同音别名、关闭引导不关摄像头、引导模型关闭卸载、PSS 统计；新增用户级部署、实机验收、并发/稳定性/回放工具与 CM5 实测报告（历史） |
| v1.4.0 | M0：移除独立 SegFormer 引导、二维连通通道与本地导航决策及专属路由/UI/配置/工具；新增 interaction.v0.1 MainControlAdapter、主控替身、握手/取图/场景/控制/事件/取消/播放反馈；行走语音改为控制意图，取图改 FrameProvider；新增合同回归测试与唯一发布清单 |
