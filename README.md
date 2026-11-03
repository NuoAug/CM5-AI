# CM5-AI

面向视障辅助实验的本地多模态测试台。当前软件版本见 [CM5-AI/MiniCPM-V4.7-1B/VERSION](CM5-AI/MiniCPM-V4.7-1B/VERSION)，完整变更记录见 [CHANGELOG.md](CHANGELOG.md)。

MiniCPM 基线完整保留在 `CM5-AI/MiniCPM-V4.7-1B/`（v1.1.25）。已建立同级独立候选 `CM5-AI/Qwen3.5-0.8B/`（v1.1.26-candidate），包含独立代码、环境、模型与运行器；目前仅完成开发机图文候选路径，默认模型仍为 MiniCPM。

当前统一使用 `main` 主线，已纳入 `voice/model-layout-v1.1.22` 的完整目录整理和后续修复。切换旧布局分支会恢复旧位置的源码，但被忽略的模型、虚拟环境和日志不会随 Git 搬移，可能出现内外混合布局。以当前分支和磁盘检查为准；本地资产保持忽略，不将它们加入源码暂存。

## 功能

| 能力 | 实现 |
|---|---|
| 看前方、找出口/卫生间/物品、场景描述、图像追问、聊天 | MiniCPM-V 4.7 1B Q4_K_M + f16 mmproj，llama.cpp CPU |
| 读文字、读取标签日期 | RapidOCR + PP-OCRv6 small ONNX；日期字段校验 |
| 中文语音指令 | 设备端 sherpa-onnx 流式 Paraformer INT8；保留整句上传基线 |
| “小冰小冰 / 小冰”唤醒 | sherpa-onnx Zipformer 3.3M KWS + 规则意图 |
| 语音播报 | 常驻 Piper huayan medium、语义分段、输出优先级与取消 |
| 远程外设调试 | Picamera2 摄像头、ALSA 录放音、音量/静音、设备启停 |
| 网页与监控 | Flask + 原生 HTML/JS，SSE、状态、延迟 trace |

独立引导模式已移除。M0 主控适配入口按 `interaction.v0.1` 恢复，默认未配置传输；没有真实主控时明确不可用。FrameProvider→有界图片／对象会话→VQA／OCR 的应用桥接已接通。真实主控传输、危险事件和播放回执闭环、真人插话仍待验收。

设备语音已接原生增量识别、最终话语路由与聊天分段播报，采集在播报时保持运行。AEC 未验证时会保护性屏蔽播放期间的识别；真人插话和真实声卡仍待验收，配置开关不代表验收通过。

本轮阶段报告：[M1 语音与取消](docs/M1-流式语音与取消-20261103.md)、[M2 图片与对象](docs/M2-图片对象与OCR-20261103.md)、[M3 Qwen独立候选与对照](docs/M3-Qwen独立候选与对照-20261103.md)。

任务依据为 [v1.1 任务文件 PR](https://github.com/NuoAug/CM5-AI/pull/3)，当前进度、接口复现和未完成项见 [M0 开发记录](docs/M0-范围与主控适配-20261103.md)。代码默认 `IMAGE_SOURCE=main-control`，不启动独立相机；`standalone-device-test` 仅供明确选择的独立设备调试，不能用于正式主控失联回退。下表保留历史设备配置，本轮没有修改整机设备配置。

## 树莓派配置

下表整理自 2026-11-02 的既有设备清单和实机记录，保留原配置；本轮未连接树莓派实时核验，也未部署或修改设备。板端当前运行版本和模型须另行检查，不能由本仓库版本推定。

| 项目 | 已有配置与记录 |
|---|---|
| 核心与载板 | Raspberry Pi Compute Module 5 Rev 1.0；官方 CM5 IO Board R2 |
| CPU / 内存 / 存储 | 四核 Cortex-A76，最高 2.4 GHz；4 GB RAM；32 GB eMMC |
| 已部署系统记录 | Debian GNU/Linux 13（trixie），aarch64；内核 6.18.50+rpt-rpi-2712；Python 3.13.5 |
| 主机 / 用户 | EYES / mylin |
| 已有应用入口 | /home/mylin/cm5-ai/current（用户部署路径） |
| 摄像头 | Camera Module 3 Wide（IMX708 / SC0874），CSI 接入，Picamera2 采集 |
| 录音 / 播放 | ReSpeaker XVF3800 USB 四麦阵列；4Ω / 5W 扬声器接音频板 SPK |
| ALSA 配置 | 输入、输出均为 plughw:CARD=Array,DEV=0；混音器 Array，播放控制 PCM,0 |
| 电源 / 散热 | 27W USB-C 5.1V/5A；散热片、风扇及金属机壳 |
| 设备模式 | IMAGE_SOURCE=standalone-device-test；VOICE_SOURCE=device；OUTPUT_SINK=alsa |
| 服务与访问 | 板端 Web 127.0.0.1:8090、VLM 127.0.0.1:8091；SSH 将本机 127.0.0.1:18090 转发到板端 Web |

沿用已有 SSH 地址、身份和密钥，在仓库根目录执行以下命令，再打开 [远程测试台](http://127.0.0.1:18090)：

```powershell
powershell -ExecutionPolicy Bypass -File CM5-AI/MiniCPM-V4.7-1B/tools/remote_bench.ps1 start
powershell -ExecutionPolicy Bypass -File CM5-AI/MiniCPM-V4.7-1B/tools/remote_bench.ps1 check
# 测试结束后关闭该工具创建的隧道
powershell -ExecutionPolicy Bypass -File CM5-AI/MiniCPM-V4.7-1B/tools/remote_bench.ps1 stop
```

仓库 deploy/ 中的 /opt/cm5-ai、cm5 用户和数字 ALSA 卡号是通用安装示例，与这台设备的已有用户部署不同。设备配置、环境文件和音量保持原样；完整模型路径、校验值和参数来源见[设备与模型](docs/设备与模型-v1.0.0.md)。

## 文档

- [功能与架构](docs/功能与架构-v1.0.0.md)：交互流程、接口和边界。
- [设备与模型](docs/设备与模型-v1.0.0.md)：保持现有树莓派设备配置、当前默认模型和校验值。
- [部署与运维](docs/CM5-部署与运维.md)：本机启动、现有设备配置保留及部署模板。
- [测试日志汇总](docs/测试日志汇总-v1.0.0.md)：当前回归证据、历史性能和未验证项。
- [测试报告模板](docs/测试报告模板.md)：后续同版本实机验收记录。
- [变更记录](CHANGELOG.md)。

## 快速启动

Windows 环境已具备模型、Python 虚拟环境和运行时后，双击 `CM5-AI/MiniCPM-V4.7-1B/启动测试台.bat`，访问 `http://127.0.0.1:8090`。独立 VLM 服务端口为 8091；关闭启动窗口会停止其管理的服务。

```powershell
powershell -ExecutionPolicy Bypass -File CM5-AI/MiniCPM-V4.7-1B/app/start.ps1
```

树莓派远程访问使用上面的 SSH 隧道命令；后续部署步骤见[部署与运维](docs/CM5-部署与运维.md)。

## 目录

```text
CM5-AI/
  MiniCPM-V4.7-1B/  当前整套 MiniCPM 运行工程
    VERSION
    启动测试台.bat
    app/            服务、网页、测试与本地虚拟环境
    deploy/         部署模板
    tools/          基准与检查工具
    testsets/       测试输入
    samples/        示例图
    models/         图文、ASR、OCR、TTS 与唤醒模型（不入库）
    runtime/        本地运行时（不入库）
    logs/           日志（不入库）
    data/           缓存（不入库）
    test-runs/      原始测试记录（不入库）
  Qwen3.5-0.8B/    独立候选完整工程；另有 config/，本地资产独立隔离
docs/               公共任务说明与报告
```

Git 仓库只分发代码、文档、示例和测试集。模型、运行时、真实凭据、虚拟环境、音频缓存及原始日志不随仓库分发。备份统一位于 `D:\Desktop\Project\Log-Cache`。

## Git 提交规范

功能更新的提交简述使用 `vX.Y.Z 修改内容`，版本号按序延续；纯文档更新使用 `docs: 修改内容`，不增加功能版本号。简短写明实际改动，按用户最新要求，仅在 main 按任务阶段提交，不自动创建支线。

## 验证

```powershell
CM5-AI/MiniCPM-V4.7-1B/app/.venv/Scripts/python.exe -m unittest discover -s CM5-AI/MiniCPM-V4.7-1B/app -p "test_*.py"
node CM5-AI/MiniCPM-V4.7-1B/app/test_remote_device_ui.cjs
```

当前回归与历史实机数据的来源及结果详见测试日志汇总。Windows、历史 Qwen3-VL 设备性能和当前 MiniCPM-V 配置分别记录。
