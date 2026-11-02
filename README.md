# CM5-AI v1.1.5

面向视障辅助实验的本地多模态测试台。软件版本以 [CM5-AI/VERSION](CM5-AI/VERSION) 为准。本版以既有本地交互核心为基线，保留远程设备测试台、设备音量控制与无盲道引导页，重新从 v1.0.0 编号；v1.1.0 调整测试台布局、默认播报与意图路由，v1.1.1 拆分图片与摄像头显示区，v1.1.2 固定状态数值宽度，v1.1.3 精简语音交互入口，v1.1.4 移除图片卡片标题，v1.1.5 将图像显示区调整为 16:9。

## 功能

| 能力 | 实现 |
|---|---|
| 看前方、找出口/卫生间/物品、场景描述、图像追问、聊天 | MiniCPM-V 4.7 1B Q4_K_M + f16 mmproj，llama.cpp CPU |
| 读文字、读取标签日期 | RapidOCR + PP-OCRv6 small ONNX；日期字段校验 |
| 中文语音指令 | sherpa-onnx Paraformer small int8，整句离线识别 |
| “小冰小冰 / 小冰”唤醒 | sherpa-onnx Zipformer 3.3M KWS + 规则意图 |
| 语音播报 | 常驻 Piper huayan medium、语义分段、输出优先级与取消 |
| 无盲道引导实验 | SegFormer-B0 ADE20K + 通道规则 + 状态机 |
| 远程外设调试 | Picamera2 摄像头、ALSA 录放音、音量/静音、设备启停 |
| 网页与监控 | Flask + 原生 HTML/JS，SSE、状态、延迟 trace |

主控 interaction.v0.1 页面目前只提供未接入提示；场景/控制接口返回 409。深度传感器和 IMU 尚未接入。引导结果是单目图像实验，像素宽度未标定为真实距离，使用范围不包括过马路或上下楼梯。

文字回答支持增量分段；语音输入仍先完成整句 ASR，再执行任务，不能据此称为全双工实时对话。

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
powershell -ExecutionPolicy Bypass -File CM5-AI/tools/remote_bench.ps1 start
powershell -ExecutionPolicy Bypass -File CM5-AI/tools/remote_bench.ps1 check
# 测试结束后关闭该工具创建的隧道
powershell -ExecutionPolicy Bypass -File CM5-AI/tools/remote_bench.ps1 stop
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

Windows 环境已具备模型、Python 虚拟环境和运行时后，双击 `CM5-AI/启动测试台.bat`，访问 `http://127.0.0.1:8090`。独立 VLM 服务端口为 8091；关闭启动窗口会停止其管理的服务。

```powershell
powershell -ExecutionPolicy Bypass -File CM5-AI/app/start.ps1
```

树莓派远程访问使用上面的 SSH 隧道命令；后续部署步骤见[部署与运维](docs/CM5-部署与运维.md)。

## 目录

```text
CM5-AI/
  VERSION       软件版本
  app/          服务、网页和回归测试
  deploy/       ARM64 系统部署模板
  tools/        基准、回放和远程隧道工具
  testsets/     意图与引导案例
  samples/      测试示例图
  models/       本地模型，不入库
  runtime/      本地运行时，不入库
  logs/         运行日志，不入库
  data/         缓存与运行产物，不入库
  test-runs/    原始测试记录，不入库
docs/           本版文档和结果摘要
```

Git 仓库只分发代码、文档、示例和测试集。模型、运行时、真实凭据、虚拟环境、音频缓存及原始日志不随仓库分发。备份统一位于 `D:\Desktop\Project\Log-Cache`。

## 验证

```powershell
CM5-AI/app/.venv/Scripts/python.exe -m unittest discover -s CM5-AI/app -p "test_*.py"
node CM5-AI/app/test_remote_device_ui.cjs
node CM5-AI/app/test_guide_ui.cjs
```

当前回归与历史实机数据的来源及结果详见测试日志汇总。Windows、历史 Qwen3-VL 设备性能和当前 MiniCPM-V 配置分别记录。
