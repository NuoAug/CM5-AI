# CM5-AI v1.0.0 部署与运维

## 现有设备配置

保持 EYES 主机、mylin 用户、/home/mylin/cm5-ai/current 用户部署路径、IMX708 摄像头、ReSpeaker XVF3800 与 ALSA Array 设备。具体清单见[设备与模型](设备与模型-v1.0.0.md)。

本次整理只修改开发机的代码版本标识和文档，没有 SSH 部署、重启设备服务或修改树莓派环境文件。既有服务名、环境文件、密钥、卡名和目录应以设备上当前配置为准。

## Windows 本地启动

需预先具备 app/.venv/Scripts/python.exe、模型和运行时。双击 CM5-AI/MiniCPM-V4.7-1B/启动测试台.bat，或在仓库根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File CM5-AI/MiniCPM-V4.7-1B/app/start.ps1
```

网页 127.0.0.1:8090，VLM 127.0.0.1:8091。关闭启动窗口或执行 app/stop.ps1 停止其管理的进程。日志位于 logs/session-*，应用日志在 logs/app.log。

启动脚本为 VLM 生成 API key。logs/vlm.key 不入库，也不复制到公开报告。

## 远程测试台

沿用已有 SSH 身份和设备地址，工具只管理自身创建的隧道：

```powershell
powershell -ExecutionPolicy Bypass -File CM5-AI/MiniCPM-V4.7-1B/tools/remote_bench.ps1 start
powershell -ExecutionPolicy Bypass -File CM5-AI/MiniCPM-V4.7-1B/tools/remote_bench.ps1 check
powershell -ExecutionPolicy Bypass -File CM5-AI/MiniCPM-V4.7-1B/tools/remote_bench.ps1 stop
```

本机访问 127.0.0.1:18090，转发到设备 127.0.0.1:8090。参数 HostName/UserName/KeyPath 可显式传入，默认记录仅供当前环境使用。浏览器授权的本机麦克风和设备麦克风是不同来源；设备模式应通过页面的树莓派控制测试。

## 后续向现有设备更新

更新前先记录当前服务单元、环境文件位置、工作目录、版本和模型 SHA256，备份代码及配置。只替换对应的应用代码与本版模型，沿用已有用户、路径、ALSA_INPUT/OUTPUT、音量混音器和网络设置；不要直接用通用安装模板覆盖用户部署。

由于本版模型与历史板端默认 Qwen 不同，应复核 VLM_MODEL/VLM_MMPROJ 指向 MiniCPM-V 4.7 1B 本版文件，并对该组合重新实机验收。此步骤是未来部署要求，本轮尚未在板端执行。

本版未启用原生流式 ASR，主控接口已移除。不要将旧发布包覆盖回新基线，或只修改标签而继续沿用旧模型性能结论。

## 通用新机部署模板

deploy/ 保留系统级安装模板，使用 /opt/cm5-ai 和 cm5 系统用户。它适合另行规划的新安装，与现有 /home/mylin 用户部署分开。引导相关模板参数已清理；其余模板参数和实际设备配置保持原样。

- install_cm5.sh：aarch64 检查、依赖安装、虚拟环境、服务用户及 systemd 单元。
- cm5-ai.env.example：模型路径、线程和设备示例；实际 ALSA 应使用 Array 卡名。
- build_llama_cpp.sh：默认 b11551，CPU 构建 llama-server，默认 -j2。是否支持本版权重须实测；不要把脚本旧 Qwen 回退提示视为本版默认模型。
- cm5-ai-vlm.service / cm5-ai-web.service：VLM 与 Flask 服务、重启退避。
- healthcheck.service/timer：周期健康检查。

模型不自动下载；Piper 和 llama-server 必须为 aarch64 可执行文件，不能复制 Windows .exe。Piper 实际路径通过 PIPER_BIN 指定；llama-server 可执行路径须与服务文件及构建输出匹配。

## 检查与日志

```bash
curl http://127.0.0.1:8090/api/health
curl http://127.0.0.1:8090/api/status
curl http://127.0.0.1:8090/api/metrics
arecord -l
aplay -l
amixer -c Array sget PCM,0
```

在设备已有 Python 环境中运行 app/precheck.py --allow-busy；当前工具 tools/check_cm5.py 可作只读环境检查。具体服务日志通过已有服务单元的 journalctl 查询。

/api/status.version 应与 CM5-AI/MiniCPM-V4.7-1B/VERSION 一致。引导模式已移除，健康检查不再依赖分割模型。状态正常不能替代模型精度、扬声器真实出声或长期稳定性验收。

## 数据与备份

- logs/：应用、会话和 trace；原始记录不入库。
- test-runs/：测试批次、环境、输出及清单；保留原批次名称。
- data/：缓存、音频样本及基准产物，不入库。
- 本次所有备份统一位于 D:\Desktop\Project\Log-Cache；旧文档与 Git bundle 在 CM5-AI-pre-v1.0.0-20261102/。
- 公开测试总结只保存必要统计与可复现说明，不提交 key、真实环境文件和模型二进制。

## 当前验收状态

本机逻辑/UI 回归及历史结果详见[测试日志汇总](测试日志汇总-v1.0.0.md)。当前 MiniCPM 默认组合尚未在树莓派重新验收；历史 Qwen 的 30 分钟记录有 model_stuck 采样，不能表述为本版长期稳定性已通过。后续实机测试使用[报告模板](测试报告模板.md)。
