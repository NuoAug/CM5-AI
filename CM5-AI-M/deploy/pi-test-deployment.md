# CM5-AI-M 本轮实机部署与收尾

这是专属临时测试部署模板，不代表已经安装、启动、测试通过或恢复初始系统。实际变更、安装包、模型摘要、运行命令及失败应记录到本轮执行记录。Windows测试步骤与题目见仓库根《树莓派MiniCPM实机测试指南.md》。

## 固定范围

- 登录／服务用户：`mylin`；项目根：`/home/mylin/CM5-AI-M`，与原 CM5-AI 分开。
- 主机本次授权目录标记：`.task-owner`，内容必须为 `CM5-AI-M authorized test 20261103`。
- 首次盘点：`/home/mylin/.cm5-ai-m-task-20261103/baseline.json`；已取得的本地副本保留在 Windows `D:\Desktop\Project\Log-Cache` 的本轮目录。
- Web／运行器只绑定 `127.0.0.1:8090/8091`。Windows通过已有SSH能力建立本地转发，不修改网络、SSH配置或既有密钥。
- 图像为明确的 `standalone-device-test`，语音为 `device`，输出为 `alsa`；摄像头、麦克风和扬声器均在树莓派。与整机主控正式集成不同，不把本轮独立硬件测试写成主控联调通过。
- ALSA为 `plughw:CARD=Array,DEV=0`，播放Mixer为 `Array / PCM,0`。执行前以实际设备枚举和播放控制核实；未知硬件不可用时不改用Windows外设。

## 安装前与配置

先记录现有包名／版本／manual标记、相关配置摘要／原件、设备枚举、PCM每声道音量／静音和测试端口。不得覆盖已有同名目录、环境文件或systemd单元而没有原件备份及来源说明。记录最终部署源码SHA、应用VERSION、运行器版本、权重及组件摘要。

本次模板使用现有 `mylin` 和其实际设备权限，不创建专属系统用户，不要求容器或隔离。只安装真实运行需要且尚缺的依赖；不执行系统整体升级，不修改相机／ALSA全局配置来掩盖程序问题。

将 `pi-test.env.example` 复制到本工程 `config/pi-test.env`，权限600，真实环境文件不入Git。设置随机本机API key，运行器与网页保持一致；不要将key打印到日志或文档。确认：

- Linux ARM64运行器：`runtime/llama.cpp/build/bin/llama-server`；模型路径保留 `MiniCPM-V 4.7 1B` 中的空格。
- Piper发行包解至 `runtime/piper`，实际执行文件为 `runtime/piper/piper`；不要误用Windows程序。
- Python为 `app/.venv/bin/python`；可使用真实系统相机依赖，不强制隔离。
- 原生流式ASR使用配置中的INT8 encoder／decoder及tokens；KWS、OCR、TTS模型及TTS JSON要完整离线可加载。
- `AEC_REFERENCE_VERIFIED=0`。在真实播放参考／去回声路径未验证前，播报期间持续采集但保护性屏蔽识别，不能称真人插话已通过。
- 首轮使用VLM 3线程、context2048，给ASR／OCR留出调度余量；线程、上下文及内存预算是起点；只根据本轮真实数据调整并记录，不把模板参数当性能结果。

## 手工启动与检查

仅在确认同名目标单元不存在或已备份归属之后，把两份专属unit安装到 `/etc/systemd/system`。不复用旧 `cm5-ai-*` 通用模板，不执行它们的安装脚本。新单元没有 `[Install]` 节；本次只手工start，禁止enable，也不新增开机启动、健康检查timer或网络规则。

```bash
sudo systemd-analyze verify /home/mylin/CM5-AI-M/deploy/cm5-ai-m-test-vlm.service /home/mylin/CM5-AI-M/deploy/cm5-ai-m-test-web.service
sudo systemctl daemon-reload
sudo systemctl start cm5-ai-m-test-vlm.service
sudo systemctl start cm5-ai-m-test-web.service
systemctl status cm5-ai-m-test-vlm.service cm5-ai-m-test-web.service --no-pager
```

先安装unit再start；上面的verify不等于模型或设备已通过。Web预启动对固定loopback运行器做有界健康／模型身份检查，不发送用户图片和文本，不跟随HTTP重定向；错误身份／认证失败直接拒绝。单元采用 `Restart=no`，崩溃不会被自动重启掩盖；用户重启服务前应保存错误日志并标注重启。

打开Windows转发链接后核实实际相机画面、麦克风来源、扬声器试听和音量读回。通过测试批次记录题号、实际原话、转写、答案、原图证据、人工评分、软件时延和资源采样。识别正确率仅代表已人工评题；HTTP返回／音频入队不等于物理首声。真实测试期间服务保留运行；收到用户测试完成通知后再收尾。

## 清理顺序：先留结果，再核对所有权

1. 等待本题完成，结束测试批次，导出JSON和必要原图；读取日志并整理优化选项及实现方法供用户审核。将本轮变更记录、包差异、配置原件、证据、源码版本及审计结果保存在Windows本轮Log-Cache。不要把密码、私钥或本机API key写入成果仓库。
2. 清理前运行只读 `deploy/pi-test-cleanup-audit.py`。它检查固定目录标记、专属单元摘要与状态，以及引用本工程路径的进程候选；不停止、不删除、不卸载，也不宣称系统已恢复。
3. 比对baseline和本轮安装记录，确认单元确为本次新增且未被其他用途修改，才停止这两个专属单元。它们的 `KillMode=control-group` 会停止所属子进程；不要按`python`、`llama`或端口批量杀进程。
4. 核对残余进程的实际exe／路径、启动时间和本轮记录。审计中的路径匹配只是候选，不能单凭名称停止别人的进程。Windows SSH隧道同样按本次PID及创建时间停止，保留原有SSH连接与配置。
5. 对本次新增单元，核对安装时记录的摘要及原先不存在，再移除精确文件并daemon-reload；原先存在且被本次替换的配置，只在当前仍等于本轮版本时恢复备份。不要删整目录或通配所有`cm5-*`服务。
6. 仅恢复本次改动过的播放PCM音量／静音及已明确记录的配置。不要对整张声卡粗放恢复而覆盖测试之外的录音增益或后来设置；没有原值时明确报告无法验证。
7. 只对本次实际新增的依赖做审查。先与baseline包版本／manual状态对照，再模拟移除并检查其影响；仍被原有组件或其他任务依赖的包保留并报告。禁止无边界 `autoremove`，禁止通过purge原有系统组件来追求“干净”。已升级原有依赖的情况单独说明，不自动降级。
8. 验证精确路径 `/home/mylin/CM5-AI-M` 未变为符号链接、归属和marker均匹配，并确认结果已保存后，才删除本次新增且不再需要的文件。新建任务状态目录同样须先保存基线／变更证据并核对归属，最后处理；未知文件保留并报告。Windows的独立源码与用户要求保存的指南保留。
9. 清理后复查：两单元不再活动／未enabled、所属cgroup和子进程消失、8090/8091监听归属、本次文件与临时文件残留、包差异、相关配置摘要和音量状态。必要时从已保存的审计脚本通过SSH stdin执行，避免为了审计重新留下已删除的部署目录。

审计脚本故意把 `restored_initial_state` 留为null。最终只报告实际检查项目和残留；未经对照、无原件或未验证的部分明确写“未能确认”，不得声称已恢复初始状态。不要为满足这句话重新刷机或动网络／SSH配置。
