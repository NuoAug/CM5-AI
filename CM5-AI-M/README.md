# CM5-AI-M 树莓派 MiniCPM 实机特调版

版本：1.1.31-pi-tune。与原CM5-AI同级独立存放；原MiniCPM和Qwen未修改。

Windows测试台：[http://127.0.0.1:18090/](http://127.0.0.1:18090/)，通过本机专属SSH隧道访问树莓派。仅使用树莓派摄像头、麦克风、ALSA扬声器，浏览器外设不回退。包括真实音量调节、测试批次／题号／人工评分、结果JSON导出及有界原图证据。

完整步骤和提问文案见仓库根目录 [树莓派MiniCPM实机测试指南](../树莓派MiniCPM实机测试指南.md)。

部署根/home/mylin/CM5-AI-M，两服务cm5-ai-m-test-vlm/web.service手工启动、不enable；运行器b12066 ARM64，CPU3线程ctx2048。ASR原生流式、KWS／OCR／Piper均本地运行。AEC未验证，播报时持续采集但识别保护，不宣称真人插话通过。

已验证真实HTTP访问、相机抓图、麦克风采集和ASR模型加载、ALSA合成播放软件路径、音量读回与恢复。权重校验修复后，新图实测约155.7秒，当前相机画面近黑且模型仍猜测物体，需要开灯／检查遮挡后记录真实失败；未将接口可访问称作质量或性能通过。Windows软件回归399项，398通过／1项本地Piper条件跳过；Node两组回归通过。

模型、ARM运行时、venv、缓存、真实照片与原始测试结果不入Git。完整本次记录位于D:/Desktop/Project/Log-Cache/CM5-AI-M-20261103。Pi安装前状态和原配置副本在/home/mylin/.cm5-ai-m-task-20261103；清理审计及说明见deploy/pi-test-deployment.md。

服务保留供用户测试；用户通知结束后，先读取结果并给出优化选项审核，再清理本次内容并验证残留。未修改网络／SSH或启动配置，无新apt包；当前仍在测试阶段，不声称已恢复初始状态。

批次实际目录为板端data/test-runs/；原图与JSON留在同一批次目录。Windows启动入口只打开SSH转发网页。
