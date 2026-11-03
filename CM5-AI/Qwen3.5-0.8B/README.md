# Qwen3.5-0.8B 独立候选工程

此目录是完整候选源码工程，最初派生自 e213bce16a9e4ad2713033e726d3507f21113f20 的 MiniCPM 工程；后续公共代码按 config/shared-source.json 同步至 M2 dad8709ae1953532a30e1dd22fadce73693c5ad9（含 M1 和稳定性修复）。profile.baseline_code_sha 记录最初派生点，shared-source.source_commit 记录最新公共同步点。版本标明 candidate，产品默认模型仍保留原选择。仅在 main 提交，不另建分支。

## 目录与边界

- app/、deploy/、tools/、testsets/、samples/：独立源码、测试与部署模板。
- config/model-profile.json：独立权重、视觉组件、运行器与模型身份；config/model-manifest.json 记录上游及转换来源与未闭合的音频/OCR许可项。
- models/、runtime/、app/.venv/、logs/、data/、test-runs/：本工程独有本地资产，Git忽略；没有复制 MiniCPM 权重、原虚拟环境或原运行时。
- 默认语音输入/输出为 browser 调试，IMAGE_SOURCE=main-control；没有真实主控传输地址，也不会自动启独立相机。候选不包含已验证的完整音频/OCR运行包。

## 固定候选

语言：Unsloth Qwen3.5-0.8B-Q4_K_M.gguf；视觉：同一固定revision的 mmproj-F16.gguf。原始模型 Qwen/Qwen3.5-0.8B 与转换制品分别记录，不能将转换方当原厂。运行器使用官方 llama.cpp b12066 CPU x64；目标板运行器须另在目标系统构建与验证，Windows结果不外推。

准备环境后显式安装：

```powershell
python -m venv app/.venv
app/.venv/Scripts/python.exe -m pip install -r app/requirements.txt
app/.venv/Scripts/python.exe tools/prepare_candidate.py --models --runner
app/.venv/Scripts/python.exe app/precheck.py --vlm-only --json
```

下载工具固定revision，核SHA256、文件大小和ZIP路径。完整本地安装回执在 models/assets-installed.json 与 runtime/installed.json。VLM-only预检判断独立依赖和非空资产存在，不证明中文质量、实际出声、取消计算已回收或目标板通过。完整预检逐项检查 ASR 模型/tokens、流式 encoder/decoder/tokens、OCR det/rec/cls、TTS 模型/JSON/可执行和 KWS encoder/decoder/joiner/tokens/关键词。还必须有 profile 的 complete_interaction_assets_installed=true，以及 models/interaction-ready.json 中当前版本、profile、逐资产SHA256和真实ASR解码/流式解码/OCR/TTS PCM/KWS运行检查回执全部匹配；当前没有该回执，标记为false，不允许空文件冒充已就绪。此工具不会自行制造运行通过回执。

## 独立启动与停止

启动必须显式提供两个不同的本机空闲端口，示例18100/18101仅是候选实验参数，与主控无关：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File app/start.ps1 -WebPort 18100 -VlmPort 18101 -RunnerOnly
```

- RunnerOnly 只启动图文模型，不触发全交互资产或设备采集。
- 完整启动须先通过完整交互预检；Start-Process 使用隐藏窗口，所有PID与日志在本目录。
- 端口占用或 /v1/models 别名不匹配时失败，不接管另一个模型。
- start.ps1清理并固定继承的 DATA_DIR、PIPER_BIN/MODEL、KWS_MODEL_DIR/KEYWORDS 和流式ASR模型目录。直接 python app/server.py 同样校验环境和profile路径必须在本候选根内，且完整启动前须完整资产/回执通过；不会绕过预检借用原项目资产或缓存。
- ModelRunner和实际server生成前逐次验证本机HTTP根地址及实际模型别名，身份查询和生成请求均禁用重定向、明确拒绝3xx。
- stop.ps1只停止本候选PID记录中路径和启动时间均匹配的进程，不按端口停止进程。
- 部署模板根为 /opt/cm5-ai-qwen-candidate、专属cm5-ai-qwen-*服务和/etc/cm5-ai-qwen.env；未在目标板安装或执行。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File app/stop.ps1
```

## ModelRunner 与对照

app/model_runner.py 支持纯文字、单图、两图与流式事件；只允许显式本机HTTP地址并验证实际模型别名。消息中的A/B图片顺序保留。模型输出不具有主控空间身份。

请求通过 chat_template_kwargs.enable_thinking=false 关闭该候选的思考输出，记录该设置；不能把隐藏思考当用户首声。当前取消先阻断客户端输出并关闭响应，运行器后台计算实际退出仍须实测，不声明算力已即时回收。

```powershell
app/.venv/Scripts/python.exe -X utf8 -m unittest discover -s app -p test_candidate_runner.py
```

公共语音、取消和视觉会话修改应从同一已记录提交同步，不能将两套不同应用代码结果当成纯模型差异。正式模型选择仍需独立质量集和目标板资源对照；本目录存在不等于候选已被采用。

## 上游资料

- https://huggingface.co/Qwen/Qwen3.5-0.8B
- https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF
- https://github.com/ggml-org/llama.cpp/releases/tag/b12066

上游标注许可不自动覆盖每一份音色、词典、转换组件或训练数据；待核项明确保留，不用仓库根许可证代替。

## 本轮实际状态（2026-11-03）

独立模型两组件、CPU运行器与专属虚拟环境已在本机取得。真实文字、多轮、单图与双图调用可运行，但不将可运行写成产品质量通过。公共模块同步后最终回归355项，351通过、4项因未安装候选KWS模型跳过；其中 ModelRunner 14项、候选路径/预检/实际生成边界10项。NodeUI通过；start/stop 经内置 Windows PowerShell5.1 实际启动/停止验证，脚本UTF-8 BOM兼容5.1；VLM-only ready=true，完整interaction ready=false。真实音频和目标板测试未执行，不能将跳过或mock用例写成整机通过。

已用同一个b12066运行器、相同4线程/context2048/CPU/temperature0.2/top_p0.9/cache=false/thinking=false，对MiniCPM与Qwen各轮流跑固定四题；文字预算128/160/256，图像128，输入摘要全部相同。Qwen三项文字输出均达length上限，MiniCPM多轮仍有人称混淆；两者都没有独立质量验收通过。这是运行器级相同输入对照，未经过两套应用端到端执行，不声称应用完整链路已在此对照中一致实测。进程内存和耗时原始结果仅代表本机单次运行器小样，不是TTS首声、目标板P95或整个交互峰值。

```powershell
app/.venv/Scripts/python.exe tools/compare_runners.py --baseline-root ../MiniCPM-V4.7-1B --port 18111 --output test-runs/paired-runner-compare/results.json
```

该工具只读显式给出的MiniCPM权重路径，使用本候选独立运行器；不复制或修改原权重，也不更改原默认运行器。每个运行器结束后清理自己的PID，再启动下一个。本地初始与对照结果分别位于 test-runs/initial-runner-smoke 与 test-runs/paired-runner-compare，公开报告由主线程汇总。

## 失败审查与训练门槛

```powershell
app/.venv/Scripts/python.exe tools/export_failure_review.py --input test-runs/paired-runner-compare/results.json --output test-runs/paired-runner-compare/failure-review.jsonl
```

导出保留 case_id、输入摘要、模型/已记录版本、finish_reason、原样回答；缺失版本填null，不能填后来版本冒充实测。只自动标记length等客观观察，中文自然度、人称、事实、图像正确性由人工审核，未审核不写质量通过。

先修输出预算、角色/历史、图像身份和字段关联等工程失败，固定同题复测。工程原因排除后，再判断是否需要一个较高精度（例如Q8）对照来区分量化退化与模型能力限制。本轮不下载额外精度模型、不自动启动训练。

这四题是开发小样，不能作为独立验收或训练收益证明。审查模板的训练/验证/独立测试划分按商品/场景/拍摄批次/对话来源分组，默认unassigned；数据许可、泄漏检查和人工答案复核未通过时，approved_for_training保持false。只有独立测试和目标板收益支持时才采用微调版本，微调不是主要降内存手段。

最终适配后的本机开发烟测使用同一 start.ps1 和 ModelRunner：32tokens上限，文字690.1ms/stop/22生成tokens，猫图1901.0ms/stop/回答2/2生成tokens。停止退出0，18120与18121端口释放，PID记录清除。这些wall时间包含身份查询和模型HTTP完成，不是有效首声或目标板指标；该两题不构成中文与视觉质量验收。
