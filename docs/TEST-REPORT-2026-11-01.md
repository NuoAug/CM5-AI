# 本轮修改与测试报告（2026-11-01）

## 一、本轮修改的文件

| 文件 | 内容 |
|---|---|
| `CM5-AI/app/server.py` | 会话/画面分离、聊天与详细描述独立规则、日期解析重写、分句重构、流式接口、资源限制、OCR 内存图像、线程配置、captured_at 过期检查 |
| `CM5-AI/app/static/index.html` | 即时抓拍与 frame_id、高清图上传、聊天流式播报、停止按钮、session_id、旧响应丢弃 |
| `CM5-AI/app/test_logic.py` | 新增回归测试（43 项） |
| `CM5-AI/README.md` | 功能、接口、环境变量、真机候选说明 |

## 二、测试性质说明

- **单元回归测试**：`test_logic.py` 共 43 项，不加载任何模型，只测纯函数、路由规则、提示词与状态机。属于 mock 级别，不代表模型真实表现。
- **API 冒烟测试**：在本机 Windows（Ryzen 7 4800H，非 CM5）上调用真实服务，使用现有样例图，调用真实 VLM/OCR/TTS。
- **浏览器验证**：真实页面，未使用真实麦克风录音。
- **未在 CM5 上运行任何测试**。下文所有耗时与内存数字均为本机数据，不能当作 CM5 结果。

## 三、单元测试结果（43 项全部通过）

| 分组 | 覆盖点 | 结果 |
|---|---|---|
| SettleTests | “水杯在右边，没有被挡住”不被误判；真正缺失才收敛兜底 | 通过 |
| IntentTests | 卫生间/出口/日期/读字/聊天/新闻/找物品/比较/详细/场景路由 | 通过 |
| DateTests | 正常字段、占位语“见瓶底”、非法日期 2026-19-45、score 缺失、同字段冲突、包装日期≠生产日期、保质期时长、MFG 小写、按 box 取邻行、无关键词 | 通过 |
| SentenceTests | 不静默丢内容、保留问号、超长无标点硬切且不超长 | 通过 |
| SessionTests | 会话隔离、切帧保留聊天并标记旧画面失效、轮数上限、会话数上限、TTL 过期 | 通过 |
| PromptTests | chat 含用户问题与历史、followup 允许图片、detail 用“三到五句”规则、scene 用短规则 | 通过 |
| FrameAgeTests | 新帧、空值、非法格式、过期帧判定 | 通过 |
| ResourceConfigTests | temperature 分工、调试快照默认关闭、线程配置为正 | 通过 |

运行命令：

```powershell
CM5-AI\app\.venv\Scripts\python.exe -m unittest test_logic
```

## 四、API 冒烟结果（本机真实模型）

| 用例 | 结果 | 备注 |
|---|---|---|
| chat「今天有点烦」 | 正常回应，不再描述画面 | 1.7 s |
| chat「那我该怎么办」同会话 | 延续上一轮内容 | 1.9 s |
| chat「今天有什么新闻」 | 规则直接拒答“没有联网” | 0 ms，不调模型 |
| date 样例标牌 | 拒猜，返回“日期没看清…” | 1.2 s |
| followup 两轮（同帧） | 第二轮能答颜色，说明历史+图片生效 | 8.6 s |
| 切帧后再追问 | 保留聊天历史，旧画面标记失效 | 7.9 s |
| OCR 内存图像（不再落盘） | 三行文字识别结果与改动前一致 | 源为 OCR |
| 过期帧 captured_at=-120s | 返回 400「画面已过期」 | 生效 |
| `/api/talk` 流式 | 两句话逐句推送 + done 事件 | 正常 |
| 浏览器点“读日期” | 显示“日期没看清…”，无 JS 报错 | 通过 |

## 五、本轮两项低风险优化

### A. 减少图片落盘

- RapidOCR 3.10 的 `__call__` 确认支持 `numpy.ndarray` / `PIL.Image` / `bytes` / `Path` / `str`，已改为直接传内存图像对象，`/api/ocr`、文字任务、日期任务都不再写固定文件。
- 调试快照默认关闭（`DEBUG_SNAPSHOT=0`）。开启后文件名带 `时间戳-线程ID`，不会并发写到同一文件导致串图。
- `DATA_DIR` 支持环境变量覆盖，Windows 默认仍为 `CM5-AI/data`，行为不变。
- 验证：清理旧图后跑 OCR 与日期任务，`data/` 没有新增文件。

### B. 线程数可配置

| 变量 | 默认 | 作用 |
|---|---|---|
| `TTS_THREADS` | 4 | MeloTTS / Kokoro |
| `ASR_THREADS` | 4 | Paraformer |
| `OCR_INTRA_THREADS` | 4 | RapidOCR 的 ORT intra-op |
| `OCR_INTER_THREADS` | 1 | RapidOCR 的 ORT inter-op |
| `FRAME_MAX_AGE` | 20 | 画面过期秒数 |

- ORT 线程参数键为 `EngineConfig.onnxruntime.intra_op_num_threads` / `inter_op_num_threads`，已在 RapidOCR 源码中核实。
- 默认值与改动前一致，未改变现有行为。
- `OMP_NUM_THREADS` 不能保证控制 ONNX Runtime 线程池，ORT 使用自己的 intra/inter 设置；本轮未使用环境变量做线程控制。
- 未做绑核（taskset / 亲和性），因为没有 CM5 实测数据支撑。

## 六、未完成项

1. **语音入口仍是非流式**：`/api/voice` 先整段生成再返回，前端再分句播报；只有文字 `/api/talk` 是真流式。两条路径已统一 prompt、temperature、max_tokens、会话记录与兜底逻辑，但流式能力不一致。属“具备集成路径，尚未打通”。
2. **CM5 未实测**：延迟、内存峰值、热降频、线程数收益全部未知。
3. **真实麦克风未验证**：本轮只用 TTS 回放音频，未录真实人声。
4. **日期标签未用真实照片验证**：只用构造的 OCR 行文本做单元测试和样例图。
5. **流式取消的资源释放只依赖 Flask 关闭生成器**，未做并发压测确认 `leave()` 一定执行。
6. **取消后旧响应写回**：前端用 generation 计数丢弃，后端用 session 的 frame 比对标记 stale；未做高并发下的竞争测试。

## 七、真机 A/B 候选（本轮一律未执行）

以下项目需要 CM5 实机验证，且都有风险，仅作为候选列出：

| 候选 | 风险提示 |
|---|---|
| tmpfs 挂载 `data/` | tmpfs 占用 RAM，会进一步压缩 4GB 可用内存 |
| zram | 用 CPU 换内存，推理时 CPU 争用会拖慢 VLM |
| swap 文件 | SD/eMMC 写入寿命与延迟问题 |
| 限制 journald 大小 | 收益小，但可减少磁盘写入 |
| 改 fstab / gpu_mem | 影响启动与显存分配，需谨慎 |
| systemd `MemoryMax` | 超限会直接杀服务，不适合作为第一道限制 |
| 线程数 1/2/4 组合 | 需要逐组实测吞吐与延迟，不能凭经验定值 |
| CPU 亲和性绑核 | 可能反而降低吞吐，不建议盲目做 |

所有提速百分比、内存预算、预计耗时都**待 CM5 实测**，本轮不提供具体数字。

## 九、前端竞态修复与自动化测试（补充）

| 缺陷 | 修复 | 验证结果 |
|---|---|---|
| `splitSentences` 仍 `slice(0,8)`，丢问号、长句不拆 | 与后端同逻辑重写，不截断、保留标点、超长按逗号硬切 | 20 句全部保留（4 段）；问号保留；超长拆 3 段且每段不超长 |
| `stopSpeech` 只 pause，`playUrl` 的 Promise 不 resolve | 统一 `player` 对象，停止时调用 resolver 并清零 running | 播放中停止后 `pumpPlayer` 正常退出，`running=false` |
| 停止按钮不作废旧请求 | `beginInteraction` 递增 `interactionGen` + `AbortController` | 旧响应代次不符直接丢弃；测试中 gen 1→2 且旧 signal aborted |
| 连续点击互抢同一 audio | 所有入口共用同一 player 队列与 `pumpPlayer` | 连续两次播报，最后一次胜出，队列清空 |
| 停止后新回答不能播 | `stopPlayer` 重置 running，新 `speakAll` 重新入队 | 测试通过 |
| `/api/talk` 先取历史再 `session_begin` | 改为先 `session_begin` 再 `session_context`，客户端 context 仅在无服务端历史时兜底 | 代码修正，会话测试覆盖隔离与 stale |
| 流式请求未受 VLM 调度约束 | `generate` 内 `VLM_LOCK.acquire(timeout=30)`，`finally` 显式 `inner.close()` 并释放锁与计数 | 代码修正，未做并发压测 |

播放器与分句逻辑已抽到 `CM5-AI/app/static/app-core.js`，页面和测试共用同一份实现，避免实现与测试分叉。

前端测试运行方式：浏览器打开 `http://127.0.0.1:8090/static/test-player.html`，页面直接输出 PASS/FAIL。本轮 **8 项全部通过**（20 句不丢失、保留问号、超长拆分、停止后退出、停止后可播、连续播报最后一次胜出、旧请求作废、URL 回收）。

前端测试限制：

- 使用 mock `audio` 与 mock `synthTts`，不接真实音频设备。
- 未验证真实麦克风、回声消除、采样率切换在 CM5 上的表现。
- 流式取消的后端资源释放依赖 Flask 关闭生成器，未做并发压测。
- **语音入口仍是整段等待**：`/api/voice` 要等 ASR 与推理全部完成才返回，前端再分句播报；只有文字聊天走 `/api/talk` 真流式。不能称为完整实时语音。

## 十、结论

- 后端 43 项逻辑回归全部通过；前端播放器 8 项自动化测试全部通过；真实 API 冒烟覆盖聊天、日期、多轮追问、切帧、过期帧、流式与 OCR 内存路径。
- 两项低风险优化已落地且默认行为不变，调试快照默认关闭已验证。
- 语音入口与文字流式路径的参数和逻辑已统一，**但语音仍是整段等待**，只有文字聊天为真流式。
- 所有性能结论与硬件音频表现均需 CM5 实测，本轮未提供具体数字。
