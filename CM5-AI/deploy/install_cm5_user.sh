#!/usr/bin/env bash
# CM5 用户级部署（不需要 sudo）：发布目录 + 共享模型 + user systemd 单元。
#
# 用法：
#   bash deploy/install_cm5_user.sh                    # 在仓库 CM5-AI 目录执行
#   CM5_ROOT=$HOME/cm5-ai RELEASE_ID=v1.3.0 bash deploy/install_cm5_user.sh
#
# 约定：
#   $CM5_ROOT/models      模型共享目录（不随发布重复复制）
#   $CM5_ROOT/runtime     llama.cpp / piper 共享运行时
#   $CM5_ROOT/releases/  每次发布一个目录
#   $CM5_ROOT/current     指向当前发布，回滚只需改这个链接
#   $CM5_ROOT/env/cm5-ai.env  运行配置（权限 600，含随机 API key）
#   $CM5_ROOT/data, logs  运行数据（跨发布保留）
#
# 本脚本不下载模型、不升级系统包、不启动服务。
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${CM5_ROOT:-$HOME/cm5-ai}"
RELEASE_ID="${RELEASE_ID:-v1.3.0-cm5}"
RELEASE="$ROOT/releases/$RELEASE_ID"
ENV_FILE="$ROOT/env/cm5-ai.env"
UNIT_DIR="$HOME/.config/systemd/user"
PY="$RELEASE/app/.venv/bin/python"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "只支持 aarch64（CM5），当前：$(uname -m)" >&2
  exit 1
fi
if [[ ! -f "$SRC/app/server.py" ]]; then
  echo "找不到 $SRC/app/server.py，请在 CM5-AI 目录内运行" >&2
  exit 1
fi

echo "==> 发布目录 $RELEASE"
mkdir -p "$RELEASE" "$ROOT/models" "$ROOT/runtime" "$ROOT/data" "$ROOT/logs" "$ROOT/env" "$ROOT/test-runs"
for item in app deploy tools testsets samples; do
  rm -rf "$RELEASE/$item"
  cp -a "$SRC/$item" "$RELEASE/$item"
done
find "$RELEASE/app" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$RELEASE/app" -name '*.pyc' -delete 2>/dev/null || true
rm -rf "$RELEASE/app/.venv" "$RELEASE/app/logs" "$RELEASE/app/data"

# 共享目录用相对链接，回滚或换发布都不复制模型
ln -sfn ../../models "$RELEASE/models"
ln -sfn ../../runtime "$RELEASE/runtime"
ln -sfn ../../logs "$RELEASE/logs"
ln -sfn ../../data "$RELEASE/data"

echo "==> 发布清单 release-manifest.json"
GIT_COMMIT="${GIT_COMMIT:-unknown}"
GIT_DIRTY="${GIT_DIRTY:-unknown}"
python3 - "$RELEASE" "$GIT_COMMIT" "$GIT_DIRTY" "$ROOT" <<'PY'
import hashlib, json, os, sys, time
from pathlib import Path

release = Path(sys.argv[1])
commit, dirty, root = sys.argv[2], sys.argv[3], Path(sys.argv[4])

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

files = []
for base in ("app", "deploy", "tools", "testsets", "samples"):
    for path in sorted((release / base).rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or ".venv" in path.parts:
            continue
        files.append({
            "path": str(path.relative_to(release)),
            "bytes": path.stat().st_size,
            "sha256": sha(path),
        })

models = {}
for rel in (
    "vlm/minicpm-v-4.7-1b/MiniCPM-V-4_7-1B-Q4_K_M.gguf",
    "vlm/minicpm-v-4.7-1b/mmproj-model-f16.gguf",
    "ocr/pp-ocrv6-small/PP-OCRv6_det_small.onnx",
    "ocr/pp-ocrv6-small/PP-OCRv6_rec_small.onnx",
    "ocr/pp-ocrv6-small/ch_ppocr_mobile_v2.0_cls_mobile.onnx",
    "asr/sherpa-onnx-paraformer-zh-small-2024-03-09/model.int8.onnx",
    "tts/piper-zh_CN-huayan-medium/zh_CN-huayan-medium.onnx",
):
    path = root / "models" / rel
    models[rel] = {"exists": path.exists(), "bytes": path.stat().st_size if path.exists() else None,
                   "sha256": sha(path) if path.exists() else None}

manifest = {
    "release_id": release.name,
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "git_commit": commit,
    "git_dirty_entries": dirty,
    "python": sys.version.split()[0],
    "arch": os.uname().machine,
    "files": files,
    "models": models,
    "runtime": {
        "llama_cpp": str(root / "runtime/llama.cpp"),
        "piper": str(root / "runtime/piper"),
    },
}
(release / "release-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"    {len(files)} 个文件，{len(models)} 个模型")
PY

echo "==> Python 虚拟环境（--system-site-packages 复用系统 picamera2）"
python3 -m venv --system-site-packages "$RELEASE/app/.venv"
"$RELEASE/app/.venv/bin/pip" install --disable-pip-version-check --no-input \
  -r "$RELEASE/deploy/requirements-cm5.txt"
"$RELEASE/app/.venv/bin/pip" freeze --disable-pip-version-check > "$RELEASE/pip-freeze.txt"

echo "==> 运行配置 $ENV_FILE"
if [[ ! -f "$ENV_FILE" ]]; then
  KEY="$(openssl rand -hex 16)"
  cat > "$ENV_FILE" <<EOF
# CM5-AI 运行配置。权限 600；不要提交到仓库。
WEB_HOST=127.0.0.1
WEB_PORT=8090
PYTHONUNBUFFERED=1

# 模型（共享目录；只使用本机发布清单中的制品）
VLM_URL=http://127.0.0.1:8091
VLM_MODEL=$ROOT/models/vlm/minicpm-v-4.7-1b/MiniCPM-V-4_7-1B-Q4_K_M.gguf
VLM_MMPROJ=$ROOT/models/vlm/minicpm-v-4.7-1b/mmproj-model-f16.gguf
PIPER_BIN=$ROOT/runtime/piper/piper
PIPER_MODEL=$ROOT/models/tts/piper-zh_CN-huayan-medium/zh_CN-huayan-medium.onnx
KWS_MODEL_DIR=$ROOT/models/kws/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01
KWS_KEYWORDS=$ROOT/current/app/config/keywords.txt
DATA_DIR=$ROOT/data

# 主控接口 interaction.v0.1；正式取图只走 FrameProvider
IMAGE_SOURCE=main_control
MAIN_CONTROL_URL=
MAIN_CONTROL_RPC_PATH=/interaction/rpc
MAIN_CONTROL_EVENTS_PATH=/interaction/events
MAIN_CONTROL_ASSET_ROOT=$ROOT/data/main-control
MAIN_CONTROL_ALLOW_FIXTURE=0

VLM_API_KEY=$KEY

# 4GB 整机预算，必须留系统内存
MEM_BUDGET_GB=3.6
# 读取大进程的 smaps_rollup 开销明显；状态/健康轮询用 10 秒缓存
MEM_SAMPLE_TTL=10
VLM_THREADS=3
VLM_CTX=1024
VLM_IMAGE_MAX_SIDE=512

ASR_THREADS=1
ASR_IDLE_SECONDS=180
OCR_INTRA_THREADS=2
OCR_INTER_THREADS=1
KWS_THREADS=1

TTS_BACKEND=persistent
TTS_SMOOTH=0
TTS_IDLE_SECONDS=0
TTS_TIMEOUT=90

KWS_SCORE=2.0
KWS_THRESHOLD=0.10
KWS_TRAILING_BLANKS=8
KWS_INT8=0

VOICE_SOURCE=device
OUTPUT_SINK=alsa
# 用稳定卡名，不依赖会变化的 card 编号
ALSA_INPUT=plughw:CARD=Array,DEV=0
ALSA_OUTPUT=plughw:CARD=Array,DEV=0

FRAME_MAX_AGE=20
ARMED_SECONDS=15
MIN_VOICE_SECONDS=0.6
MIN_VOICE_RMS=0.008
DEBUG_SNAPSHOT=0
EOF
  chmod 600 "$ENV_FILE"
else
  echo "    已存在，保留原配置（不覆盖既有密钥与调参）"
fi

echo "==> 运行时链接"
if [[ -z "$(ls -A "$ROOT/runtime")" ]]; then
  LLAMA_BUILD=""
  for cand in "$HOME/src/llama.cpp-b11960/build" "$HOME/src/llama.cpp/build"; do
    [[ -x "$cand/bin/llama-server" ]] && LLAMA_BUILD="$cand" && break
  done
  if [[ -n "$LLAMA_BUILD" ]]; then
    ln -sfn "$LLAMA_BUILD" "$ROOT/runtime/llama.cpp"
    echo "    llama.cpp -> $LLAMA_BUILD"
  else
    echo "    WARN 未找到已编译的 llama-server，稍后把 build 目录链接到 $ROOT/runtime/llama.cpp"
  fi
  if [[ -x "$HOME/piper/piper" ]]; then
    ln -sfn "$HOME/piper" "$ROOT/runtime/piper"
    echo "    piper -> $HOME/piper"
  else
    echo "    WARN 未找到 $HOME/piper/piper（Piper 引擎 GPL-3.0，需自行放置 aarch64 版本）"
  fi
fi

echo "==> user systemd 单元"
mkdir -p "$UNIT_DIR"
cat > "$UNIT_DIR/cm5-ai-vlm.service" <<EOF
[Unit]
Description=CM5-AI llama.cpp VLM (127.0.0.1:8091)
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
Type=simple
EnvironmentFile=$ENV_FILE
WorkingDirectory=$ROOT/runtime/llama.cpp/bin
ExecStart=/bin/bash -lc 'exec taskset -c 0-2 $ROOT/runtime/llama.cpp/bin/llama-server -m "\$VLM_MODEL" --mmproj "\$VLM_MMPROJ" -c "\$VLM_CTX" -t "\$VLM_THREADS" -ngl 0 --jinja --parallel 1 --host 127.0.0.1 --port 8091 --api-key "\$VLM_API_KEY" --cors-origins http://127.0.0.1:8090'
Restart=on-failure
RestartSec=5
NoNewPrivileges=true
PrivateTmp=true
TimeoutStopSec=30

[Install]
WantedBy=default.target
EOF

cat > "$UNIT_DIR/cm5-ai-web.service" <<EOF
[Unit]
Description=CM5-AI interaction web, KWS and device audio (127.0.0.1:8090)
After=network.target cm5-ai-vlm.service
Wants=cm5-ai-vlm.service
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
Type=simple
EnvironmentFile=$ENV_FILE
WorkingDirectory=$ROOT/current/app
ExecStartPre=$PY $ROOT/current/app/precheck.py --allow-busy
ExecStart=$PY $ROOT/current/app/server.py
Restart=on-failure
RestartSec=5
KillSignal=SIGTERM
TimeoutStopSec=20
NoNewPrivileges=true
PrivateTmp=true
LimitNOFILE=65536
# 不用 MemoryMax 直接杀服务；内存预算由应用检查并返回 503。

[Install]
WantedBy=default.target
EOF

cat > "$UNIT_DIR/cm5-ai-healthcheck.service" <<EOF
[Unit]
Description=CM5-AI health check (read-only)

[Service]
Type=oneshot
Environment=CM5_HEALTH_URL=http://127.0.0.1:8090/api/health
Environment=CM5_HEALTH_LOG=$ROOT/logs/health.log
ExecStart=$ROOT/bin/cm5-ai-healthcheck
EOF

cat > "$UNIT_DIR/cm5-ai-healthcheck.timer" <<EOF
[Unit]
Description=CM5-AI health check every 10 minutes

[Timer]
OnBootSec=3min
OnUnitActiveSec=10min
Persistent=true
Unit=cm5-ai-healthcheck.service

[Install]
WantedBy=timers.target
EOF

mkdir -p "$ROOT/bin"
install -m 755 "$RELEASE/deploy/healthcheck.sh" "$ROOT/bin/cm5-ai-healthcheck"

ln -sfn "releases/$RELEASE_ID" "$ROOT/current"

systemctl --user daemon-reload
systemctl --user enable cm5-ai-vlm.service cm5-ai-web.service cm5-ai-healthcheck.timer >/dev/null
echo "    已 enable（未启动）"

echo "==> 预检（跳过端口占用，旧实例可能仍在运行）"
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
"$PY" "$ROOT/current/app/precheck.py" --allow-busy || true

cat <<EOF

部署完成：$RELEASE
当前链接：$ROOT/current -> releases/$RELEASE_ID
配置：$ENV_FILE（600）
启动：systemctl --user start cm5-ai-vlm && sleep 5 && systemctl --user start cm5-ai-web
状态：systemctl --user status cm5-ai-vlm cm5-ai-web --no-pager
日志：journalctl --user -u cm5-ai-web -f
回滚：ln -sfn releases/<旧版本> $ROOT/current && systemctl --user restart cm5-ai-vlm cm5-ai-web
EOF
