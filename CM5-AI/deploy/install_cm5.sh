#!/usr/bin/env bash
# CM5 ARM64 安装：只安装依赖、创建目录和 systemd 单元，不下载模型、不购买服务。
set -euo pipefail

ROOT_DIR="${CM5_AI_ROOT:-/opt/cm5-ai}"
SERVICE_USER="${CM5_SERVICE_USER:-cm5}"
START_NOW="${START_NOW:-0}"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "此脚本只用于 aarch64（CM5）。当前架构：$(uname -m)" >&2
  exit 1
fi

if [[ ! -d "$ROOT_DIR/app" ]]; then
  echo "未找到 $ROOT_DIR/app，请先把 CM5-AI 目录放到 $ROOT_DIR" >&2
  exit 1
fi

sudo apt-get update
sudo apt-get install -y \
  python3-venv python3-dev python3-picamera2 \
  alsa-utils libopenblas0 libgomp1 libcamera-tools \
  git cmake build-essential \
  curl ca-certificates

python3 -m venv --system-site-packages "$ROOT_DIR/app/.venv"
"$ROOT_DIR/app/.venv/bin/pip" install --upgrade pip
"$ROOT_DIR/app/.venv/bin/pip" install -r "$ROOT_DIR/app/requirements.txt"

if ! id -u "$SERVICE_USER" >/dev/null 2>&1; then
  sudo useradd --system --create-home --shell /bin/bash "$SERVICE_USER"
fi
sudo usermod -aG audio,video,render "$SERVICE_USER" 2>/dev/null || true

mkdir -p "$ROOT_DIR/logs" "$ROOT_DIR/data" "$ROOT_DIR/runtime" "$ROOT_DIR/models"
sudo chown -R "$SERVICE_USER:$SERVICE_USER" "$ROOT_DIR"

if [[ ! -f /etc/cm5-ai.env ]]; then
  sudo install -m 600 "$ROOT_DIR/deploy/cm5-ai.env.example" /etc/cm5-ai.env
  echo "已生成 /etc/cm5-ai.env，请编辑模型路径、ALSA 设备与 VLM_API_KEY 后再启动。"
fi

sudo install -m 644 "$ROOT_DIR/deploy/cm5-ai-vlm.service" /etc/systemd/system/cm5-ai-vlm.service
sudo install -m 644 "$ROOT_DIR/deploy/cm5-ai-web.service" /etc/systemd/system/cm5-ai-web.service
sudo install -m 644 "$ROOT_DIR/deploy/cm5-ai-healthcheck.service" /etc/systemd/system/cm5-ai-healthcheck.service
sudo install -m 644 "$ROOT_DIR/deploy/cm5-ai-healthcheck.timer" /etc/systemd/system/cm5-ai-healthcheck.timer
sudo install -m 755 "$ROOT_DIR/deploy/healthcheck.sh" /usr/local/bin/cm5-ai-healthcheck

sudo systemctl daemon-reload
sudo systemctl enable cm5-ai-vlm.service cm5-ai-web.service cm5-ai-healthcheck.timer

if [[ "$START_NOW" == "1" ]]; then
  sudo systemctl start cm5-ai-vlm.service
  sleep 5
  sudo systemctl start cm5-ai-web.service
  sudo systemctl start cm5-ai-healthcheck.timer
fi

echo
echo "安装完成。下一步："
echo "  1) 编辑 /etc/cm5-ai.env（VLM_API_KEY 用 openssl rand -hex 16 生成）"
echo "  2) 把模型放到 $ROOT_DIR/models，Piper 与 llama-server 放到 $ROOT_DIR/runtime"
echo "  3) $ROOT_DIR/app/.venv/bin/python $ROOT_DIR/app/precheck.py --allow-busy"
echo "  4) sudo systemctl start cm5-ai-vlm && sudo systemctl start cm5-ai-web"
echo "  5) curl http://127.0.0.1:8090/api/health"
echo "回滚：sudo systemctl disable --now cm5-ai-web cm5-ai-vlm cm5-ai-healthcheck.timer"
