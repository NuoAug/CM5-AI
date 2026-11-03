#!/usr/bin/env bash
# 5 分钟健康检查：API 可用、内存不超预算、温度与欠压。只读，不重启服务。
set -uo pipefail

API="${CM5_HEALTH_URL:-http://127.0.0.1:18100/api/health}"
LOG="${CM5_HEALTH_LOG:-/opt/cm5-ai-qwen-candidate/logs/health.log}"
mkdir -p "$(dirname "$LOG")"

status=0
body="$(curl -fsS --max-time 5 "$API" 2>/dev/null)" || {
  echo "$(date -Is) FAIL api unreachable $API" >>"$LOG"
  exit 1
}

if ! printf '%s' "$body" | python3 -c '
import json,sys
j=json.load(sys.stdin)
ok=bool(j.get("ok"))
over=bool(j.get("memory_over_budget"))
print("FAIL" if (not ok or over) else "OK", "over_budget="+str(over))
sys.exit(0 if (ok and not over) else 1)
' >>"$LOG" 2>&1; then
  status=1
fi

if command -v vcgencmd >/dev/null 2>&1; then
  temp="$(vcgencmd measure_temp 2>/dev/null | sed -n 's/.*temp=\([0-9.]*\).*/\1/p')"
  flags="$(vcgencmd get_throttled 2>/dev/null | sed -n 's/.*throttled=\(0x[0-9a-fA-F]*\).*/\1/p')"
  echo "$(date -Is) ENV temp=${temp:-?}C throttled=${flags:-?}" >>"$LOG"
  if [[ -n "${flags:-}" && "$flags" != "0x0" ]]; then
    echo "$(date -Is) WARN undervoltage/throttle $flags" >>"$LOG"
    status=1
  fi
  if [[ -n "${temp:-}" ]] && python3 -c "import sys; sys.exit(0 if float('$temp') < 80 else 1)"; then
    :
  else
    echo "$(date -Is) WARN temperature high ${temp:-?}C" >>"$LOG"
    status=1
  fi
fi

exit "$status"
