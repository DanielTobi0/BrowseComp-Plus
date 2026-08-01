#!/bin/bash
# Install the retrieval MCP server as a supervisor service, expose it over a
# Cloudflare quick tunnel, and print the URL to paste into Claude web
# (Settings -> Connectors -> Add custom connector).
#
# Idempotent: safe to re-run after a reboot to get the current tunnel URL.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PORT="${MCP_PORT:-10100}"
SERVICE="bcp-mcp"
CONF="/etc/supervisor/conf.d/${SERVICE}.conf"

cat > "${CONF}" <<EOF
[program:${SERVICE}]
environment=PROC_NAME="%(program_name)s",MCP_PORT="${PORT}",REPO_DIR="${REPO_DIR}"
command=${REPO_DIR}/scripts_mcp/serve_claude_web.sh
autostart=true
autorestart=unexpected
exitcodes=0
startsecs=10
stopasgroup=true
killasgroup=true
stopsignal=TERM
stopwaitsecs=15
stdout_logfile=/dev/stdout
redirect_stderr=true
stdout_events_enabled=true
stdout_logfile_maxbytes=0
stdout_logfile_backups=0
EOF

supervisorctl reread >/dev/null
supervisorctl update >/dev/null
supervisorctl restart "${SERVICE}" >/dev/null 2>&1 || supervisorctl start "${SERVICE}" >/dev/null

SECRET_FILE="${REPO_DIR}/scripts_mcp/.connector_secret"
echo "Waiting for the retrieval server to load its models (this takes a minute)..."
for _ in $(seq 1 120); do
  if [[ -s "${SECRET_FILE}" ]] \
     && curl -s -o /dev/null -m 2 "http://127.0.0.1:${PORT}/$(cat "${SECRET_FILE}")/mcp"; then
    break
  fi
  sleep 5
done

SECRET="$(cat "${SECRET_FILE}")"
MCP_PATH="/${SECRET}/mcp"
TARGET="http://localhost:${PORT}"

# The instance's tunnel manager owns cloudflared; ask it for a tunnel to our port.
TUNNEL="$(curl -s "http://localhost:11111/get-existing-quick-tunnel/$(python3 -c 'import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=""))' "${TARGET}")" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d.get("tunnelUrl") or "")' 2>/dev/null || true)"
if [[ -z "${TUNNEL}" ]]; then
  curl -s -X POST "http://localhost:11111/start-quick-tunnel/$(python3 -c 'import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=""))' "${TARGET}")" >/dev/null
  for _ in $(seq 1 20); do
    TUNNEL="$(curl -s http://localhost:11111/get-all-quick-tunnels \
      | python3 -c 'import json,sys;t=sys.argv[1];print(next((x["tunnelUrl"] for x in json.load(sys.stdin) if x.get("targetUrl")==t), ""))' "${TARGET}")"
    [[ -n "${TUNNEL}" ]] && break
    sleep 3
  done
fi

if [[ -z "${TUNNEL}" ]]; then
  echo "Could not obtain a Cloudflare quick tunnel. The server is still running locally at:"
  echo "  http://127.0.0.1:${PORT}${MCP_PATH}"
  exit 1
fi

echo
echo "============================================================"
echo "Claude web custom connector URL (Remote MCP, streamable HTTP):"
echo
echo "  ${TUNNEL}${MCP_PATH}"
echo
echo "The random path IS the access credential - treat the URL as a secret."
echo "Quick tunnels are ephemeral: re-run this script after a reboot to get the new URL."
echo "Logs: tail -f /var/log/portal/${SERVICE}.log"
echo "============================================================"
