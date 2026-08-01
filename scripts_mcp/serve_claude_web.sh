#!/bin/bash
# Run the BrowseComp-Plus retrieval MCP server for use as a Claude web custom connector.
#
# Serves streamable-HTTP on 127.0.0.1 only; public access comes from the Cloudflare
# tunnel started by register_connector.sh, so nothing is exposed on an open port.
# Runs in the foreground (supervisor manages restarts).
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_DIR}"

# Supervisor does not inherit the login-shell environment; HF_HOME in particular
# decides whether model/dataset caches are found or re-downloaded.
if [[ -f /opt/supervisor-scripts/utils/environment.sh ]]; then
  # shellcheck disable=SC1091
  . /opt/supervisor-scripts/utils/environment.sh
elif [[ -f /etc/environment ]]; then
  set -a; . /etc/environment; set +a
fi

PORT="${MCP_PORT:-10100}"
K="${MCP_K:-5}"
SNIPPET_MAX_TOKENS="${MCP_SNIPPET_MAX_TOKENS:-512}"

# Dense side. The 0.6B index/model is what the local eval runs used; point these at
# indexes/qwen3-embedding-8b + Qwen/Qwen3-Embedding-8B to match the paper's setup.
EMBEDDING_MODEL="${MCP_EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
FAISS_INDEX="${MCP_FAISS_INDEX:-indexes/qwen3-embedding-0.6b/corpus.shard*_of_4.pkl}"
BM25_INDEX="${MCP_BM25_INDEX:-indexes/bm25}"

# The URL path doubles as the shared secret: Claude web connectors cannot send an
# Authorization header, so an unguessable path is what keeps the tunnel private.
# Generated once and reused so the connector URL survives restarts.
SECRET_FILE="${REPO_DIR}/scripts_mcp/.connector_secret"
if [[ ! -s "${SECRET_FILE}" ]]; then
  head -c 24 /dev/urandom | base64 | tr -d '=+/' | cut -c1-24 > "${SECRET_FILE}"
  chmod 600 "${SECRET_FILE}"
fi
SECRET="$(cat "${SECRET_FILE}")"
MCP_PATH="/${SECRET}/mcp"

source "${REPO_DIR}/.venv/bin/activate"

exec python searcher/mcp_server.py \
  --searcher-type custom \
  --bm25-index-path "${BM25_INDEX}" \
  --faiss-index-path "${FAISS_INDEX}" \
  --embedding-model-name "${EMBEDDING_MODEL}" \
  --get-document \
  --k "${K}" \
  --snippet-max-tokens "${SNIPPET_MAX_TOKENS}" \
  --transport streamable-http \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --path "${MCP_PATH}"
