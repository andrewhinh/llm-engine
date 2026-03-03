#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_NAME="$(basename "${PROJECT_DIR}")"
STAGE_ROOT="$(mktemp -d)"
STAGE_DIR="${STAGE_ROOT}/${PROJECT_NAME}"

VOLUME="llm-engine-src"
DEFAULT_CMD="cd /workspace/${PROJECT_NAME} 2>/dev/null || cd /workspace 2>/dev/null || cd /; exec /bin/bash"
CMD="${DEFAULT_CMD}"
PTY=true
GPU_TYPE="H100"
GPU_COUNT="1"

usage() {
  cat <<'EOF'
Usage: conn.sh [--cmd <cmd>|-- <cmd...>] [--pty|--no-pty] [--gpu <type>] [--gpu-count <n>]
EOF
}

die() {
  echo "$1"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cmd)
      shift
      [[ $# -gt 0 ]] || die "--cmd needs a value"
      CMD="$1"
      ;;
    --pty)
      PTY=true
      ;;
    --no-pty)
      PTY=false
      ;;
    --gpu|--gpu-type)
      shift
      [[ $# -gt 0 ]] || die "--gpu needs a value"
      GPU_TYPE="$1"
      ;;
    --gpu-count)
      shift
      [[ $# -gt 0 ]] || die "--gpu-count needs a value"
      GPU_COUNT="$1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      if [[ $# -gt 0 ]]; then
        CMD="$*"
      fi
      break
      ;;
    *)
      CMD="$*"
      break
      ;;
  esac
  shift
done

if [[ "${GPU_TYPE}" == *:* ]]; then
  GPU_COUNT="${GPU_TYPE##*:}"
  GPU_TYPE="${GPU_TYPE%%:*}"
fi

MODAL_CMD=()
if command -v uvx >/dev/null 2>&1; then
  MODAL_CMD=(uvx modal)
elif command -v modal >/dev/null 2>&1; then
  MODAL_CMD=(modal)
else
  echo "modal CLI not found. Install uv (preferred) or modal."
  exit 1
fi

run_modal() {
  "${MODAL_CMD[@]}" "$@"
}

run_modal_with_env_script() {
  if ! command -v script >/dev/null 2>&1; then
    echo "script not found; install util-linux to use non-tty mode."
    exit 1
  fi

  script -q /dev/null env "LLM_ENGINE_GPU_TYPE=${GPU_TYPE}" "LLM_ENGINE_GPU_COUNT=${GPU_COUNT}" "${MODAL_CMD[@]}" "$@" </dev/null
}

cleanup() {
  rm -rf "${STAGE_ROOT}"
}
trap cleanup EXIT

volume_exists() {
  run_modal volume list | rg -q "(^|[[:space:]])${VOLUME}([[:space:]]|$)"
}

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync not found; install rsync to use make conn."
  exit 1
fi

if ! volume_exists; then
  run_modal volume create "${VOLUME}"
fi

mkdir -p "${STAGE_DIR}"
rsync -a --delete --exclude ".git" --exclude "target" "${PROJECT_DIR}/" "${STAGE_DIR}/"
run_modal volume rm "${VOLUME}" "/${PROJECT_NAME}" -r >/dev/null 2>&1 || true
run_modal volume put "${VOLUME}" "${STAGE_DIR}" "/" --force

ARGS=(shell "${PROJECT_DIR}/scripts/modal_shell.py::dev_shell")
if "${PTY}"; then
  ARGS+=(--pty)
fi
ARGS+=(--cmd "${CMD}")

if [[ -t 0 ]]; then
  if "${PTY}"; then
    LLM_ENGINE_GPU_TYPE="${GPU_TYPE}" LLM_ENGINE_GPU_COUNT="${GPU_COUNT}" run_modal "${ARGS[@]}"
  else
    LLM_ENGINE_GPU_TYPE="${GPU_TYPE}" LLM_ENGINE_GPU_COUNT="${GPU_COUNT}" run_modal "${ARGS[@]}" </dev/null
  fi
else
  run_modal_with_env_script "${ARGS[@]}"
fi
