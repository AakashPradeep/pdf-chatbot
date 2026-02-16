#!/usr/bin/env bash
set -euo pipefail

APP_MODULE="src.rag.app"
VENV_DIR=".venv"
REQ_FILE="requirements.txt"

ensure_venv_and_deps() {
  # Create venv if missing
  if [[ ! -d "${VENV_DIR}" ]]; then
    echo "🛠  Creating virtualenv at ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
  fi

  # Activate venv
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"

  # Install requirements if missing marker or if user forced install
  if [[ "${INSTALL_DEPS:-0}" == "1" ]]; then
    echo "📦 Installing/upgrading requirements from ${REQ_FILE}..."
    python -m pip install --upgrade pip
    python -m pip install -r "${REQ_FILE}"
    return
  fi

  # Lightweight check: if pip can't import a core dependency, install once
  if ! python -c "import click" >/dev/null 2>&1; then
    echo "📦 Dependencies not found in venv. Installing from ${REQ_FILE}..."
    python -m pip install --upgrade pip
    python -m pip install -r "${REQ_FILE}"
  fi
}

run_ingest() {
  local PDF_ROOT="${1:-./pdfs}"
  echo "➡️  Ingesting PDFs from: ${PDF_ROOT}"
  python -m "${APP_MODULE}" ingest --pdf-root "${PDF_ROOT}"
}

run_chat() {
  echo "➡️  Starting chat..."
  python -m "${APP_MODULE}" chat
}

usage() {
  echo "Usage:"
  echo "  ./run.sh                 # interactive menu"
  echo "  ./run.sh ingest --pdf-root ./pdfs"
  echo "  ./run.sh chat"
  echo "Options:"
  echo "  --install               Force install/upgrade requirements"
}

# Parse optional global flags
INSTALL_DEPS=0
ARGS=()
for a in "$@"; do
  case "$a" in
    --install)
      INSTALL_DEPS=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      ARGS+=("$a")
      ;;
  esac
done

# Ensure venv + deps first
ensure_venv_and_deps

# Reassign args after stripping flags
set -- "${ARGS[@]:-}"

# Non-interactive mode (command provided)
if [[ $# -gt 0 ]]; then
  CMD="$1"; shift || true
  case "${CMD}" in
    ingest)
      python -m "${APP_MODULE}" ingest "$@"
      ;;
    chat)
      python -m "${APP_MODULE}" chat "$@"
      ;;
    *)
      echo "Unknown command: ${CMD}"
      echo "Valid: ingest | chat"
      exit 1
      ;;
  esac
  exit 0
fi

# Interactive mode
echo "=== PDF Q&A CLI ==="
echo "1) Ingest PDFs (build/update index)"
echo "2) Chat (ask questions)"
echo "3) Ingest then Chat"
echo "4) Exit"
echo

read -r -p "Choose an option [1-4]: " choice

case "${choice}" in
  1)
    read -r -p "PDF folder to ingest [default: ./pdfs]: " pdf_root
    pdf_root="${pdf_root:-./pdfs}"
    run_ingest "${pdf_root}"
    ;;
  2)
    run_chat
    ;;
  3)
    read -r -p "PDF folder to ingest [default: ./pdfs]: " pdf_root
    pdf_root="${pdf_root:-./pdfs}"
    run_ingest "${pdf_root}"
    echo
    run_chat
    ;;
  4)
    echo "Bye!"
    exit 0
    ;;
  *)
    echo "Invalid option."
    exit 1
    ;;
esac
