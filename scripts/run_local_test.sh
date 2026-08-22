#!/usr/bin/env bash
# ==============================================================================
# Edge AI CCTV - One-Click Local PC Test Runner
# ==============================================================================
# Runs the full Edge AI CCTV pipeline locally on your computer using:
# • Your ESP32-S3 IP Camera (Wi-Fi or USB)
# • Local USB Webcam (/dev/video0)
# • Or Built-In Synthetic Benchmark Stream
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/.venv_test"

echo "======================================================="
echo "   🛡️ Edge AI CCTV Local Pipeline Benchmark & Test"
echo "======================================================="

cd "$PROJECT_ROOT"

# 1. Check or Create Virtual Environment
if [ ! -d "$VENV_DIR" ]; then
  echo "[+] Creating local Python virtual environment in $VENV_DIR..."
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# 2. Install / Verify Dependencies
echo "[+] Checking Python test dependencies..."
pip install --quiet --upgrade pip
pip install --quiet \
    opencv-python-headless \
    numpy \
    pydantic \
    pydantic-settings \
    sqlalchemy \
    aiosqlite \
    fastapi \
    uvicorn \
    pyjwt \
    passlib \
    bcrypt \
    httpx

# 3. Parse Custom Stream URL if provided
STREAM_URL="${1:-}"

if [ -n "$STREAM_URL" ]; then
  echo "[+] Testing with custom stream: $STREAM_URL"
  python3 "$PROJECT_ROOT/scripts/test_local_system.py" --stream "$STREAM_URL" --duration 10
else
  echo "[+] No stream URL passed. Testing with USB webcam / Synthetic generator..."
  echo "    (Tip: Pass your ESP32 IP stream as: ./scripts/run_local_test.sh http://192.168.1.150:81/stream)"
  python3 "$PROJECT_ROOT/scripts/test_local_system.py" --duration 10
fi

echo "======================================================="
echo " ✅ Test finished! Recorded clips are saved in:"
echo "    • $PROJECT_ROOT/storage/clips/"
echo "    • $PROJECT_ROOT/storage/dvr/"
echo "======================================================="
