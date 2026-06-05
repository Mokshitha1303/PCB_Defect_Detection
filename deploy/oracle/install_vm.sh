#!/usr/bin/env bash
# install_vm.sh — Bootstrap Docker + deploy PCB Defect Inspector on Oracle Ubuntu Arm64
# Run as: sudo bash install_vm.sh

set -euo pipefail

APP_DIR="/opt/pcb-inspector"
REPO_URL="${REPO_URL:-}"   # set to your git repo URL if cloning from git
IMAGE_TAG="${IMAGE_TAG:-pcb-defect-yolo26:latest}"

echo "=== PCB Defect Inspector — VM Bootstrap ==="

# ── 1. System update ──────────────────────────────────────────────────────────
apt-get update && apt-get upgrade -y

# ── 2. Install Docker ─────────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
  echo "[*] Installing Docker …"
  curl -fsSL https://get.docker.com | sh
  usermod -aG docker ubuntu || true
  systemctl enable docker
  systemctl start docker
fi

# ── 3. Install docker compose plugin ─────────────────────────────────────────
if ! docker compose version &>/dev/null; then
  apt-get install -y docker-compose-plugin
fi

# ── 4. Create app directory ───────────────────────────────────────────────────
mkdir -p "$APP_DIR/runs" "$APP_DIR/data/samples" "$APP_DIR/configs"

# ── 5. Copy app files (if not cloning from git) ───────────────────────────────
# Assumes you scp'd the project tarball to /tmp/pcb-inspector.tar.gz
if [ -f /tmp/pcb-inspector.tar.gz ]; then
  tar -xzf /tmp/pcb-inspector.tar.gz -C "$APP_DIR" --strip-components=1
fi

# ── 6. Build image ────────────────────────────────────────────────────────────
cd "$APP_DIR"
docker build -t "$IMAGE_TAG" .

# ── 7. Copy .env if not present ───────────────────────────────────────────────
if [ ! -f "$APP_DIR/.env" ]; then
  cp "$APP_DIR/.env.example" "$APP_DIR/.env"
  echo "[!] Review and update $APP_DIR/.env before starting."
fi

echo "[*] Bootstrap complete."
echo "    Run: cd $APP_DIR && docker compose up -d"
