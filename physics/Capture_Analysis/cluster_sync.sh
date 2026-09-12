#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# cluster_sync.sh  –  Upload new files from ~/Documents/monitor_bin_captures → cluster scratch storage
#
# Usage:
#   chmod +x cluster_sync.sh
#   ./cluster_sync.sh
#
# Runs until you close it (Ctrl-C). Syncs every 30s.
# Only uploads files not already on the cluster (--ignore-existing).
# Dependencies: apt install sshpass rsync
# ─────────────────────────────────────────────────────────────────────────────

LOCAL_DIR="$HOME/Documents/LibriSpeech_Capture/monitor_bin_captures"
CLUSTER_USER="user"
CLUSTER_HOST="<cluster_login_host>"
REMOTE_DIR="<REPO_ROOT>/data/Monitor"
POLL_INTERVAL=600

# Check deps
for cmd in sshpass rsync; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: '$cmd' not found. Run: sudo apt install $cmd"
        exit 1
    fi
done

# Prompt for password once at startup
read -rsp "cluster password for $CLUSTER_USER: " CLUSTER_PASS
echo
echo "Syncing every ${POLL_INTERVAL}s — press Ctrl-C to stop."

do_rsync() {
    sshpass -p "$CLUSTER_PASS" rsync -avz --ignore-existing \
        "$LOCAL_DIR/" \
        "${CLUSTER_USER}@${CLUSTER_HOST}:${REMOTE_DIR}/"
}

while true; do
    do_rsync
    sleep "$POLL_INTERVAL"
done
