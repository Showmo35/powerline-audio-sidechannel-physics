#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# onedrive_sync.sh  –  Upload new files from ~/Documents/monitor_bin_captures → OneDrive
#
# Usage:
#   chmod +x onedrive_sync.sh
#   ./onedrive_sync.sh
#
# Runs until you close it (Ctrl-C). Syncs every 10 minutes.
# Only uploads files not already on OneDrive (--ignore-existing).
# Dependencies: apt install rclone  +  rclone config (run once)
# ─────────────────────────────────────────────────────────────────────────────

LOCAL_DIR="$HOME/Documents/LibriSpeech_Capture/monitor_bin_captures"
REMOTE_NAME="onedrive"
REMOTE_DIR="Powerline_Data_Captures/Monitor"
POLL_INTERVAL=600

# Check dep
if ! command -v rclone &>/dev/null; then
    echo "ERROR: rclone not found. Run: sudo apt install rclone"
    exit 1
fi

echo "Syncing to OneDrive every $((POLL_INTERVAL/60)) minutes — press Ctrl-C to stop."

do_rclone() {
    rclone copy "$LOCAL_DIR/" "${REMOTE_NAME}:${REMOTE_DIR}/" \
        --ignore-existing \
        --progress \
        --transfers 4
}

while true; do
    do_rclone
    sleep "$POLL_INTERVAL"
done
