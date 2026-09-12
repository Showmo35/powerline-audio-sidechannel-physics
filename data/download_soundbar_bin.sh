#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# download_soundbar_bin.sh – Download soundbar_bin_captures from OneDrive to HPC
#
# Usage:  ./download_soundbar_bin.sh
#
# First run: auto-configures rclone OneDrive auth (needs a browser on any
#            machine — the script walks you through it).
# Subsequent runs: just copies new/missing files, no interaction needed.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REMOTE_NAME="onedrive"
REMOTE_PATHS=(
    "Powerline_Data_Captures/soundbar_bin_captures"
    "Documents/Powerline_Data_Captures/soundbar_bin_captures"
    "soundbar_bin_captures"
)
LOCAL_DIR="<REPO_ROOT>/data/soundbar_bin_captures"

die()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo "[INFO] $*"; }
hr()   { echo "────────────────────────────────────────────────────────────"; }

# ── 1. Ensure rclone is available ────────────────────────────────────────────
command -v rclone &>/dev/null || die "rclone not found (try: module load rclone)"

# ── 2. Auto-configure OneDrive remote if missing ─────────────────────────────
configure_remote() {
    hr
    echo "  OneDrive not configured — one-time setup (~ 1 minute)"
    hr
    echo ""
    echo "  On your LOCAL machine (laptop/desktop with a browser), run:"
    echo ""
    echo "      rclone authorize \"onedrive\""
    echo ""
    echo "  If rclone isn't installed locally, get it from rclone.org/install"
    echo "  Sign in with your Microsoft account when the browser opens."
    echo "  After signing in, copy the JSON token printed in the terminal."
    echo "  It looks like:  {\"access_token\":\"ey...\",\"refresh_token\":\"...\"}"
    echo ""
    echo "  Paste the token JSON below, then press Enter followed by Ctrl-D:"
    TOKEN_JSON=$(cat)
    echo ""

    [[ -z "$TOKEN_JSON" ]] && die "No token provided."

    # Validate JSON is complete
    python3 -c "import json,sys; json.loads(sys.stdin.read())" <<< "$TOKEN_JSON" \
        || die "Token JSON appears incomplete or invalid."

    # Extract access token and discover drive_id via MS Graph API
    ACCESS_TOKEN=$(python3 -c "import json,sys; print(json.loads(sys.stdin.read())['access_token'])" <<< "$TOKEN_JSON")
    info "Discovering OneDrive drive ID..."
    DRIVE_JSON=$(curl -sf -H "Authorization: Bearer $ACCESS_TOKEN" \
        "https://graph.microsoft.com/v1.0/me/drives" 2>/dev/null || echo "{}")
    DRIVE_ID=$(python3 -c "
import json, sys
data = json.loads(sys.stdin.read())
for d in data.get('value', []):
    if d.get('driveType') in ('business', 'personal', 'documentLibrary'):
        print(d['id']); break
" <<< "$DRIVE_JSON" 2>/dev/null || echo "")

    [[ -z "$DRIVE_ID" ]] && die "Could not discover drive ID — check your token and network."
    info "Drive ID: $DRIVE_ID"

    # Write config directly (avoids rclone's interactive flow)
    RCLONE_CONF="${HOME}/.config/rclone/rclone.conf"
    mkdir -p "$(dirname "$RCLONE_CONF")"
    printf '[%s]\ntype = onedrive\ntoken = %s\ndrive_type = business\ndrive_id = %s\n' \
        "$REMOTE_NAME" "$TOKEN_JSON" "$DRIVE_ID" > "$RCLONE_CONF"

    info "Remote '$REMOTE_NAME' configured."
    echo ""
}

if ! rclone listremotes 2>/dev/null | grep -q "^${REMOTE_NAME}:$"; then
    configure_remote
fi

# ── 3. Resolve the remote path ───────────────────────────────────────────────
REMOTE_DIR=""
for path in "${REMOTE_PATHS[@]}"; do
    if rclone lsd "${REMOTE_NAME}:${path}" &>/dev/null 2>&1 \
       || rclone ls  "${REMOTE_NAME}:${path}" &>/dev/null 2>&1; then
        REMOTE_DIR="$path"
        break
    fi
done

[[ -z "$REMOTE_DIR" ]] && die "Could not find soundbar_bin_captures on OneDrive.
Checked:
$(printf '  %s\n' "${REMOTE_PATHS[@]}")
Run 'rclone lsd ${REMOTE_NAME}:' to browse your OneDrive."

mkdir -p "$LOCAL_DIR"

# ── 4. Copy ───────────────────────────────────────────────────────────────────
hr
echo "  Source:  ${REMOTE_NAME}:${REMOTE_DIR}"
echo "  Dest:    ${LOCAL_DIR}"
hr
echo ""

rclone copy "${REMOTE_NAME}:${REMOTE_DIR}" "${LOCAL_DIR}" \
    --progress \
    --transfers 8 \
    --checkers 16 \
    --stats 5s \
    --ignore-existing

echo ""
info "Done. Files saved to: ${LOCAL_DIR}"
