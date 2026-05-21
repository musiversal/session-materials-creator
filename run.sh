#!/usr/bin/env bash
# run.sh — single entry point for audio-tools on this machine.
#
#   ./run.sh -i song.wav --skip-sections   # CLI pipeline
#   ./run.sh web                           # start web UI (default port 3001)
#   PORT=4000 ./run.sh web                 # web UI on custom port
#   ./run.sh tunnel                        # web UI + Cloudflare quick tunnel
#   ./run.sh url                           # print the currently active tunnel URL
#
# venv_allin1 is not installed here, so always pass --skip-sections on the CLI
# (or set AUDIO_TOOLS_SKIP_SECTIONS=1 to default it).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

STABILIZER_PY="$SCRIPT_DIR/venv_stabilizer/bin/python"
export AUDIO_TOOLS_PYTHON="$STABILIZER_PY"

PORT="${PORT:-3001}"
export PORT

CLOUDFLARED="${CLOUDFLARED:-$HOME/.local/bin/cloudflared}"

if [[ "${1:-}" == "web" ]]; then
    shift
    cd web
    exec npm run dev "$@"
fi

if [[ "${1:-}" == "url" ]]; then
    TUNNEL_LOG="$SCRIPT_DIR/.run-logs/tunnel.log"
    if [[ ! -f "$TUNNEL_LOG" ]]; then
        echo "No tunnel log found at $TUNNEL_LOG — is the service running?" >&2
        exit 1
    fi
    URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" | tail -1)
    if [[ -z "$URL" ]]; then
        echo "No tunnel URL in $TUNNEL_LOG yet — tunnel may still be starting." >&2
        exit 1
    fi
    echo "$URL"
    exit 0
fi

if [[ "${1:-}" == "tunnel" ]]; then
    shift
    if [[ ! -x "$CLOUDFLARED" ]] && ! command -v cloudflared >/dev/null 2>&1; then
        echo "cloudflared not found at $CLOUDFLARED or on PATH" >&2
        exit 1
    fi
    [[ -x "$CLOUDFLARED" ]] || CLOUDFLARED="$(command -v cloudflared)"

    LOG_DIR="$SCRIPT_DIR/.run-logs"
    mkdir -p "$LOG_DIR"
    WEB_LOG="$LOG_DIR/web.log"
    TUNNEL_LOG="$LOG_DIR/tunnel.log"
    : > "$WEB_LOG"
    : > "$TUNNEL_LOG"

    cleanup() {
        local code=$?
        [[ -n "${WEB_PID:-}" ]]    && kill "$WEB_PID"    2>/dev/null || true
        [[ -n "${TUNNEL_PID:-}" ]] && kill "$TUNNEL_PID" 2>/dev/null || true
        exit "$code"
    }
    trap cleanup INT TERM EXIT

    echo "▶  Starting web UI on http://localhost:$PORT …"
    ( cd web && npm run dev ) >"$WEB_LOG" 2>&1 &
    WEB_PID=$!

    # Wait for Next.js "Ready"
    for _ in $(seq 1 60); do
        grep -q "Ready in" "$WEB_LOG" 2>/dev/null && break
        sleep 0.5
    done
    if ! grep -q "Ready in" "$WEB_LOG"; then
        echo "✗  Web UI didn't become ready. See $WEB_LOG" >&2
        tail -20 "$WEB_LOG" >&2
        exit 1
    fi
    echo "✓  Web UI ready."

    echo "▶  Starting Cloudflare quick tunnel …"
    "$CLOUDFLARED" tunnel --no-autoupdate --url "http://localhost:$PORT" >"$TUNNEL_LOG" 2>&1 &
    TUNNEL_PID=$!

    # Cloudflared writes the URL to its log after a few seconds
    PUBLIC_URL=""
    for _ in $(seq 1 60); do
        PUBLIC_URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" | head -1 || true)
        [[ -n "$PUBLIC_URL" ]] && break
        sleep 0.5
    done
    if [[ -z "$PUBLIC_URL" ]]; then
        echo "✗  Tunnel did not produce a URL. See $TUNNEL_LOG" >&2
        tail -30 "$TUNNEL_LOG" >&2
        exit 1
    fi

    # Notify Slack with the new URL (best-effort — never fails the service)
    SLACK_WEBHOOK_FILE="${SLACK_WEBHOOK_FILE:-$HOME/.config/audio-tools/slack-webhook}"
    if [[ -s "$SLACK_WEBHOOK_FILE" ]]; then
        SLACK_URL="$(head -n1 "$SLACK_WEBHOOK_FILE" | tr -d '[:space:]')"
        if [[ -n "$SLACK_URL" ]]; then
            HOSTNAME_SHORT="$(hostname -s 2>/dev/null || hostname)"
            PAYLOAD=$(printf '{"text":"🎧 *audio-tools* is up on `%s` → %s"}' "$HOSTNAME_SHORT" "$PUBLIC_URL")
            curl -sS -X POST -H 'Content-Type: application/json' \
                --data "$PAYLOAD" --max-time 10 \
                "$SLACK_URL" >/dev/null 2>&1 \
                && echo "✓  Slack notified." \
                || echo "⚠  Slack notification failed (see logs)."
        fi
    fi

    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Public URL:  $PUBLIC_URL"
    echo "║  Local URL:   http://localhost:$PORT"
    echo "║  Logs:        $WEB_LOG"
    echo "║               $TUNNEL_LOG"
    echo "║  Stop:        Ctrl-C"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""

    wait
fi

DEFAULT_ARGS=()
if [[ "${AUDIO_TOOLS_SKIP_SECTIONS:-0}" == "1" ]]; then
    DEFAULT_ARGS+=("--skip-sections")
fi

exec "$STABILIZER_PY" pipeline.py "${DEFAULT_ARGS[@]}" "$@"
