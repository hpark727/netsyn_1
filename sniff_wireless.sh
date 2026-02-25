#!/usr/bin/env bash

set -euo pipefail

DURATION_SECONDS=185
INTERFACE="wlan0"
CHANNEL=""
CHANNEL_WIDTH=""
OUTPUT_FILE=""

usage() {
  cat <<'EOF'
Usage:
  ./sniff_wireless.sh -c <channel> -W <channel_width> [-i <interface>] [-o <capture.pcapng>] [-d <seconds>]

Description:
  Configures a wireless interface channel/channel-width (via iw), then runs tshark
  in monitor mode to sniff wireless frames.

Required:
  -c    Wi-Fi channel (example: 1, 6, 11, 36)
  -W    Channel width token passed to 'iw' (example: HT20, HT40+, HT40-, 80MHz)

Optional:
  -i    Wireless interface (default: wlan0)
  -o    Output capture file (.pcapng). If omitted, packets print to terminal.
  -d    Capture duration in seconds (default: 185)
  -h    Show this help

Examples:
  ./sniff_wireless.sh -i wlan0 -c 6 -W HT20
  ./sniff_wireless.sh -i wlan0mon -c 36 -W 80MHz -o capture.pcapng
EOF
}

while getopts ":i:c:W:o:d:h" opt; do
  case "$opt" in
    i) INTERFACE="$OPTARG" ;;
    c) CHANNEL="$OPTARG" ;;
    W) CHANNEL_WIDTH="$OPTARG" ;;
    o) OUTPUT_FILE="$OPTARG" ;;
    d) DURATION_SECONDS="$OPTARG" ;;
    h)
      usage
      exit 0
      ;;
    :)
      echo "Missing argument for -$OPTARG" >&2
      usage >&2
      exit 1
      ;;
    \?)
      echo "Unknown option: -$OPTARG" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$CHANNEL" || -z "$CHANNEL_WIDTH" ]]; then
  echo "Both -c <channel> and -W <channel_width> are required." >&2
  usage >&2
  exit 1
fi

if ! [[ "$DURATION_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "Duration must be an integer number of seconds." >&2
  exit 1
fi

for cmd in iw tshark; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Required command not found: $cmd" >&2
    exit 1
  fi
done

echo "Configuring interface '$INTERFACE' to channel $CHANNEL width $CHANNEL_WIDTH..."
echo "You may need root privileges for 'iw' and 'tshark' (run with sudo)."

# Set Wi-Fi channel and width. The width token is passed through directly to iw.
iw dev "$INTERFACE" set channel "$CHANNEL" "$CHANNEL_WIDTH"

echo "Starting tshark capture for ${DURATION_SECONDS}s on '$INTERFACE'..."
if [[ -n "$OUTPUT_FILE" ]]; then
  tshark -I -i "$INTERFACE" -a "duration:${DURATION_SECONDS}" -w "$OUTPUT_FILE"
  echo "Capture saved to: $OUTPUT_FILE"
else
  tshark -I -i "$INTERFACE" -a "duration:${DURATION_SECONDS}"
fi

