#!/usr/bin/env bash
set -euo pipefail

DURATION_SECONDS=185
INTERFACE="en0"
OUTPUT_DIR=""
BASENAME=""
CHANNEL=""        # accepted, but may not be enforceable via CLI on modern macOS
CHANNEL_WIDTH=""  # accepted, but may not be enforceable via CLI on modern macOS

usage() {
  cat <<'EOF'
Usage:
  sudo ./sniff_macos_tshark.sh -O <output_dir> [-i <iface>] [-d <seconds>] [-b <basename>] [-c <channel>] [-W <width>]

What it does:
  - Captures using tshark on macOS (tries monitor-mode 802.11 + radiotap)
  - Saves: <output_dir>/<basename>.pcapng
  - Exports: <output_dir>/<basename>.csv with columns:
      time_epoch, frame_len, fc_type, fc_subtype, duration, sa, da, rssi_dbm, noise_dbm

Notes:
  - On newer macOS, setting channel/width from CLI may not be possible (airport removed).
    -c/-W are accepted for bookkeeping/logging, but may not be applied.

Examples:
  sudo ./sniff_macos_tshark.sh -O ./captures -d 60 -b test1
  sudo ./sniff_macos_tshark.sh -O ./captures -i en0 -c 149 -W 40MHz -d 60 -b ch149
EOF
}

while getopts ":O:i:d:b:c:W:h" opt; do
  case "$opt" in
    O) OUTPUT_DIR="$OPTARG" ;;
    i) INTERFACE="$OPTARG" ;;
    d) DURATION_SECONDS="$OPTARG" ;;
    b) BASENAME="$OPTARG" ;;
    c) CHANNEL="$OPTARG" ;;
    W) CHANNEL_WIDTH="$OPTARG" ;;
    h) usage; exit 0 ;;
    :) echo "Missing argument for -$OPTARG" >&2; usage >&2; exit 1 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
  echo "Error: -O <output_dir> is required." >&2
  usage >&2
  exit 1
fi

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Error: This script is for macOS (Darwin)." >&2
  exit 1
fi

if ! [[ "$DURATION_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "Error: duration must be an integer number of seconds." >&2
  exit 1
fi

for cmd in tshark awk sed tail mkdir; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "Missing required command: $cmd" >&2; exit 1; }
done

mkdir -p "$OUTPUT_DIR"
if [[ -z "$BASENAME" ]]; then
  BASENAME="capture_$(date +%Y%m%d_%H%M%S)"
fi

PCAP_PATH="${OUTPUT_DIR%/}/${BASENAME}.pcapng"
CSV_PATH="${OUTPUT_DIR%/}/${BASENAME}.csv"

echo "macOS tshark capture"
echo "  iface   : $INTERFACE"
echo "  duration: ${DURATION_SECONDS}s"
if [[ -n "$CHANNEL" || -n "$CHANNEL_WIDTH" ]]; then
  echo "  requested channel/width: ${CHANNEL:-<none>} ${CHANNEL_WIDTH:-<none>}"
  echo "  note: channel/width may not be settable via CLI on recent macOS"
fi
echo "  pcap    : $PCAP_PATH"
echo "  csv     : $CSV_PATH"
echo

# Pick a link-layer type that yields 802.11+radiotap if possible
# tshark -L lists supported link-layer types for the interface
LINKTYPE=""
if tshark -i "$INTERFACE" -L 2>/dev/null | grep -q "IEEE802_11_RADIO"; then
  LINKTYPE="IEEE802_11_RADIO"
elif tshark -i "$INTERFACE" -L 2>/dev/null | grep -q "IEEE802_11"; then
  LINKTYPE="IEEE802_11"
else
  LINKTYPE="" # fall back to default (often Ethernet-like)
fi

echo "Detected link-layer for $INTERFACE: ${LINKTYPE:-<default>}"
echo "Starting capture (requires sudo)..."

if [[ -n "$LINKTYPE" ]]; then
  sudo tshark -I -i "$INTERFACE" -y "$LINKTYPE" -a "duration:${DURATION_SECONDS}" -w "$PCAP_PATH"
else
  echo "Warning: No 802.11 link-layer type exposed by tshark for $INTERFACE."
  echo "         Capture may be 'Ethernet-like' and miss 802.11 headers/RSSI/noise."
  sudo tshark -i "$INTERFACE" -a "duration:${DURATION_SECONDS}" -w "$PCAP_PATH"
fi

echo "Capture saved: $PCAP_PATH"
echo

# Choose RSSI/Noise fields if present in this tshark build
has_field() { tshark -G fields 2>/dev/null | awk -v pat="$1" '$0 ~ pat {f=1} END{exit f?0:1}'; }

RSSI_FIELD=""
NOISE_FIELD=""

# Prefer radiotap fields when available
if has_field "radiotap\\.dbm_antsignal"; then RSSI_FIELD="radiotap.dbm_antsignal"; fi
if has_field "radiotap\\.dbm_antnoise";  then NOISE_FIELD="radiotap.dbm_antnoise";  fi

# Fall back to wlan_radio
if [[ -z "$RSSI_FIELD" ]] && has_field "wlan_radio\\.signal_dbm"; then RSSI_FIELD="wlan_radio.signal_dbm"; fi
if [[ -z "$NOISE_FIELD" ]] && has_field "wlan_radio\\.noise_dbm";  then NOISE_FIELD="wlan_radio.noise_dbm";  fi

echo "Exporting CSV..."
echo "  RSSI field : ${RSSI_FIELD:-<not available -> blank>}"
echo "  Noise field: ${NOISE_FIELD:-<not available -> blank>}"
echo

BASE_FIELDS=(
  "frame.time_epoch"
  "frame.len"
  "wlan.fc.type"
  "wlan.fc.subtype"
  "wlan.duration"
  "wlan.sa"
  "wlan.da"
)

cmd=(tshark -r "$PCAP_PATH" -T fields -E header=y -E separator=, -E quote=d -E occurrence=f)
for f in "${BASE_FIELDS[@]}"; do cmd+=(-e "$f"); done
cmd+=(-e "${RSSI_FIELD:-fake.rssi_dbm}" -e "${NOISE_FIELD:-fake.noise_dbm}")

"${cmd[@]}" > "${CSV_PATH}.tmp"

{
  echo 'time_epoch,frame_len,fc_type,fc_subtype,duration,sa,da,rssi_dbm,noise_dbm'
  tail -n +2 "${CSV_PATH}.tmp"
} > "$CSV_PATH"

rm -f "${CSV_PATH}.tmp"

echo "CSV saved: $CSV_PATH"
echo "Done."