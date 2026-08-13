#!/usr/bin/env bash
# ig_encode.sh — normalize / stitch ComfyUI clips into an Instagram-ready vertical MP4.
#
# Output spec: 1080x1920 (9:16), 30 fps, H.264 high yuv420p + AAC 48 kHz, +faststart.
# Accepts any ffmpeg-readable inputs (mp4 / webm / webp / gif ...); each clip is
# scaled to fit and letterboxed if its aspect ratio differs.
#
# Audio policy (predictable for automation): source-clip audio is always dropped.
#   -a FILE  -> FILE is looped/trimmed to the video length with a 1.5 s fade-out.
#   no -a    -> a silent stereo track is added (IG expects an audio stream).
#
# Usage:
#   ig_encode.sh -o reel.mp4 [-a bgm.mp3] [-s 1080x1920] [-f 30] [-q 18] clip1 [clip2 ...]
set -euo pipefail

OUT="" AUDIO="" SIZE="1080x1920" FPS=30 CRF=18
usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

while getopts "o:a:s:f:q:h" opt; do
  case $opt in
    o) OUT=$OPTARG ;;
    a) AUDIO=$OPTARG ;;
    s) SIZE=$OPTARG ;;
    f) FPS=$OPTARG ;;
    q) CRF=$OPTARG ;;
    *) usage ;;
  esac
done
shift $((OPTIND - 1))

[ -n "$OUT" ] && [ $# -ge 1 ] || usage
[ -z "$AUDIO" ] || [ -f "$AUDIO" ] || { echo "[ig] audio not found: $AUDIO" >&2; exit 1; }
W=${SIZE%x*}; H=${SIZE#*x}

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# 1) Normalize every clip to identical codec/size/fps so concat is lossless.
i=0
for clip in "$@"; do
  [ -f "$clip" ] || { echo "[ig] input not found: $clip" >&2; exit 1; }
  i=$((i + 1))
  seg=$(printf '%s/seg_%03d.mp4' "$TMP" "$i")
  echo "[ig] normalizing ($i/$#): $clip"
  ffmpeg -hide_banner -loglevel error -y -i "$clip" \
    -vf "scale=${W}:${H}:force_original_aspect_ratio=decrease,\
pad=${W}:${H}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,fps=${FPS}" \
    -c:v libx264 -preset medium -crf "$CRF" -pix_fmt yuv420p -an "$seg"
  printf "file '%s'\n" "$seg" >> "$TMP/list.txt"
done

# 2) Concat (stream copy — segments are already identical).
if [ "$i" -eq 1 ]; then
  cp "$TMP/seg_001.mp4" "$TMP/joined.mp4"
else
  ffmpeg -hide_banner -loglevel error -y -f concat -safe 0 \
    -i "$TMP/list.txt" -c copy "$TMP/joined.mp4"
fi

DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$TMP/joined.mp4")

# 3) Mux audio.
mkdir -p "$(dirname "$OUT")"
if [ -n "$AUDIO" ]; then
  FADE_ST=$(awk "BEGIN{d=$DUR-1.5; if (d<0) d=0; printf \"%.2f\", d}")
  ffmpeg -hide_banner -loglevel error -y \
    -i "$TMP/joined.mp4" -stream_loop -1 -i "$AUDIO" \
    -map 0:v -map 1:a -t "$DUR" \
    -c:v copy -c:a aac -b:a 192k -ar 48000 -ac 2 \
    -af "afade=t=out:st=${FADE_ST}:d=1.5" \
    -movflags +faststart "$OUT"
else
  ffmpeg -hide_banner -loglevel error -y \
    -i "$TMP/joined.mp4" \
    -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=48000 \
    -map 0:v -map 1:a -t "$DUR" \
    -c:v copy -c:a aac -b:a 128k \
    -movflags +faststart "$OUT"
fi

# 4) Verify against IG Reels expectations and report.
read -r OW OH < <(ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height -of csv=p=0 "$OUT" | tr ',' ' ')
ODUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT")
OSIZE=$(stat -c%s "$OUT" 2>/dev/null || stat -f%z "$OUT")

echo "[ig] wrote $OUT"
echo "[ig] ${OW}x${OH}, $(printf '%.1f' "$ODUR")s, $((OSIZE / 1024 / 1024)) MB, ${FPS}fps, H.264+AAC"
awk "BEGIN{r=$OW/$OH; exit (r>0.556 && r<0.569) ? 0 : 1}" \
  || echo "[ig] WARN: aspect ratio is not 9:16 — IG will crop or letterbox"
awk "BEGIN{exit ($ODUR <= 180) ? 0 : 1}" \
  || echo "[ig] WARN: longer than 180 s (Reels max is 3 min)"
[ "$OSIZE" -le $((4 * 1024 * 1024 * 1024)) ] \
  || echo "[ig] WARN: file exceeds the 4 GB upload limit"
