#!/bin/zsh
# run_models.sh — Separate songs with each Demucs model and time each run.
#
#   bench/run_models.sh song1.mp3 song2.m4a ...
#   DEMUCS_MODELS="htdemucs htdemucs_ft" bench/run_models.sh song.mp3
#   python3 bench/make_review.py && open bench/review.html
#
# Output: bench/out/<song>/{original.wav, times.txt, demucs_<m>/...}
# Models are downloaded/warmed once before timing, so times are pure inference.

R="$(cd "$(dirname "$0")/.." && pwd)"
OUT_ROOT="$R/bench/out"
WARM="$OUT_ROOT/.warm"; mkdir -p "$WARM"

DEMUCS_MODELS=(${=DEMUCS_MODELS:-htdemucs})

t() {  # t <name> <outdir> cmd...
  local name=$1 out=$2; shift 2
  local s=$(date +%s.%N)
  "$@" > "$out/$name.log" 2>&1; local rc=$?
  local e=$(date +%s.%N)
  printf "%-28s rc=%d  %6.1fs\n" "$name" $rc $((e - s)) | tee -a "$out/times.txt"
}

# Warm-up: download every model once on a 3 s clip.
ffmpeg -v error -y -i "$1" -t 3 -ar 44100 -ac 2 "$WARM/warm.wav"
for m in $DEMUCS_MODELS; do
  "$R/venv_demucs/bin/python" -m demucs -n $m -o "$WARM" "$WARM/warm.wav" > /dev/null 2>&1
done

for f in "$@"; do
  name="${${f:t}:r}"
  out="$OUT_ROOT/$name"; mkdir -p "$out"; : > "$out/times.txt"
  ffmpeg -v error -y -i "$f" -ar 44100 -ac 2 "$out/original.wav"
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$out/original.wav")
  echo "== $name (${dur}s)"
  for m in $DEMUCS_MODELS; do
    t demucs_$m "$out" "$R/venv_demucs/bin/python" -m demucs -n $m --overlap 0 --shifts 0 \
      -o "$out/demucs_$m" "$out/original.wav"
  done
done
