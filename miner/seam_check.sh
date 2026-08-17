SRC_URL="https://tile.loc.gov/storage-services/service/mbrs/ntscrm/00047382/00047382.mp4"
SRC_LOCAL="/tmp/manual-test-source.mp4"
DIR="/tmp/manual-test-seams"

curl -L "$SRC_URL" -o "$SRC_LOCAL"

rm -rf "$DIR"
mkdir -p "$DIR"

ffmpeg -hide_banner -y -i "$SRC_LOCAL" \
  -map 0:v:0 -map 0:a? -c copy -sn -dn \
  -f segment -segment_time 600 -reset_timestamps 1 -segment_format mp4 \
  "$DIR/input_%05d.mp4"

python3 - <<'PY'
import glob, subprocess

elapsed = 0.0
files = sorted(glob.glob("/tmp/manual-test-seams/input_*.mp4"))

def ts(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

for index, path in enumerate(files):
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ])
    duration = float(out.decode().strip())

    print(f"chunk {index + 1}: starts {ts(elapsed)}, duration {duration:.3f}s")
    elapsed += duration

    if index < len(files) - 1:
        print(f"  seam before chunk {index + 2}: {elapsed:.3f}s = {ts(elapsed)}")

print(f"final duration from chunks: {elapsed:.3f}s = {ts(elapsed)}")
PY

