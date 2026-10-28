#!/bin/bash
# Download original LIBERO HDF5 demos (with sim states) and write a manifest of local paths.
# Usage: scripts/a0_download_libero.sh <raw_dir> <manifest_out> [suite ...]   (default: spatial object goal)
set -euo pipefail
RAW=$1; MANIFEST=$2; shift 2
SUITES=${*:-libero_spatial libero_object libero_goal}
BASE=https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets
: > "$MANIFEST.tmp"
for s in $SUITES; do
  mkdir -p "$RAW/$s"
  curl -sf "https://huggingface.co/api/datasets/yifengzhu-hf/LIBERO-datasets/tree/main/$s" \
    | ${PYTHON:-python3} -c 'import json,sys; [print(x["path"], x["size"]) for x in json.load(sys.stdin) if x["path"].endswith(".hdf5")]' \
    | while read -r path size; do
        dst="$RAW/$path"
        if [ ! -s "$dst" ] || [ "$(stat -c %s "$dst")" != "$size" ]; then
          curl -sfL -o "$dst.part" "$BASE/resolve/main/$path" && mv "$dst.part" "$dst"
        fi
        [ "$(stat -c %s "$dst")" = "$size" ] || { echo "size mismatch: $dst" >&2; exit 1; }
        echo "$dst" >> "$MANIFEST.tmp"
      done
done
sort "$MANIFEST.tmp" > "$MANIFEST" && rm "$MANIFEST.tmp"
echo "$(wc -l < "$MANIFEST") files in $MANIFEST"
