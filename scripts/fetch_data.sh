#!/usr/bin/env bash
# Downloads the free SPX/SPXW end-of-day options sample (July-December 2022) from
# HistoricalData.net and verifies every file against the vendor's published SHA-256 manifest.
#
# The data is licensed for research use but NOT for redistribution, which is why it is fetched
# here rather than committed. See data/raw/LICENSE.txt after download.
set -euo pipefail

URL="https://historicaldata.net/file/options_sample_2022H2.zip"
DEST="$(cd "$(dirname "$0")/.." && pwd)/data/raw"
ZIP="$DEST/options_sample_2022H2.zip"

mkdir -p "$DEST"
if [[ ! -f "$ZIP" ]]; then
  echo "Downloading $URL (~240 MB)..."
  curl -fL --retry 3 -A "Mozilla/5.0" -o "$ZIP.part" "$URL"
  mv "$ZIP.part" "$ZIP"
fi

echo "Extracting..."
unzip -oq "$ZIP" -d "$DEST"

echo "Verifying checksums..."
(cd "$DEST" && python3 verify.py day_by_date)
