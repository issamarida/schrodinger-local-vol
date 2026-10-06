#!/usr/bin/env bash
# Second test period: the free August 2019 end-of-day sample of historicaloptiondata.com
# (DeltaNeutral LLC), reduced to SPX/SPXW rows, plus the US Treasury par curve for 2019.
#
# Licence: the vendor publishes no licence specific to its samples. The site terms grant
# "personal, non-commercial transitory viewing" only, so the quotes are never committed and
# results computed from them are kept out of the repository unless the vendor agrees.
set -euo pipefail

URL="https://www.dropbox.com/scl/fi/1a0tlv4do2ejx2v7cumhe/Sample_L2_2019_August.zip?rlkey=ibxlrbngoirxke47w0bfx1r3y&dl=1"
TREASURY="https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/2019/all?type=daily_treasury_yield_curve&field_tdr_date_value=2019&page&_format=csv"
DEST="$(cd "$(dirname "$0")/.." && pwd)/data/raw_2019_08"
ZIP="$DEST/Sample_L2_2019_August.zip"

mkdir -p "$DEST"
if [[ ! -f "$ZIP" ]]; then
  echo "Downloading the August 2019 sample (~600 MB)..."
  curl -fL --retry 3 -A "Mozilla/5.0" -o "$ZIP.part" "$URL"
  mv "$ZIP.part" "$ZIP"
fi

echo "Extracting SPX/SPXW rows..."
for name in $(unzip -Z1 "$ZIP" | grep '^L2_options_'); do
  unzip -p "$ZIP" "$name" | awk -F, 'NR == 1 || $1 == "SPX" || $1 == "SPXW"' > "$DEST/$name"
done

echo "Downloading the 2019 Treasury par yield curve..."
curl -fsL -A "Mozilla/5.0" -o "$DEST/treasury_par_curve_2019.csv" "$TREASURY"
ls "$DEST"/L2_options_*.csv | wc -l | xargs echo "trading days:"
