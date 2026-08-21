#!/usr/bin/env bash
# Download the Kaggle Avazu CTR dataset and convert it to day-partitioned parquet.
#
# Prerequisites:
#   - `pip install -r requirements.txt` (kaggle, pandas, pyarrow)
#   - Kaggle API token at ~/.kaggle/kaggle.json
#   - Competition rules accepted at https://www.kaggle.com/c/avazu-ctr-prediction
#
# Output layout (raw CSV is never re-read after conversion):
#   data/raw/train.gz
#   data/parquet/day=21/part-00000.parquet ... day=30/...
set -euo pipefail

cd "$(dirname "$0")"
PYBIN="${PYTHON:-python}"

mkdir -p raw parquet

if [ ! -f raw/train.gz ]; then
    echo "==> Downloading avazu-ctr-prediction from Kaggle..."
    kaggle competitions download -c avazu-ctr-prediction -p raw
    unzip -o raw/avazu-ctr-prediction.zip -d raw
else
    echo "==> raw/train.gz already present, skipping download."
fi

echo "==> Converting to day-partitioned parquet (chunked, ~40M rows)..."
"$PYBIN" - <<'PY'
import glob
import os

import pandas as pd

RAW = "raw/train.gz"
OUT = "parquet"
CHUNK_ROWS = 2_000_000  # ~40M rows total -> ~20 chunks; keeps peak memory modest

if glob.glob(os.path.join(OUT, "day=*", "*.parquet")):
    print("parquet partitions already exist; delete data/parquet/ to rebuild.")
    raise SystemExit(0)

# Integer-coded columns stay integers; hashed/hex identifiers stay strings.
DTYPES = {
    "click": "int8",
    "hour": "int32",  # YYMMDDHH
    "C1": "int32",
    "banner_pos": "int32",
    "site_id": "string", "site_domain": "string", "site_category": "string",
    "app_id": "string", "app_domain": "string", "app_category": "string",
    "device_id": "string", "device_ip": "string", "device_model": "string",
    "device_type": "int32", "device_conn_type": "int32",
    **{f"C{i}": "int32" for i in range(14, 22)},
}

reader = pd.read_csv(
    RAW,
    compression="gzip",
    usecols=list(DTYPES),  # drop the row `id`
    dtype=DTYPES,
    chunksize=CHUNK_ROWS,
)

total = 0
for i, chunk in enumerate(reader):
    day = (chunk["hour"] // 100) % 100  # YYMMDDHH -> DD
    for d, part in chunk.groupby(day):
        ddir = os.path.join(OUT, f"day={d:02d}")
        os.makedirs(ddir, exist_ok=True)
        part.to_parquet(
            os.path.join(ddir, f"part-{i:05d}.parquet"),
            index=False,
            engine="pyarrow",
        )
    total += len(chunk)
    print(f"  chunk {i:2d}: {total:>10,} rows written")

print(f"done: {total:,} rows -> {OUT}/day=*/")
PY

echo "==> Done. Partitions:"
du -sh parquet/day=* 2>/dev/null || true
