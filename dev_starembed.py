#!/usr/bin/env python3
"""Prepare sampled StarEmbed light curves for pipeline validation/testing.

Input:
  - data root (expects StarEmbed-style split folders with Arrow shards)
  - number of samples per class
  - output directory

  





python /home/rui/code/project/pipeline_agentic_light_curve/dev_starembed.py     /home/rui/data/timeseries/data_complete 10 /home/rui/code/project/pipeline_agentic_light_curve/input/starembed_sample



Output:
  - split-specific directories (train/val/test/anom) with:
      - light curve CSV files named "{ra}_{dec}.csv"
      - manifest CSV mapping labels and light-curve paths
  - top-level manifest CSV combining all splits
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

# Official StarEmbed/Catalina mapping:
# https://github.com/skai-institute/StarEmbed/blob/master/src/datasets/Catalina/Catalina_to_hf.py
CLASS_ID_TO_NAME = {
    "1": "EW",
    "2": "EA",
    "3": "Beta_Lyrae",
    "4": "RRab",
    "5": "RRc",
    "6": "RRd",
    "7": "Blazhko",
    "8": "RS CVn",
    "9": "ACEP",
    "10": "Cep-II",
    "11": "HADS",
    "12": "LADS",
    "13": "LPV",
    "14": "ELL",
    "15": "Hump",
    "16": "PCEB",
    "17": "EA_UP",
}

# In-distribution classes in your train/validation/test splits.
IN_DISTRIBUTION_CLASSES = ["1", "2", "4", "5", "6", "8", "13"]
SPLIT_OUTPUT_NAME = {"validation": "val", "train": "train", "test": "test", "anom": "anom"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample and export StarEmbed light curves.")
    parser.add_argument("data_root", type=Path, help="Path like ~/data/timeseries/data_complete")
    parser.add_argument("num_per_class", type=int, help="Number of samples per class")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for class sampling")
    return parser.parse_args()


def _arrow_paths(split_dir: Path) -> List[Path]:
    return [Path(p) for p in sorted(glob.glob(str(split_dir / "data-*.arrow")))]


def _iter_rows_from_split(split_dir: Path) -> Iterable[dict]:
    for arrow_path in _arrow_paths(split_dir):
        with pa.memory_map(str(arrow_path), "r") as source:
            reader = ipc.open_stream(source)
            for batch in reader:
                for row in batch.to_pylist():
                    yield row


def _load_coords(coords_csv: Path) -> Dict[int, tuple]:
    df = pd.read_csv(coords_csv, usecols=["panstarrs_source_id", "ra", "dec"])
    mapping: Dict[int, tuple] = {}
    for r in df.itertuples(index=False):
        mapping[int(r.panstarrs_source_id)] = (float(r.ra), float(r.dec))
    return mapping


def _format_coord(value: Optional[float]) -> str:
    if value is None:
        return "nan"
    return f"{value:.8f}"


def _build_filename(source_id: int, ra: Optional[float], dec: Optional[float]) -> str:
    if ra is None or dec is None or (not math.isfinite(ra)) or (not math.isfinite(dec)):
        return f"unknown_{source_id}.csv"
    return f"{_format_coord(ra)}_{_format_coord(dec)}.csv"


def _sample_rows(
    split_dir: Path,
    target_classes: Optional[List[str]],
    num_per_class: int,
    rng: random.Random,
) -> List[dict]:
    by_class: Dict[str, List[dict]] = defaultdict(list)
    rows_by_class: Dict[str, List[dict]] = defaultdict(list)

    for row in _iter_rows_from_split(split_dir):
        c = str(row["class_str"])
        if target_classes is not None and c not in target_classes:
            continue
        rows_by_class[c].append(row)

    classes = target_classes if target_classes is not None else sorted(rows_by_class.keys(), key=lambda x: int(x))
    for c in classes:
        candidates = rows_by_class.get(c, [])
        if not candidates:
            continue
        k = min(num_per_class, len(candidates))
        by_class[c] = rng.sample(candidates, k)

    sampled = []
    for c in classes:
        sampled.extend(by_class.get(c, []))
    return sampled


def _write_light_curve_csv(light_curve_path: Path, bands_data: dict) -> int:
    rows = []
    for band in sorted(bands_data.keys()):
        band_data = bands_data.get(band)
        if not band_data:
            continue
        mjd = band_data.get("mjd") or []
        mag = band_data.get("target") or []
        e_mag = band_data.get("past_feat_dynamic_real") or []
        n = min(len(mjd), len(mag), len(e_mag))
        for i in range(n):
            rows.append((float(mjd[i]), float(mag[i]), float(e_mag[i]), band))

    rows.sort(key=lambda x: x[0])
    light_curve_path.parent.mkdir(parents=True, exist_ok=True)
    with light_curve_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "mag", "e_mag", "band"])
        writer.writerows(rows)
    return len(rows)


def process_split(
    split_name: str,
    split_dir: Path,
    coords_map: Dict[int, tuple],
    num_per_class: int,
    out_root: Path,
    rng: random.Random,
) -> List[dict]:
    output_split = SPLIT_OUTPUT_NAME.get(split_name, split_name)
    split_out_dir = out_root / output_split
    lc_out_dir = split_out_dir / "light_curves"

    if split_name in ("train", "validation", "test"):
        target_classes = IN_DISTRIBUTION_CLASSES
    else:
        target_classes = None

    sampled_rows = _sample_rows(split_dir, target_classes, num_per_class, rng)
    manifests = []
    for row in sampled_rows:
        source_id = int(row["sourceid"])
        class_id = str(row["class_str"])
        class_name = CLASS_ID_TO_NAME.get(class_id, f"unknown_{class_id}")
        ra, dec = coords_map.get(source_id, (None, None))
        ra_s = _format_coord(ra)
        dec_s = _format_coord(dec)
        filename = _build_filename(source_id, ra, dec)
        lc_path = lc_out_dir / filename

        n_obs = _write_light_curve_csv(lc_path, row["bands_data"])
        manifests.append(
            {
                "split": output_split,
                "sourceid": source_id,
                "class_id": class_id,
                "class_name": class_name,
                "ra": ra_s,
                "dec": dec_s,
                "num_observations": n_obs,
                "light_curve_path": str(lc_path.resolve()),
            }
        )

    split_out_dir.mkdir(parents=True, exist_ok=True)
    split_manifest_path = split_out_dir / "manifest.csv"
    pd.DataFrame(manifests).to_csv(split_manifest_path, index=False)
    return manifests


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    out_root = args.output_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    coords_csv = data_root / "starembedcmGaia.csv"
    if not coords_csv.exists():
        raise FileNotFoundError(f"Missing coordinate file: {coords_csv}")

    coords_map = _load_coords(coords_csv)
    rng = random.Random(args.seed)

    all_records = []
    for split_name in ("train", "validation", "test", "anom"):
        split_dir = data_root / split_name
        if not split_dir.exists():
            continue
        records = process_split(
            split_name=split_name,
            split_dir=split_dir,
            coords_map=coords_map,
            num_per_class=args.num_per_class,
            out_root=out_root,
            rng=rng,
        )
        all_records.extend(records)

    manifest_path = out_root / "manifest.csv"
    pd.DataFrame(all_records).to_csv(manifest_path, index=False)
    print(f"Wrote {len(all_records)} sampled objects to: {manifest_path}")


if __name__ == "__main__":
    main()
