"""
Rename GCA dataset feature keys to LeRobot / SmolVLA convention.

This dataset ships as:
  image, wrist_image, state, actions

LeRobot training expects:
  observation.images.*, observation.state, action

rename_map alone is not enough: batch_to_transition only keeps keys that
already start with "observation.".
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path.home() / ".cache/huggingface/lerobot/hyzhang01/GCA_parallel_adaptation_b01"

FEATURE_RENAME = {
    "image": "observation.images.image",
    "wrist_image": "observation.images.wrist_image",
    "state": "observation.state",
    "actions": "action",
}


def rename_columns(names: list[str]) -> list[str]:
    out = []
    for name in names:
        # Exact feature rename
        if name in FEATURE_RENAME:
            out.append(FEATURE_RENAME[name])
            continue
        # Episode-level stats/<feature>/<stat>
        if name.startswith("stats/"):
            rest = name[len("stats/") :]
            feat, _, suffix = rest.partition("/")
            if feat in FEATURE_RENAME and suffix:
                out.append(f"stats/{FEATURE_RENAME[feat]}/{suffix}")
                continue
        out.append(name)
    return out


def rewrite_parquet(path: Path) -> None:
    table = pq.read_table(path)
    new_names = rename_columns(table.column_names)
    if new_names == table.column_names:
        return
    table = table.rename_columns(new_names)
    # Drop embedded HF feature metadata so it doesn't keep stale names
    meta = dict(table.schema.metadata or {})
    meta.pop(b"huggingface", None)
    table = table.replace_schema_metadata(meta)
    pq.write_table(table, path)
    print(f"  renamed columns in {path}")


def rewrite_info_json(path: Path) -> None:
    info = json.loads(path.read_text())
    features = info.get("features", {})
    new_features = {}
    for key, value in features.items():
        new_features[FEATURE_RENAME.get(key, key)] = value
    info["features"] = new_features
    path.write_text(json.dumps(info, indent=4) + "\n")
    print(f"  updated {path}")


def rewrite_stats_json(path: Path) -> None:
    if not path.exists():
        return
    stats = json.loads(path.read_text())
    new_stats = {FEATURE_RENAME.get(k, k): v for k, v in stats.items()}
    path.write_text(json.dumps(new_stats, indent=4) + "\n")
    print(f"  updated {path}")


def main() -> None:
    if not ROOT.exists():
        raise SystemExit(f"Dataset root not found: {ROOT}")

    print(f"Renaming features under {ROOT}")
    rewrite_info_json(ROOT / "meta" / "info.json")
    rewrite_stats_json(ROOT / "meta" / "stats.json")

    for parquet_path in sorted(ROOT.rglob("*.parquet")):
        # tasks.parquet has no feature columns to rename
        if parquet_path.name == "tasks.parquet":
            continue
        rewrite_parquet(parquet_path)

    print("Done.")


if __name__ == "__main__":
    main()
