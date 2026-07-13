#!/usr/bin/env python3
"""Merge source manifests and split into train/val JSONL files."""

from __future__ import annotations

import argparse
import json
import pathlib
import random


def _load_jsonl(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_root", default="data/processed/manifests")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_total", type=int, default=400)
    args = parser.parse_args()

    manifest_root = pathlib.Path(args.manifest_root)
    libri = _load_jsonl(manifest_root / "libritts_small.jsonl")
    cv = _load_jsonl(manifest_root / "common_voice_en_small.jsonl")

    mixed = libri + cv
    random.seed(args.seed)
    random.shuffle(mixed)

    if args.max_total > 0:
        mixed = mixed[: args.max_total]

    val_size = int(len(mixed) * args.val_ratio)
    val_rows = mixed[:val_size]
    train_rows = mixed[val_size:]

    _write_jsonl(manifest_root / "train_small.jsonl", train_rows)
    _write_jsonl(manifest_root / "val_small.jsonl", val_rows)

    print(f"Merged total rows: {len(mixed)}")
    print(f"Train rows: {len(train_rows)}")
    print(f"Val rows: {len(val_rows)}")


if __name__ == "__main__":
    main()
