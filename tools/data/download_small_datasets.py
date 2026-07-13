#!/usr/bin/env python3
"""Download small educational subsets from two public speech datasets.

Outputs files into:
  - data/downloaded/libritts
  - data/downloaded/common_voice_en
"""

from __future__ import annotations

import argparse
import pathlib
import shutil

from datasets import Audio, load_dataset


def _copy_rows(
    dataset,
    out_dir: pathlib.Path,
    transcript_key: str,
    limit: int,
    prefix: str,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = dataset.cast_column("audio", Audio(decode=False))

    copied = 0
    for idx, row in enumerate(dataset):
        if copied >= limit:
            break
        if "audio" not in row or not row["audio"]:
            continue
        if transcript_key not in row:
            continue

        audio_info = row["audio"]
        src_path = audio_info.get("path") if isinstance(audio_info, dict) else None
        audio_bytes = audio_info.get("bytes") if isinstance(audio_info, dict) else None
        transcript = (row.get(transcript_key) or "").strip()
        if not transcript:
            continue

        suffix = ".wav"
        if src_path:
            suffix = pathlib.Path(src_path).suffix or suffix

        dst = out_dir / f"{prefix}_{copied:05d}{suffix}"
        if src_path and pathlib.Path(src_path).exists():
            src = pathlib.Path(src_path)
            shutil.copy2(src, dst)
        elif audio_bytes:
            dst.write_bytes(audio_bytes)
        else:
            continue
        txt = out_dir / f"{prefix}_{copied:05d}.txt"
        txt.write_text(transcript, encoding="utf-8")
        copied += 1

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1} rows for {prefix}; kept {copied}.")

    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_root",
        default="data/downloaded",
        help="Output directory root.",
    )
    parser.add_argument(
        "--dataset_mode",
        choices=["dummy", "full"],
        default="dummy",
        help="Dataset source mode. Use 'dummy' for fast smoke tests.",
    )
    parser.add_argument(
        "--libri_limit",
        type=int,
        default=200,
        help="Number of LibriSpeech files to keep.",
    )
    parser.add_argument(
        "--cv_limit",
        type=int,
        default=200,
        help="Number of Common Voice files to keep.",
    )
    args = parser.parse_args()

    root = pathlib.Path(args.output_root)
    libri_dir = root / "libritts"
    cv_dir = root / "common_voice_en"

    if args.dataset_mode == "dummy":
        libri_dataset_name = "hf-internal-testing/librispeech_asr_dummy"
        libri_config = "clean"
        libri_split = "validation"
        cv_dataset_name = "patrickvonplaten/librispeech_asr_dummy"
        cv_config = "clean"
        cv_split = "validation"
        libri_transcript_key = "text"
        cv_transcript_key = "text"
    else:
        libri_dataset_name = "openslr/librispeech_asr"
        libri_config = "clean"
        libri_split = "train.100"
        cv_dataset_name = "mozilla-foundation/common_voice_17_0"
        cv_config = "en"
        cv_split = "train"
        libri_transcript_key = "text"
        cv_transcript_key = "sentence"

    print(f"Loading Libri split from {libri_dataset_name}...")
    libri = load_dataset(
        libri_dataset_name,
        libri_config,
        split=libri_split,
    )
    libri_count = _copy_rows(
        dataset=libri,
        out_dir=libri_dir,
        transcript_key=libri_transcript_key,
        limit=args.libri_limit,
        prefix="libri",
    )
    print(f"Saved {libri_count} files to {libri_dir}")

    print(f"Loading Common Voice split from {cv_dataset_name}...")
    cv = load_dataset(
        cv_dataset_name,
        cv_config,
        split=cv_split,
    )
    cv_count = _copy_rows(
        dataset=cv,
        out_dir=cv_dir,
        transcript_key=cv_transcript_key,
        limit=args.cv_limit,
        prefix="cv",
    )
    print(f"Saved {cv_count} files to {cv_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
