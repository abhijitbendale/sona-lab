#!/usr/bin/env python3
"""Build per-source JSONL manifests from downloaded data folders."""

from __future__ import annotations

import argparse
import json
import pathlib
import wave


def _duration_and_sr(path: pathlib.Path) -> tuple[float, int]:
    # Wave files are preferred for training; if not wave-readable, return fallback.
    try:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            sr = wf.getframerate()
            if sr <= 0:
                return 0.0, 0
            return frames / float(sr), sr
    except Exception:
        return 0.0, 0


def _iter_audio_files(source_dir: pathlib.Path):
    for pattern in ("*.wav", "*.flac", "*.mp3"):
        for p in source_dir.glob(pattern):
            yield p


def _build_source_manifest(source_dir: pathlib.Path, out_jsonl: pathlib.Path) -> int:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_jsonl.open("w", encoding="utf-8") as fout:
        for audio_path in sorted(_iter_audio_files(source_dir)):
            txt_path = audio_path.with_suffix(".txt")
            if not txt_path.exists():
                continue
            transcript = txt_path.read_text(encoding="utf-8").strip()
            if not transcript:
                continue
            duration, sr = _duration_and_sr(audio_path)
            entry = {
                "transcript": transcript,
                "language": "en",
                "wav_path": str(audio_path.resolve()),
                "duration": round(duration, 3),
                "sample_rate": sr,
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_root", default="data/downloaded")
    parser.add_argument("--manifest_root", default="data/processed/manifests")
    args = parser.parse_args()

    download_root = pathlib.Path(args.download_root)
    manifest_root = pathlib.Path(args.manifest_root)

    libri_count = _build_source_manifest(
        source_dir=download_root / "libritts",
        out_jsonl=manifest_root / "libritts_small.jsonl",
    )
    print(f"Built libritts manifest rows: {libri_count}")

    cv_count = _build_source_manifest(
        source_dir=download_root / "common_voice_en",
        out_jsonl=manifest_root / "common_voice_en_small.jsonl",
    )
    print(f"Built common voice manifest rows: {cv_count}")


if __name__ == "__main__":
    main()
