#!/usr/bin/env python3
"""Run a local TTS inference smoke test.

This script is intended to work with the assets already present in this repo:
- Local Llama base model in models/base_lm/Llama-3.2-1B-Instruct
- Local xcodec2 checkpoint in models/codec/xcodec2/ckpt/epoch=4-step=1400000.ckpt
- A prompt audio file from data/downloaded/libritts

Example:
    python tools/serving/sample_inference.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torchaudio
import transformers

from tts.core import constants, prompting
from tts.core.codec import decoding, encoding
from tts.data import data_utils
from tts.inference import inferencing
from tts.core import modeling


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local TTS inference smoke test.")
    parser.add_argument(
        "--model_checkpoint_path",
        default="",
        help="Optional path to a trained checkpoint file (e.g. final_model.pt).",
    )
    parser.add_argument(
        "--base_model_dir",
        default="models/base_lm/Llama-3.2-1B-Instruct",
        help="Path to the local base language model directory.",
    )
    parser.add_argument(
        "--codec_checkpoint",
        default="models/codec/xcodec2/ckpt/epoch=4-step=1400000.ckpt",
        help="Path to the local codec checkpoint.",
    )
    parser.add_argument(
        "--prompt_wav",
        default="data/downloaded/libritts/libri_00014.flac",
        help="Path to the prompt audio file.",
    )
    parser.add_argument(
        "--prompt_transcription",
        default="BY HARRY QUILTER M A",
        help="Transcript for the prompt audio.",
    )
    parser.add_argument(
        "--text",
        default="Hello from a local sample inference.",
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--output_path",
        default="audios/sample_inference.wav",
        help="Where to save the generated audio.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for generation.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=24,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--min_tokens",
        type=int,
        default=1,
        help="Minimum number of tokens to generate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_checkpoint_path = Path(args.model_checkpoint_path) if args.model_checkpoint_path else None
    base_model_dir = Path(args.base_model_dir)
    codec_checkpoint = Path(args.codec_checkpoint)
    prompt_wav_path = Path(args.prompt_wav)
    output_path = Path(args.output_path)

    if not base_model_dir.exists():
        raise FileNotFoundError(f"Base model directory not found: {base_model_dir}")
    if not codec_checkpoint.exists():
        raise FileNotFoundError(f"Codec checkpoint not found: {codec_checkpoint}")
    if not prompt_wav_path.exists():
        raise FileNotFoundError(f"Prompt audio not found: {prompt_wav_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if model_checkpoint_path is not None:
        print(f"Loading trained checkpoint: {model_checkpoint_path}")
        tokenizer, _, model, _ = modeling.load_tokenizer_config_and_model(
            str(model_checkpoint_path)
        )
        model.eval()
    else:
        print("Loading tokenizer...")
        tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_dir)

        print("Loading base model...")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            base_model_dir,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
        model.eval()

    print("Loading codec encoder/decoder...")
    audio_encoder = encoding.CachingAudioEncoder(
        model_path=str(codec_checkpoint), device=device
    )
    audio_decoder = decoding.AudioDecoder(
        model_path=str(codec_checkpoint),
        config=decoding.DecoderConfig(
            model_type="xcodec2",
            sample_rate=16000,
            token_rate=50,
            hop_length=320,
            upsample_factors=[],
            kernel_sizes=[],
        ),
        device=device,
    )

    prompt_compiler = prompting.InferencePromptCompiler()
    tts_model = inferencing.LocalTtsModel(
        model=model,
        device=device,
        tokenizer=tokenizer,
        audio_encoder=audio_encoder,
        audio_decoder=audio_decoder,
        prompt_compiler=prompt_compiler,
        use_vllm=False,
    )

    print(f"Loading prompt audio: {prompt_wav_path}")
    prompt_wav, sample_rate = data_utils.load_wav(
        str(prompt_wav_path), target_sample_rate=constants.CODEC_SAMPLE_RATE
    )
    print(f"Prompt sample rate: {sample_rate}")

    inference_settings = inferencing.InferenceSettings(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        top_p=1.0,
        top_k=50,
        repetition_penalty=1.0,
        frequency_penalty=0.0,
        seed=42,
    )

    print("Running inference...")
    result = tts_model.synthesize_speech(
        inference_settings=inference_settings,
        text_to_synthesize=args.text,
        prompt_id="sample_inference_prompt",
        prompt_wav=prompt_wav,
        audio_prompt_transcription=args.prompt_transcription,
        voice_description="",
        enable_instruction=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), result.wav, audio_decoder.sample_rate)

    print(f"Saved output audio to: {output_path}")
    print(f"Output waveform shape: {tuple(result.wav.shape)}")
    print(f"Encoding time: {result.encoding_time:.2f}s")
    print(f"Decoding time: {result.decoding_time:.2f}s")
    print(f"Inference time: {result.inference_time:.2f}s")


if __name__ == "__main__":
    main()
