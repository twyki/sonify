import argparse
from .transcribe import transcribe_with_cache, cached_wav
from .diarize import diarize_audio
from datetime import timedelta
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Whisper transcription with caching and optional diarization"
    )
    parser.add_argument("audio", help="Audio file path")
    parser.add_argument(
        "-m", "--model", default="medium",
        help="Whisper model size"
    )
    parser.add_argument(
        "-l", "--lang", default="en",
        help="Language code"
    )
    parser.add_argument(
        "--hf_token",
        help="HuggingFace token for diarization"
    )
    parser.add_argument(
        "--out",
        help="Output transcript file"
    )
    args = parser.parse_args()

    if not args.out:
        args.out = f"{Path(args.audio).stem}-transcript.txt"

    result = transcribe_with_cache(
        args.audio, args.model, args.lang
    )
    Path(args.out).write_text(
        result.get('text', '').strip(), encoding="utf-8"
    )
    print(f"Transcript saved to {args.out}")

    if args.hf_token:
        diar = diarize_audio(
            args.audio, result['segments'], args.hf_token
        )
        diar_file = Path(args.out).with_suffix('.diarized.txt')
        with open(diar_file, 'w', encoding='utf-8') as f:
            for item in diar:
                s = timedelta(seconds=int(item['start']))
                e = timedelta(seconds=int(item['end']))
                f.write(f"{item['speaker']} [{s}-{e}]: {item['text']}\n")
        print(f"Diarized transcript saved to {diar_file}")
