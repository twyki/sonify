import argparse
import logging
from pathlib import Path
from .transcribe import transcribe_with_cache
from .diarize import diarize_audio
from datetime import timedelta


def format_segments(segments):
    """
    Return a string representation of transcript segments
    in the form: [HH:MM:SS – HH:MM:SS] text
    """
    lines = []
    for s in segments:
        start = timedelta(seconds=int(s.get('start', 0)))
        end = timedelta(seconds=int(s.get('end', 0)))
        text = s.get('text', '').strip()
        lines.append(f"[{start} – {end}] {text}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Whisper transcription with caching, optional diarization, and controlled logging"
    )
    parser.add_argument("audio", help="Audio file path")
    parser.add_argument("-m", "--model", default="medium", help="Whisper model size")
    parser.add_argument("-l", "--lang", default="de", help="Language code")
    parser.add_argument("-hft", "--hf_token", help="HuggingFace token for diarization")
    parser.add_argument("-O", "--out_dir", default="output", help="Output directory for transcript and diarization files")
    parser.add_argument("-f", "--force", action="store_true", help="Force refresh of outputs")
    parser.add_argument("-c", "--chunk_size", type=int, help="Split audio into chunks of given length (seconds) for per-chunk caching")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging for sonify modules and print segments")
    args = parser.parse_args()

    # Configure root logger with filter to allow only sonify logs
    root = logging.getLogger()
    root.handlers.clear()
    level = logging.DEBUG if args.verbose else logging.INFO
    root.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    allowed_prefixes = ('sonify.transcribe', 'sonify.diarize', __name__)

    class ModuleFilter(logging.Filter):
        def filter(self, record):
            return any(record.name.startswith(prefix) for prefix in allowed_prefixes)

    handler.addFilter(ModuleFilter())
    root.addHandler(handler)
    logger = logging.getLogger(__name__)

    stem = Path(args.audio).stem
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = out_dir / f"{stem}.transcript.txt"
    segments_path = out_dir / f"{stem}.segments.txt"
    diar_path = out_dir / f"{stem}.diarized.txt"

    # Force: remove existing outputs
    if args.force:
        for path in (transcript_path, segments_path):
            if path.exists():
                path.unlink()
                logger.debug(f"Removed existing file: {path}")
        if args.hf_token and diar_path.exists():
            diar_path.unlink()
            logger.debug(f"Removed existing diarization: {diar_path}")

    # Skip if all required outputs exist and not forcing
    skip_transcripts = transcript_path.exists() and segments_path.exists()
    skip_diar = not args.hf_token or diar_path.exists()
    if not args.force and skip_transcripts and skip_diar:
        msg = f"Skipping: outputs exist. Transcript at {transcript_path}, Segments at {segments_path}"
        if args.hf_token:
            msg += f", Diarization at {diar_path}"
        logger.info(msg)
        # Print existing segments only if verbose
        if args.verbose:
            try:
                print(segments_path.read_text())
            except Exception:
                pass
        return

    # Transcription
    result = transcribe_with_cache(
        args.audio, args.model, args.lang,
        force=args.force, chunk_size=args.chunk_size
    )

    # Write full transcript
    transcript_path.write_text(result.get("text", "").strip(), encoding="utf-8")
    logger.info(f"Transcript saved to {transcript_path}")

    # Always write transcript segments to file
    segments = result.get("segments", []) or []
    segments_str = format_segments(segments)
    segments_path.write_text(segments_str, encoding="utf-8")
    logger.info(f"Transcript segments saved to {segments_path}")

    # Print segments to stdout only if verbose
    if args.verbose:
        print(segments_str)

    # Diarization
    if args.hf_token:
        if not args.force and diar_path.exists():
            logger.info(f"Skipping diarization: file exists at {diar_path}")
            return
        logger.info(f"Starting diarization for {args.audio}")
        diar = diarize_audio(args.audio, segments, args.hf_token)
        logger.debug(f"Diarization returned {len(diar)} segments")
        with open(diar_path, 'w', encoding='utf-8') as f:
            for item in diar:
                s = timedelta(seconds=int(item['start']))
                e = timedelta(seconds=int(item['end']))
                f.write(f"{item['speaker']} [{s}-{e}]: {item['text']}\n")
        logger.info(f"Diarized transcript saved to {diar_path}")
