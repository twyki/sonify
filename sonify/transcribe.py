import hashlib
import subprocess
import tempfile
from pathlib import Path
from tempfile import mkdtemp
import math
import shutil
from typing import Dict, Any, List, Callable
import streamlit as st
import json


# -----------------------------------------------------------------------------
# Model loading and resource caching
# -----------------------------------------------------------------------------
@st.cache_resource
def get_whisper_model(model_name: str):
    import whisper
    return whisper.load_model(model_name)


# -----------------------------------------------------------------------------
# WAV conversion and data caching
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, persist="disk")
def convert_to_wav(input_path: str) -> str:
    input_path = Path(input_path)
    if input_path.suffix.lower() == ".wav":
        return str(input_path)
    temp_dir = Path(mkdtemp(prefix="wav_conv_"))
    wav_path = temp_dir / (input_path.stem + ".wav")
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y", "-i", str(input_path),
        "-ac", "1", "-ar", "16000", str(wav_path)
    ], check=True)
    return str(wav_path)


# -----------------------------------------------------------------------------
# Core transcription helpers with caching
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, hash_funcs={
    dict: lambda d: json.dumps(d, sort_keys=True, default=str),
    list: lambda lst: json.dumps(lst, sort_keys=True, default=str),
}, persist="disk")
def transcribe_simple(wav_path: str, model_name: str, language: str) -> Dict[str, Any]:
    model = get_whisper_model(model_name)
    if language == "auto":
        return model.transcribe(wav_path, verbose=False, fp16=False)
    return model.transcribe(wav_path, language=language, verbose=False, fp16=False)


# -----------------------------------------------------------------------------
# Single-call transcription with callback (no generator)
# -----------------------------------------------------------------------------
def transcribe_stream(
    wav_path: str,
    model_name: str,
    language: str,
    chunk_size: int,
    _progress_callback: Callable[[int, int], None] = None,
) -> List[Dict[str, Any]]:
    """
    Transcribe audio in fixed-size chunks, calling `_progress_callback(completed, total)`
    after each chunk, and return the combined segments list. Chunk files are stored
    deterministically under a temp cache directory to avoid recreating them each run.
    """
    # Base cache directory for chunk files
    base = Path(tempfile.gettempdir()) / "sonify_chunks"
    key = hashlib.sha256(f"{wav_path}|{chunk_size}".encode()).hexdigest()
    cache_dir = base / key
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Only split into chunks if we haven't already
    chunk_files = sorted(cache_dir.glob("chunk_*.wav"))
    if not chunk_files:
        # Determine total number of chunks
        dur = float(
            subprocess.check_output([
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", wav_path
            ]).decode().strip()
        )
        # Run ffmpeg segmenter
        subprocess.run([
            "ffmpeg", "-loglevel", "error", "-y", "-i", wav_path,
            "-f", "segment", "-segment_time", str(chunk_size),
            "-c", "copy", str(cache_dir / "chunk_%03d.wav")
        ], check=True)
        chunk_files = sorted(cache_dir.glob("chunk_*.wav"))

    total_chunks = len(chunk_files)
    segments: List[Dict[str, Any]] = []

    for idx, chunk_file in enumerate(chunk_files, start=1):
        # Transcribe each chunk via your cached helper
        res = transcribe_simple(str(chunk_file), model_name, language)

        # Adjust timestamps and collect segments
        for seg in res.get("segments", []):
            seg["start"] += (idx - 1) * chunk_size
            seg["end"] += (idx - 1) * chunk_size
            segments.append(seg)

        # Report progress if requested
        if _progress_callback:
            _progress_callback(idx, total_chunks)

    return segments


# -----------------------------------------------------------------------------
# Full-file transcription via cache (calls transcribe_stream)
# -----------------------------------------------------------------------------
def transcribe_with_cache(
        src: str,
        model_name: str = "medium",
        language: str = "de",
        chunk_size: int | None = None,
        _progress_callback: Callable[[int, int], None] = None,
) -> Dict[str, Any]:
    wav = convert_to_wav(src)
    if chunk_size is None:
        result = transcribe_simple(wav, model_name, language)
        if _progress_callback:
            _progress_callback(1, 1)
        return {"text": result.get("text", ""), "segments": result.get("segments", [])}
    segments = transcribe_stream(wav, model_name, language, chunk_size, _progress_callback)
    text = " ".join(seg.get("text", "") for seg in segments)
    return {"text": text, "segments": segments}
