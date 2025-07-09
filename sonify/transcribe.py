import hashlib
import json
import subprocess
import logging
from pathlib import Path
from tempfile import mkdtemp
import math
import shutil
from typing import  Dict, Any, Generator, Callable

# Cache directories
CACHE_ROOT = Path.home() / ".cache" / "sonify"
TXT_CACHE = CACHE_ROOT / "json"
WAV_CACHE = CACHE_ROOT / "wav"
CHUNK_CACHE = WAV_CACHE / "chunks"

for folder in (TXT_CACHE, WAV_CACHE, CHUNK_CACHE):
    folder.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Hash helpers
# -----------------------------------------------------------------------------

def _sha256_file(path: str, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _cache_key(src: str, model: str, lang: str) -> str:
    h = hashlib.sha256()
    h.update(Path(src).read_bytes())
    h.update(model.encode())
    h.update(lang.encode())
    return h.hexdigest()[:16]


# -----------------------------------------------------------------------------
# WAV caching
# -----------------------------------------------------------------------------

def cached_wav(input_path: str) -> str:
    wav_hash = _sha256_file(input_path)
    wav_path = WAV_CACHE / f"{wav_hash}.wav"
    if wav_path.exists():
        logger.debug(f"Found cached WAV: {wav_path}")
        return str(wav_path)
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y", "-i", input_path,
        "-ac", "1", "-ar", "16000", str(wav_path)
    ], check=True)
    logger.debug(f"Converted and cached WAV: {wav_path}")
    return str(wav_path)


# -----------------------------------------------------------------------------
# Core transcription helpers
# -----------------------------------------------------------------------------

def _transcribe_simple(wav_path: str, model_name: str, language: str) -> Dict[str, Any]:
    import whisper
    model = whisper.load_model(model_name)
    logger.info(f"Transcribing {wav_path} with {model_name} ({language}) …")
    return model.transcribe(wav_path, language=language, verbose=False ,fp16=False)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def transcribe_with_cache(
        src: str,
        model_name: str = "medium",
        language: str = "en",
        force: bool = False,
        chunk_size: int | None = None,
        progress_callback: Callable[[float], None] = None,
) -> Dict[str, Any]:
    """
    Full-file transcription, cached.

    `chunk_size` is retained for backward compatibility – when provided, the
    function will split the WAV and stitch results. Internally it re-uses this
    very same function, so caching still applies per chunk + model + language.
    Calls progress_callback(progress) with float in [0,1] if provided.
    """
    key = _cache_key(src, model_name, language) if chunk_size is None else None
    cache_file = TXT_CACHE / f"{key}.json" if key else None

    # Load from cache if available, including segments
    if cache_file and cache_file.exists() and not force:
        result = json.loads(cache_file.read_text("utf-8"))
        if "segments" in result and progress_callback:
            progress_callback(1.0)
        return result

    if cache_file and cache_file.exists():
        cache_file.unlink()

    wav_path = cached_wav(src)

    # Recursive chunking path
    if chunk_size:
        dur = float(
            subprocess.check_output([
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", wav_path
            ]).decode().strip()
        )
        total_chunks = math.ceil(dur / chunk_size)
        temp_dir = Path(mkdtemp(prefix="chunks_"))
        subprocess.run([
            "ffmpeg", "-loglevel", "error", "-y", "-i", wav_path,
            "-f", "segment", "-segment_time", str(chunk_size),
            "-c", "copy", str(temp_dir / "chunk_%03d.wav")
        ], check=True)

        segments, texts = [], []
        for idx, f in enumerate(sorted(temp_dir.glob("chunk_*.wav"))):
            if progress_callback:
                progress_callback(idx / total_chunks)
            res = transcribe_with_cache(
                str(f), model_name, language, force, None, progress_callback
            )
            off = idx * chunk_size
            for s in res.get("segments", []):
                s["start"] += off
                s["end"] += off
                segments.append(s)
            texts.append(res.get("text", ""))
        shutil.rmtree(temp_dir, ignore_errors=True)
        result = {"text": " ".join(texts), "segments": segments}
    else:
        result = _transcribe_simple(wav_path, model_name, language)

    # Final caching
    if cache_file:
        cache_file.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), "utf-8"
        )
        if progress_callback:
            progress_callback(1.0)
    return result


def transcribe_stream(
        wav_path: str,
        model_name: str = "small",
        language: str = "en",
        chunk_size: int = 30,
) -> Generator[Dict[str, Any], None, None]:
    import subprocess, math, shutil, json, hashlib
    from pathlib import Path
    from tempfile import mkdtemp

    # -------------------------------------------------------------------------
    # Helpers: per‐chunk cache key + path
    # -------------------------------------------------------------------------
    def _chunk_cache_key(chunk_path: str, model: str, lang: str) -> str:
        h = hashlib.sha256()
        h.update(Path(chunk_path).read_bytes())
        h.update(model.encode())
        h.update(lang.encode())
        return h.hexdigest()[:16]

    CHUNK_JSON_CACHE = Path.home() / ".cache" / "sonify" / "json" / "chunks"
    CHUNK_JSON_CACHE.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. figure total duration & number of chunks
    # -------------------------------------------------------------------------
    dur = float(
        subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", wav_path
        ]).decode().strip()
    )
    total_chunks = math.ceil(dur / chunk_size)

    # -------------------------------------------------------------------------
    # 2. split into fixed-length chunks
    # -------------------------------------------------------------------------
    temp_dir = Path(mkdtemp(prefix="stream_chunks_"))
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y", "-i", wav_path,
        "-f", "segment", "-segment_time", str(chunk_size),
        "-c", "copy", str(temp_dir / "chunk_%03d.wav")
    ], check=True)

    # -------------------------------------------------------------------------
    # 3. transcribe each chunk, loading/saving per-chunk cache
    # -------------------------------------------------------------------------
    all_segs = []
    chunks = sorted(temp_dir.glob("chunk_*.wav"))
    for idx, chunk in enumerate(chunks):
        key = _chunk_cache_key(str(chunk), model_name, language)
        cache_f = CHUNK_JSON_CACHE / f"{key}.json"

        if cache_f.exists():
            # load cached result
            res = json.loads(cache_f.read_text("utf-8"))
        else:
            # run Whisper on this chunk
            res = _transcribe_simple(str(chunk), model_name, language)
            # save to cache
            cache_f.write_text(json.dumps(res, ensure_ascii=False, indent=2), "utf-8")

        # shift timestamps into global timeline
        offset = idx * chunk_size
        for s in res.get("segments", []):
            s["start"] += offset
            s["end"] += offset
            all_segs.append(s)

        # yield this chunk’s progress and segments
        yield {
            "chunk_index": idx + 1,
            "total_chunks": total_chunks,
            "progress": (idx + 1) / total_chunks,
            "segments": res.get("segments", []),
        }

    # -------------------------------------------------------------------------
    # 4. cleanup & final 100% bump
    # -------------------------------------------------------------------------
    shutil.rmtree(temp_dir, ignore_errors=True)
    yield {
        "chunk_index": total_chunks,
        "total_chunks": total_chunks,
        "progress": 1.0,
        "segments": [],
    }
