import hashlib
import json
import subprocess
from pathlib import Path

# Cache directories
CACHE_ROOT = Path.home() / ".cache" / "sonify"
TXT_CACHE = CACHE_ROOT / "json"
WAV_CACHE = CACHE_ROOT / "wav"

# Ensure caches exist
for folder in (TXT_CACHE, WAV_CACHE):
    folder.mkdir(parents=True, exist_ok=True)


def _sha256(path: str, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def cached_wav(input_path: str) -> str:
    """
    Return path to 16-kHz mono WAV, caching conversion.
    """
    wav_path = WAV_CACHE / f"{_sha256(input_path)}.wav"
    if wav_path.exists():
        return str(wav_path)

    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y", "-i", input_path,
        "-ac", "1", "-ar", "16000", str(wav_path)
    ], check=True)
    return str(wav_path)


def transcribe_with_cache(
        src: str,
        model_name: str = "medium",
        language: str = "en"
) -> dict:
    """
    Transcribe audio using Whisper, caching JSON result.
    """
    cache_file = TXT_CACHE / f"{_sha256(src)}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    import whisper
    model = whisper.load_model(model_name)
    result = model.transcribe(src, language=language, verbose=False)

    cache_file.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    return result
