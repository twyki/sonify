import json
import hashlib
import logging
from typing import List, Dict
import torch
from pyannote.audio.models.blocks.pooling import StatsPool
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.audio import Pipeline
from .transcribe import cached_wav  # ← import at the top of the file
from pathlib import Path
import math


# … your existing imports …

class StreamlitHook:
    """
    A Pyannote-compatible hook that calls `callback(step_name, completed, total)`
    every time the pipeline emits an update.
    """

    def __init__(self, callback):
        # callback will be called as callback(step_name, completed, total)
        self.callback = callback

    def __enter__(self):
        # nothing to initialize
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # nothing to clean up
        return False

    def __call__(self, step_name, step_artifact, file=None, total=None, completed=None):
        # Pyannote guarantees total/completed on each call
        # fire your callback immediately
        self.callback(step_name, completed, total)


# Patch StatsPool for Pyannote
def patched_forward(self, sequences, weights=None):
    mean = sequences.mean(dim=-1)
    if sequences.size(-1) > 1:
        std = sequences.std(dim=-1, correction=1)
    else:
        std = torch.zeros_like(mean)
    return torch.cat([mean, std], dim=-1)


StatsPool.forward = patched_forward

# Cache directory
DIAR_CACHE_DIR = Path.home() / ".cache" / "sonify" / "diar"
DIAR_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _diar_cache_key(file_id: str, segments: list) -> str:
    m = hashlib.md5()
    m.update(file_id.encode("utf-8"))
    seg_json = json.dumps(
        sorted(segments, key=lambda s: (s["start"], s["end"], s["text"])),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    m.update(seg_json.encode("utf-8"))
    return m.hexdigest()


def load_cached_diar(file_id: str, segments: list) -> List[Dict]:
    key = _diar_cache_key(file_id, segments)
    fpth = DIAR_CACHE_DIR / f"{key}.json"
    if fpth.exists():
        try:
            return json.loads(fpth.read_text(encoding="utf-8"))
        except:
            fpth.unlink()
    return []


def save_cached_diar(file_id: str, segments: list, turns: list):
    key = _diar_cache_key(file_id, segments)
    fpth = DIAR_CACHE_DIR / f"{key}.json"
    fpth.write_text(json.dumps(turns, ensure_ascii=False, indent=2), encoding="utf-8")


def diarize_audio(
        src: str,
        segments: list,
        hf_token: str,
        progress_callback: callable = None
) -> List[Dict]:
    """
    Run (or load) speaker diarization, align with segments, and cache results.

    progress_callback: optional fn(str) to receive tqdm-style text.
    """

    wav_path = cached_wav(src)
    # 1) Prepare WAV & file_id
    from sonify.streamlit_app import generate_file_id
    audio_bytes = Path(wav_path).read_bytes()
    file_id = generate_file_id(audio_bytes, Path(wav_path).name)

    # 2) Try cached diarization
    cached = load_cached_diar(file_id, segments)
    if cached:
        logging.info("Loaded diarization from cache.")
        return cached

    # 3) Load pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )

    # If a callback is provided, wrap it in our hook
    if progress_callback:
        hook = StreamlitHook(progress_callback)
        with hook as h:
            print("diarizing...")
            diar = pipeline(wav_path, hook=h)
    else:
        # no callback — just run normally
        with ProgressHook() as hook:
            diar = pipeline(wav_path, hook=hook)
    # 5) Extract raw turns
    raw_turns = [
        (speaker, turn.start, turn.end)
        for turn, _, speaker in diar.itertracks(yield_label=True)
    ]

    # 6) Align word-segments
    aligned = []
    used = set()
    for speaker, start, end in raw_turns:
        texts = []
        for idx, seg in enumerate(segments):
            if idx in used:
                continue
            if seg["start"] < end and seg["end"] > start:
                texts.append(seg["text"].strip())
                used.add(idx)
        if texts:
            aligned.append({
                "speaker": speaker,
                "start": start,
                "end": end,
                "text": " ".join(texts),
            })

    # 7) Cache & return
    save_cached_diar(file_id, segments, aligned)
    logging.info(f"Saved {len(aligned)} diarization turns to cache.")
    return aligned
