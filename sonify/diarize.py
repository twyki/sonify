from typing import List, Dict, Callable
import streamlit as st
import torch
import json
from pyannote.audio.models.blocks.pooling import StatsPool
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.audio import Pipeline
from .transcribe import convert_to_wav  # ensure WAV conversion is available


# -----------------------------------------------------------------------------
# Patch StatsPool for Pyannote
# -----------------------------------------------------------------------------
def patched_forward(self, sequences, weights=None):
    mean = sequences.mean(dim=-1)
    if sequences.size(-1) > 1:
        std = sequences.std(dim=-1, correction=1)
    else:
        std = torch.zeros_like(mean)
    return torch.cat([mean, std], dim=-1)


StatsPool.forward = patched_forward


# -----------------------------------------------------------------------------
# Streamlit-compatible progress hook
# -----------------------------------------------------------------------------
class StreamlitHook:
    """
    A Pyannote-compatible hook that calls `callback(step_name, completed, total)`
    every time the pipeline emits an update.
    """

    def __init__(self, callback: Callable[[str, int, int], None]):
        self.callback = callback

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def __call__(self, step_name, step_artifact=None, file=None, total=None, completed=None):
        if total is not None and completed is not None:
            self.callback(step_name, completed, total)


# -----------------------------------------------------------------------------
# Resource-cached pipeline loader
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_diar_pipeline(model_id: str, hf_token: str) -> Pipeline:
    """
    Load and cache the Pyannote speaker-diarization pipeline as a resource.
    """
    return Pipeline.from_pretrained(model_id, use_auth_token=hf_token)


# -----------------------------------------------------------------------------
# Data-cached diarization
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, hash_funcs={dict: lambda d: json.dumps(d, sort_keys=True, default=str), list: lambda lst: json.dumps(lst, sort_keys=True, default=str)})
def run_diarization(
        src_wav: str,
        segments: List[Dict],
        model_id: str,
        hf_token: str
) -> List[Dict]:
    """
    Perform speaker diarization on WAV file and align with word segments.
    Cached by Streamlit to avoid re-computation.
    """
    #print(f"{src_wav=}, {segments=}, {model_id=}, {hf_token=}")
    pipeline = get_diar_pipeline(model_id, hf_token)
    diar = pipeline(src_wav)

    raw_turns = [
        (speaker, turn.start, turn.end)
        for turn, _, speaker in diar.itertracks(yield_label=True)
    ]

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
    return aligned


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def diarize_audio(
        src: str,
        segments: List[Dict],
        hf_token: str,
        model_id: str = "pyannote/speaker-diarization-3.1",
        _progress_callback: Callable[[str, int, int], None] = None
) -> List[Dict]:
    """
    High-level speaker diarization wrapper using Streamlit caching.
    If no Streamlit callback provided, uses Pyannote's CLI ProgressHook for stdout updates.
    """
    # Convert source to WAV if necessary
    src_wav = convert_to_wav(src)

    # CLI mode: no callback supplied, use Pyannote ProgressHook
    if _progress_callback is None:
        with ProgressHook() as hook:
            pipeline = get_diar_pipeline(model_id, hf_token)
            pipeline(src_wav, hook=hook)
        # After CLI progress, return the cached full result4
        #print(f"{src_wav=}, {segments=}, {model_id=}, {hf_token=}")
        return run_diarization(src_wav, segments, model_id, hf_token)

    # Streamlit mode: use StreamlitHook for progress
    hook = StreamlitHook(_progress_callback)
    with hook:
        pipeline = get_diar_pipeline(model_id, hf_token)
        pipeline(src_wav, hook=hook)

    # Return cached full result after streaming

    #print(f"{src_wav=}, {segments=}, {model_id=}, {hf_token=}")
    return run_diarization(src_wav, segments, model_id, hf_token)
