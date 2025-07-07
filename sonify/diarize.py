from pathlib import Path
from datetime import timedelta
from .transcribe import cached_wav


def diarize_audio(
        src: str,
        segments: list,
        hf_token: str
) -> list:
    """
    Align Whisper segments with pyannote speaker turns.
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        raise RuntimeError(
            "pyannote.audio not installed; install with `pip install git+https://github.com/pyannote/pyannote-audio`"
        )

    wav = cached_wav(src)
    pipe = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=hf_token
    )
    diar = pipe(wav)

    used = set()
    aligned = []
    for turn, _, speaker in diar.itertracks(yield_label=True):
        texts = []
        for idx, seg in enumerate(segments):
            if idx in used:
                continue
            if seg['start'] < turn.end and seg['end'] > turn.start:
                texts.append(seg['text'].strip())
                used.add(idx)
        if texts:
            aligned.append({
                'speaker': speaker,
                'start': turn.start,
                'end': turn.end,
                'text': ' '.join(texts)
            })
    return aligned
