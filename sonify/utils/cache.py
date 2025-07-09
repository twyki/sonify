import json

import hashlib
from pathlib import Path
from typing import List, Dict

BASE = Path.home() / ".cache" / "sonify"
SEG = BASE / "segments"
SEG.mkdir(parents=True, exist_ok=True)
DIAR = BASE / "diar"
DIAR.mkdir(parents=True, exist_ok=True)


def _cache_key(file_id: str, model: str, language: str) -> str:
    raw = f"{file_id}-{model}-{language}"
    return hashlib.md5(raw.encode()).hexdigest()


def load_cached_segments(file_id: str, model: str, language: str) -> List[Dict]:
    key = _cache_key(file_id, model, language)
    fpth = SEG / f"{key}.json"
    if fpth.exists():
        try:
            return json.loads(fpth.read_text())
        except:
            fpth.unlink()
    return []


def save_cached_segments(file_id: str, model: str, language: str, segs: List[Dict]):
    key = _cache_key(file_id, model, language)
    fpth = SEG / f"{key}.json"
    fpth.write_text(json.dumps(segs))


def load_cached_turns(file_id: str, model: str, language: str) -> List[Dict]:
    key = _cache_key(file_id, model, language)
    fpth = DIAR / f"{key}.json"
    if fpth.exists():
        try:
            return json.loads(fpth.read_text())
        except:
            fpth.unlink()
    return []


def save_cached_turns(file_id: str, model: str, language: str, turns: List[Dict]):
    key = _cache_key(file_id, model, language)
    fpth = DIAR / f"{key}.json"
    fpth.write_text(json.dumps(turns))


def generate_file_id(data: bytes, name: str) -> str:
    digest = hashlib.sha256(data[:64]).hexdigest()[:8]
    return f"{name}-{len(data)}-{digest}"