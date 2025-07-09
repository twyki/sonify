import json
import tempfile
import hashlib
from pathlib import Path
from datetime import timedelta
import time
from typing import List, Dict

import streamlit as st
from cryptography.fernet import Fernet

from sonify.transcribe import cached_wav, transcribe_stream
from sonify.diarize import diarize_audio

CONFIG_PATH = Path.home() / ".sonify_config.json"
KEY_PATH = Path.home() / ".sonify_key"
AUDIO_TYPES = ["mp3", "wav", "m4a", "flac", "aac", "opus", "ogg"]

# Cache for transcript segments
CACHE_DIR_SEG = Path.home() / ".cache" / "sonify" / "segments"
CACHE_DIR_SEG.mkdir(parents=True, exist_ok=True)

CACHE_DIR_DIAR = Path.home() / ".cache" / "sonify" / "diar"
CACHE_DIR_DIAR.mkdir(parents=True, exist_ok=True)


def format_hms(seconds: float) -> str:
    hrs, rem = divmod(int(seconds), 3600)
    mins, secs = divmod(rem, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def load_or_generate_key() -> Fernet:
    if not KEY_PATH.exists():
        key = Fernet.generate_key()
        KEY_PATH.write_bytes(key)
    else:
        key = KEY_PATH.read_bytes()
    return Fernet(key)


def init_session():
    """Initialize session state and decrypt stored HF token if present."""
    # 1. Load or generate your Fernet key
    from cryptography.fernet import Fernet
    if not KEY_PATH.exists():
        key = Fernet.generate_key()
        KEY_PATH.write_bytes(key)
    else:
        key = KEY_PATH.read_bytes()
    fernet = Fernet(key)
    st.session_state.setdefault("_fernet", fernet)

    # 2. Load saved config
    saved = {}
    if CONFIG_PATH.exists():
        try:
            saved = json.loads(CONFIG_PATH.read_text())
        except json.JSONDecodeError:
            saved = {}

    # 3. Decrypt hf_token (empty string on failure)
    enc_token = saved.get("hf_token", "")
    try:
        hf_token = fernet.decrypt(enc_token.encode()).decode()
    except Exception:
        hf_token = ""

    # 4. Populate defaults
    st.session_state.setdefault("cfg", {
        "model": saved.get("model", "small"),
        "language": saved.get("language", "en"),
        "hf_token": hf_token,
    })
    st.session_state.setdefault("phase", "empty")
    st.session_state.setdefault("audio_path", None)
    st.session_state.setdefault("file_id", None)
    st.session_state.setdefault("segments", [])
    st.session_state.setdefault("turns", [])
    st.session_state.setdefault("prev_model", None)
    st.session_state.setdefault("confirm_clear", False)
    st.session_state.setdefault("file_uploader_key", 0)
    st.session_state.setdefault("speaker_names", {})


def persist_config() -> None:
    """Encrypt and save current config (model, language, hf_token)."""
    cfg = st.session_state.cfg
    fernet = st.session_state._fernet

    # 1. Encrypt HF token
    try:
        enc_token = fernet.encrypt(cfg["hf_token"].encode()).decode()
    except Exception:
        st.warning("Failed to encrypt HF token.")
        enc_token = ""

    # 2. Write out the rest of the config
    to_save = {
        "model": cfg["model"],
        "language": cfg["language"],
        "hf_token": enc_token,
    }
    try:
        CONFIG_PATH.write_text(json.dumps(to_save))
    except Exception:
        st.warning("Failed to save config.")


def reset_state():
    for k in ["phase", "audio_path", "file_id", "segments", "turns"]:
        st.session_state[k] = None if k != "phase" else "empty"


def generate_file_id(data: bytes, name: str) -> str:
    digest = hashlib.sha256(data[:64]).hexdigest()[:8]
    return f"{name}-{len(data)}-{digest}"


# --- Updated cache key functions using file_id instead of path ---

def _cache_key(file_id: str, model: str, language: str) -> str:
    raw = f"{file_id}-{model}-{language}"
    return hashlib.md5(raw.encode()).hexdigest()


def load_cached_segments(file_id: str, model: str, language: str) -> List[Dict]:
    key = _cache_key(file_id, model, language)
    fpth = CACHE_DIR_SEG / f"{key}.json"
    if fpth.exists():
        try:
            return json.loads(fpth.read_text())
        except:
            fpth.unlink()
    return []


def save_cached_segments(file_id: str, model: str, language: str, segs: List[Dict]):
    key = _cache_key(file_id, model, language)
    fpth = CACHE_DIR_SEG / f"{key}.json"
    fpth.write_text(json.dumps(segs))


def load_cached_turns(file_id: str, model: str, language: str) -> List[Dict]:
    key = _cache_key(file_id, model, language)
    fpth = CACHE_DIR_DIAR / f"{key}.json"
    if fpth.exists():
        try:
            return json.loads(fpth.read_text())
        except:
            fpth.unlink()
    return []


def save_cached_turns(file_id: str, model: str, language: str, turns: List[Dict]):
    key = _cache_key(file_id, model, language)
    fpth = CACHE_DIR_DIAR / f"{key}.json"
    fpth.write_text(json.dumps(turns))


# --------------------------------------------------------------


def show_transcript(segments: List[Dict]):
    if not segments:
        return
    segs = sorted(segments, key=lambda s: s.get("start", 0))
    md, txt = [], []
    for s in segs:
        start = timedelta(seconds=int(s["start"]))
        end = timedelta(seconds=int(s["end"]))
        text = s["text"].strip()
        md.append(f"**[{start}–{end}]** {text}\n\n")
        txt.append(f"[{start}–{end}] {text}\n\n")
    md_blob = "".join(md)
    txt_blob = "".join(txt)

    with st.expander("Transcript Segments", icon=":material/article:"):
        _, _, c2 = st.columns([1, 6, 1])
        c2.download_button(
            ".txt",
            txt_blob,
            file_name="transcript.txt",
            icon=":material/download:",
            key=f"dl_txt_{st.session_state.file_id}"
        )
        st.markdown(md_blob)


def clear_cache():
    ap = st.session_state.get("audio_path")
    if ap:
        Path(ap).unlink(missing_ok=True)
    base = Path.home() / ".cache" / "sonify"
    if base.exists():
        for f in base.rglob("*"):
            f.unlink(missing_ok=True)
    reset_state()


def sidebar_settings():
    st.sidebar.header("Settings")
    cfg = st.session_state.cfg

    models = ["tiny", "base", "small", "medium", "large"]
    sel = st.sidebar.selectbox("Whisper model", models, index=models.index(cfg["model"]))
    if st.session_state.prev_model and sel != st.session_state.prev_model:
        reset_state()
    cfg["model"] = sel
    st.session_state.prev_model = sel

    cfg["language"] = st.sidebar.text_input("Language code", value=cfg["language"])
    cfg["hf_token"] = st.sidebar.text_input("HuggingFace token", value=cfg["hf_token"], type="password")
    persist_config()

    st.sidebar.markdown("---")
    if not st.session_state.confirm_clear:
        if st.sidebar.button("Clear All Cached Files and Restart"):
            st.session_state.confirm_clear = True
            st.rerun()
    else:
        st.sidebar.warning("This cannot be undone. Clear all cache?")
        y, n = st.sidebar.columns(2)
        if y.button("Yes, clear"):
            clear_cache()
            st.sidebar.success("Cache cleared.")
            st.session_state.file_uploader_key += 1
            st.session_state.confirm_clear = False
            st.rerun()
        if n.button("Cancel"):
            st.session_state.confirm_clear = False


def handle_upload():
    up = st.file_uploader("Upload audio file", type=AUDIO_TYPES,
                          key=st.session_state.file_uploader_key, accept_multiple_files=False)
    if up and st.session_state.phase == "empty":
        data = up.read()
        sid = generate_file_id(data, up.name)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix)
        tmp.write(data)
        st.session_state.audio_path = tmp.name
        st.session_state.file_id = sid
        st.session_state.phase = "uploaded"
        st.rerun()
    if not up and st.session_state.phase not in ["empty", "transcribing"]:
        ap = st.session_state.audio_path
        if ap:
            Path(ap).unlink(missing_ok=True)
        reset_state()
        st.rerun()


def handle_transcription():
    phase = st.session_state.phase
    cfg = st.session_state.cfg

    if phase == "uploaded":
        # use file_id for cache lookup
        fid = st.session_state.file_id
        cached_segs = load_cached_segments(fid, cfg["model"], cfg["language"])
        if cached_segs:
            st.session_state.segments = cached_segs
            st.session_state.phase = "transcribed"
            cached_turns = load_cached_turns(fid, cfg["model"], cfg["language"])
            if cached_turns:
                st.session_state.turns = cached_turns
                st.session_state.phase = "diarized"
            st.rerun()
        build_navigation()

    elif phase == "transcribing":
        build_navigation()
        st.session_state.prog_bar = st.progress(0.0)
        st.session_state.prog_text = st.empty()
        prog_segs = st.expander("Transcript Segments so far", icon=":material/article:")
        md_placeholder = prog_segs.empty()
        segs = []
        wav = cached_wav(st.session_state.audio_path)
        pbar = st.session_state.prog_bar
        ptxt = st.session_state.prog_text
        t0 = time.time()

        for u in transcribe_stream(
                wav,
                model_name=cfg["model"],
                language=cfg["language"],
                chunk_size=30,
        ):
            if st.session_state.phase != "transcribing":
                st.warning("Stopped by user.")
                return

            idx, tot, prog = u["chunk_index"], u["total_chunks"], u["progress"]
            elapsed = time.time() - t0
            eta = (elapsed / prog - elapsed) if prog > 0 else 0.0

            # update progress
            pbar.progress(prog)
            ptxt.text(f"{idx}/{tot} chunks  |  {int(prog * 100):3d}%  |  "
                      f"Elapsed {format_hms(elapsed)}  |  ETA {format_hms(eta)}")
            segs.extend(u["segments"])
            md_lines = []
            for s in segs:
                start = timedelta(seconds=int(s["start"]))
                end = timedelta(seconds=int(s["end"]))
                text = s["text"].strip()
                md_lines.append(f"**[{start}–{end}]** {text}\n\n")
            md_blob = "".join(md_lines)

            # 3) Overwrite the placeholder’s content
            md_placeholder.markdown(md_blob)

        st.session_state.segments = segs
        save_cached_segments(st.session_state.file_id, cfg["model"], cfg["language"], segs)
        st.session_state.phase = "transcribed"
        pbar.empty()
        ptxt.empty()
        prog_segs.empty()
        st.rerun()

    elif phase == "transcribed":
        build_navigation()
        show_transcript(st.session_state.segments)


def build_navigation():
    c1, _, c2 = st.columns([1, 6, 1])
    if st.session_state.phase == 'uploaded':
        if c1.button("Start Transcription", icon=":material/play_arrow:"):
            st.session_state.phase = "transcribing"
            st.rerun()
    if st.session_state.phase in ["transcribing", "diarizing"]:
        if c1.button("Cancel", icon=":material/cancel:"):
            st.session_state.phase = "uploaded"
            st.rerun()
    if st.session_state.phase == "transcribed":
        hf_token = st.session_state.cfg.get("hf_token", "").strip()
        disabled_sep = hf_token == ""
        if c1.button(
                "Separate Speakers",
                disabled=(st.session_state.phase != "transcribed" or disabled_sep),
                help="You need to enter a HuggingFace token in Settings before running diarization.",
                icon=":material/record_voice_over:"
        ):
            if disabled_sep:
                st.warning("You need to enter a HuggingFace token in Settings before running diarization.")
            else:
                st.session_state.phase = "diarizing"
                st.rerun()
        if c2.button("Restart", icon=":material/sync:"):
            reset_state()
            st.session_state.file_uploader_key += 1
            st.rerun()


def handle_diarization():
    phase = st.session_state.phase
    cfg = st.session_state.cfg
    fid = st.session_state.file_id
    mdl = cfg["model"]
    lang = cfg["language"]

    if phase == "diarizing":
        build_navigation()
        # placeholders for live updates
        bar = st.progress(0.0)
        txt = st.empty()

        cfg = st.session_state.cfg
        fid = st.session_state.file_id
        mdl = cfg["model"]
        lang = cfg["language"]
        turns = load_cached_turns(fid, mdl, lang)
        if not turns:
            # define our Streamlit callback
            def progress_cb(step_name, completed, total):
                pct = completed / total if total else 0.0
                pct = max(0.0, min(pct, 1.0))  # clamp into [0,1]
                bar.progress(pct)
                txt.text(f"{step_name}: {completed}/{total} ({pct:.0%})")

            # run with live updates
            turns = diarize_audio(
                st.session_state.audio_path,
                st.session_state.segments,
                cfg["hf_token"],
                progress_callback=progress_cb
            )
            save_cached_turns(fid, mdl, lang, turns)

            # finalize
        st.session_state.turns = turns
        st.session_state.phase = "diarized"
        st.rerun()

    elif phase == "diarized":
        build_navigation()
        turns = st.session_state.turns
        show_transcript(st.session_state.segments)
        prev_speaker = None
        prev_start_time = None
        buffer_msg = None
        stt = None
        all_speakers = set([t["speaker"] for t in turns])
        with st.expander("Assign Real Names to Speakers", icon=":material/account_circle:"):
            if st.session_state.speaker_names == {}:
                st.session_state.speaker_names = {lbl: lbl for lbl in all_speakers}
            for lbl in all_speakers:
                st.session_state.speaker_names[lbl] = st.text_input(f"Name for {lbl}", key=lbl, value=st.session_state.speaker_names.get(lbl))

        with st.expander("Speaker Diarization", icon=":material/record_voice_over:"):
            speaker_txt = ""
            for t in turns:
                # init
                if prev_start_time is None and prev_speaker is None:
                    stt = timedelta(seconds=int(t["start"]))
                    prev_speaker = st.session_state.speaker_names[t["speaker"]]
                    buffer_msg = t['text']
                if prev_speaker == st.session_state.speaker_names[t["speaker"]]:
                    buffer_msg += f" {t['text']}"
                else:
                    prev_start_time = t["start"]
                    ent = timedelta(seconds=int(t["start"]))
                    speaker_txt += f"**{prev_speaker}** [{stt}–{ent}]: {buffer_msg}\n\n"
                    stt = timedelta(seconds=int(t["start"]))
                    prev_speaker = st.session_state.speaker_names[t["speaker"]]
                    buffer_msg = t['text']
            if len(turns) > 0:
                ent = timedelta(seconds=int(turns[-1]["start"]))
                speaker_txt += f"**{prev_speaker}** [{stt}–{ent}]: {buffer_msg}\n\n"
            d1, _, d2 = st.columns([1, 6, 1])
            d1.download_button(
                "raw .json",
                json.dumps(turns, indent=2),
                icon=":material/download:",
                file_name=f"diarization_{st.session_state.file_id}.json",
                key=f"dl_diar_json_{st.session_state.file_id}"
            )
            d2.download_button(
                "adjusted .txt",
                speaker_txt,
                icon=":material/download:",
                file_name=f"diarization_{st.session_state.file_id}.txt",
                key=f"dl_diar_txt_{st.session_state.file_id}"
            )
            st.markdown(speaker_txt)


def main():
    st.set_page_config(page_title="Sonify – Transcription", layout="wide")
    init_session()
    sidebar_settings()

    handle_upload()
    if st.session_state.audio_path:
        st.audio(
            open(st.session_state.audio_path, "rb"),
            format=f"audio/{Path(st.session_state.audio_path).suffix[1:]}"
        )

    handle_transcription()
    handle_diarization()


if __name__ == "__main__":
    main()
