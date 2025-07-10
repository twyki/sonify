import streamlit as st
import json
from sonify.transcribe import transcribe_with_cache
from sonify.diarize import diarize_audio
from datetime import timedelta
from pathlib import Path
from typing import List, Dict
from sonify.utils.session import reset_state, init_session


try:
    cfg = st.session_state.cfg
except AttributeError:
    init_session()  # your function that does st.session_state.cfg = {...}
    cfg = st.session_state.cfg
st.set_page_config(page_title="Sonify - Single Demo", page_icon=":material/speaker:", layout="wide")
AUDIO_TYPES = ["mp3", "wav", "m4a", "flac", "aac", "opus", "ogg"]

BADGE_CSS = """
<style>
.badge {
  display: inline-block;
  padding: 0.15em 0.4em;
  font-size: 70%;
  font-weight: 600;
  line-height: 1;
  color: #fff;
  border-radius: 0.2rem;
  margin-left: 0.3rem;
}
/* Color ramp from tiny (light) to large (dark) */
.badge-tiny   { background-color: #cce5ff; color: #004085; }
.badge-base   { background-color: #99ccff; }
.badge-small  { background-color: #66b2ff; }
.badge-medium { background-color: #3385ff; }
.badge-large  { background-color: #005ce6; }
.badge-lang   { background-color: #dec484; }
.badge-phase   { background-color: #6f9f9c; }
}
</style>
"""
st.markdown(BADGE_CSS, unsafe_allow_html=True)


def header_with_badges(title: str):
    model = cfg.get("model", "unknown")
    lang = cfg.get("language", "unknown")
    # If your language is stored as ISO code, and you want human name:
    reverse_map = {v: k.title() for k, v in st.session_state.get("LANGUAGES", {}).items()}
    lang_name = reverse_map.get(lang, lang)

    st.markdown(
        f"## {title}\n"
        f"<span class='badge badge-{model}'>model: {model}</span>"
        f"<span class='badge badge-lang'>lang: {lang_name}</span>"
        f"<span class='badge badge-phase'>phase: {st.session_state.phase}</span>",
        unsafe_allow_html=True
    )


def format_hms(seconds: float) -> str:
    hrs, rem = divmod(int(seconds), 3600)
    mins, secs = divmod(rem, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


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
    chl_exp = st.session_state.phase == 'transcribed'
    with st.expander("Transcript Segments", icon=":material/article:", expanded=chl_exp):
        _, _, c2 = st.columns([1, 6, 1])
        c2.download_button(
            ".txt", txt_blob,
            file_name="transcript.txt",
            icon=":material/download:",
            key=f"dl_txt_{st.session_state.file_id}",
            type="primary"
        )
        st.markdown(md_blob)


def handle_upload():
    up = st.file_uploader(
        "Upload audio file",
        type=AUDIO_TYPES,
        key=st.session_state.file_uploader_key
    )
    if up and st.session_state.phase == "start":
        data = up.read()

        # Compute a stable filename from the file contents
        import hashlib, tempfile
        hash_hex = hashlib.sha256(data).hexdigest()
        suffix = Path(up.name).suffix

        # Use a fixed cache directory under the system temp folder
        cache_dir = Path(tempfile.gettempdir()) / f"sonify_uploads_{st.session_state.session_id}"
        cache_dir.mkdir(parents=True, exist_ok=True)

        tmp_path = cache_dir / f"{hash_hex}{suffix}"
        # Only write if we haven’t already stored this exact upload
        if not tmp_path.exists():
            tmp_path.write_bytes(data)

        st.session_state.audio_path = str(tmp_path)
        st.session_state.phase = "uploaded"
        st.rerun()


def handle_transcription():
    phase = st.session_state.phase

    if phase == "uploaded":
        build_navigation()

    elif phase == "transcribing":
        build_navigation()
        bar = st.progress(0.0)
        txt = st.empty()
        prog_segs = st.expander(
            "Transcript Segments so far", icon=":material/article:"
        )
        md_placeholder = prog_segs.empty()

        def cb(completed: int, total: int):
            """
            Progress callback for cached transcription.
            """
            prog = completed / total if total else 0
            bar.progress(prog)
            txt.text(f"{completed}/{total} chunks | {prog:.0%}")

        result = transcribe_with_cache(
            st.session_state.audio_path,
            model_name=cfg["model"],
            language=cfg["language"],
            chunk_size=30,
            _progress_callback=cb,
        )

        segments = result.get("segments", [])
        md_lines = []
        for seg in segments:
            start = timedelta(seconds=int(seg["start"]))
            end = timedelta(seconds=int(seg["end"]))
            text = seg["text"].strip()
            md_lines.append(
                f"**[{start}–{end}]** {text}\n\n"
            )
        md_placeholder.markdown("".join(md_lines))

        st.session_state.segments = segments
        bar.empty()
        txt.empty()
        prog_segs.empty()
        st.session_state.phase = "transcribed"
        st.rerun()

    elif phase == "transcribed":
        build_navigation()
        show_transcript(st.session_state.segments)


def build_navigation():
    c1, _, c2 = st.columns([1, 6, 1])
    if c2.button("Restart", icon=":material/sync:"):
        reset_state()
        st.session_state.file_uploader_key += 1
        st.rerun()
    if st.session_state.phase == 'uploaded':
        if c1.button("Start Transcription", icon=":material/play_arrow:", type="primary"):
            st.session_state.phase = "transcribing"
            st.rerun()
    if st.session_state.phase in ["transcribing", "diarizing"]:
        if c1.button("Cancel", icon=":material/cancel:", type="secondary"):
            st.session_state.phase = "uploaded"
            st.rerun()
    if st.session_state.phase == "transcribed":
        hf_token = cfg.get("hf_token", "").strip()
        disabled_sep = hf_token == ""
        if c1.button("Separate Speakers",
                     disabled=(st.session_state.phase != "transcribed" or disabled_sep),
                     help="You need to enter a HuggingFace token in Settings before running diarization.",
                     icon=":material/record_voice_over:", type="primary"):
            if disabled_sep:
                st.warning("You need to enter a HuggingFace token in Settings before running diarization.")
            else:
                st.session_state.phase = "diarizing"
                st.rerun()


def handle_diarization():
    phase = st.session_state.phase

    if phase == "diarizing":
        build_navigation()
        # placeholders for live updates
        bar = st.progress(0.0)
        txt = st.empty()
        show_transcript(st.session_state.segments)
        fid = st.session_state.file_id
        mdl = cfg["model"]
        lang = cfg["language"]

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
            _progress_callback=progress_cb
        )
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
        with st.expander("Assign Names to Speakers", icon=":material/account_circle:", expanded=True):
            if st.session_state.speaker_names == {}:
                st.session_state.speaker_names = {lbl: lbl for lbl in all_speakers}
            for lbl in all_speakers:
                st.session_state.speaker_names[lbl] = st.text_input(f"Name for {lbl}", key=lbl, value=st.session_state.speaker_names.get(lbl))

        with st.expander("Speaker Diarization", icon=":material/record_voice_over:", expanded=True):
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
                key=f"dl_diar_json_{st.session_state.file_id}",
                type="secondary"
            )
            d2.download_button(
                "adjusted .txt",
                speaker_txt,
                icon=":material/download:",
                file_name=f"diarization_{st.session_state.file_id}.txt",
                key=f"dl_diar_txt_{st.session_state.file_id}",
                type="primary"
            )
            st.markdown(speaker_txt)


try:
    header_with_badges("Transcribe & Diarize")
    handle_upload()
    if st.session_state.audio_path:
        st.audio(open(st.session_state.audio_path, "rb"),
                 format=f"audio/{Path(st.session_state.audio_path).suffix[1:]}")
        handle_transcription()
        handle_diarization()
except Exception as e:
    st.error(f"An unexpected error occurred while processing your files: {e}\n"
             f"if this occurs multiple times, please contact us")
    if st.button("Restart Workflow", key='error_restart'):
        # Clear all non-ID state and rerun
        for key in list(st.session_state.keys()):
            if key not in ('session_id', 'user_id', 'cfg'):
                del st.session_state[key]
    st.rerun()
