import json

import streamlit as st
from pathlib import Path
import tempfile
import hashlib
from datetime import timedelta

from sonify.utils.session import init_session, reset_state
from sonify.transcribe import transcribe_with_cache
from sonify.diarize import diarize_audio

# Constants
AUDIO_TYPES = ["mp3", "wav", "m4a", "flac", "aac", "opus", "ogg"]
MAX_BATCH_FILES = 20  # reasonable limit
if 'error_count' not in st.session_state:
    st.session_state.error_count = 0

st.set_page_config(page_title="Sonify - Batch Demo", page_icon=":material/speaker:", layout="wide")


def format_hms(seconds: float) -> str:
    hrs, rem = divmod(int(seconds), 3600)
    mins, secs = divmod(rem, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def process_turns(turns):
    if not turns:
        return ""
    speakers = {t['speaker'] for t in turns}
    # single-speaker fast path
    if len(speakers) == 1:
        spk = turns[0]['speaker']
        start = timedelta(seconds=int(turns[0]['start']))
        end = timedelta(seconds=int(turns[-1]['end']))
        text = " ".join(t['text'].strip() for t in turns)
        return f"{spk} [{start}–{end}]: {text}\n\n"
    # Build adjusted transcript using merged segments logic
    prev_start_time = None
    prev_speaker = None
    buffer_msg = ""
    speaker_txt = ""
    for t in turns:
        # init
        if prev_start_time is None and prev_speaker is None:
            stt = timedelta(seconds=int(t["start"]))
            prev_speaker = t["speaker"]
            buffer_msg = t['text']
        if prev_speaker == t["speaker"]:
            buffer_msg += f" {t['text']}"
        else:
            prev_start_time = t["start"]
            ent = timedelta(seconds=int(t["start"]))
            speaker_txt += f"**{prev_speaker}** [{stt}–{ent}]: {buffer_msg}\n\n"
            stt = timedelta(seconds=int(t["start"]))
            prev_speaker = t["speaker"]
            buffer_msg = t['text']
    if len(turns) > 0:
        ent = timedelta(seconds=int(turns[-1]["start"]))
        speaker_txt += f"**{prev_speaker}** [{stt}–{ent}]: {buffer_msg}\n\n"
    return speaker_txt


def main():
    init_session()
    st.title("Batch Processing")

    # Always-on Restart Button (top-right)
    _, btn_col = st.columns([9, 1])
    if btn_col.button("Restart", key='global_restart', icon=":material/sync:"):
        st.session_state.batch_phase = 'start'
        st.session_state.batch_key = st.session_state.get('batch_key', 0) + 1
        st.session_state.pop('batch_files', None)
        st.rerun()

    # Reset on page refresh if processing with no files
    if (st.session_state.get('batch_phase') == 'processing' and
            not st.session_state.get('batch_files')):
        st.session_state.batch_phase = 'start'
        st.session_state.batch_key = st.session_state.get('batch_key', 0) + 1

    # Initialize batch phase
    if 'batch_phase' not in st.session_state:
        st.session_state.batch_phase = 'start'
        st.session_state.batch_key = 0

    # Batch start phase
    if st.session_state.batch_phase == 'start':
        files = st.file_uploader(
            f"Upload up to {MAX_BATCH_FILES} audio files",
            type=AUDIO_TYPES,
            accept_multiple_files=True,
            key=f'batch_uploader_{st.session_state.batch_key}'
        )
        if files and len(files) > MAX_BATCH_FILES:
            st.warning(f"Limit to {MAX_BATCH_FILES} files. Currently: {len(files)}")
            return
        st.session_state.batch_files = files or None

        # Show start button when files selected
        if files:
            if st.button("Start Batch Processing", key='start_batch'):
                st.session_state.batch_phase = 'processing'
                st.rerun()
        return

    # Processing phase
    if st.session_state.batch_phase == 'processing':
        files = st.session_state.get("batch_files", []) or []
        processed_count = 0
        all_transcripts: dict[str, str] = {}
        with st.spinner("Processing files... Please wait."):
            total_files = len(files)
            progress_bar = st.progress(0)

            for index, up in enumerate(files, start=1):
                processed_count += 1
                progress_bar.progress(index / total_files)

                up.seek(0)
                data = up.read()
                content_hash = hashlib.sha256(data).hexdigest()
                suffix = Path(up.name).suffix
                cache_dir = (
                        Path(tempfile.gettempdir())
                        / f"batch_{st.session_state.session_id}"
                )
                cache_dir.mkdir(parents=True, exist_ok=True)
                path = cache_dir / f"{content_hash}{suffix}"

                if not path.exists():
                    path.write_bytes(data)

                # Transcribe & Diarize
                res = transcribe_with_cache(
                    str(path),
                    model_name=st.session_state.cfg['model'],
                    language=st.session_state.cfg['language'],
                    chunk_size=30,
                )
                segments = res.get('segments', [])
                turns = diarize_audio(
                    str(path), segments, st.session_state.cfg.get('hf_token', '')
                )

                # Expander for speaker diarization
                with st.expander(
                        f"Speaker Diarization: {up.name}",
                        icon=":material/record_voice_over:",
                        expanded=False
                ):

                    speaker_txt = process_turns(turns)
                    st.markdown(speaker_txt)
                    all_transcripts[up.name] = speaker_txt

        # Download button
        _, _, d1 = st.columns([1, 6, 1])
        d1.download_button(
            "all  transcripts .json",
            json.dumps(all_transcripts, indent=2),
            icon=":material/download:",
            file_name=f"all_transcirpts.text",
            key=f"all_transcripts_{st.session_state.batch_key}",
            type="primary"
        )


try:
    main()
except Exception as e:
    st.session_state.error_count += 1

    # If too many errors, stop the app
    if st.session_state.error_count >= 5:
        st.error("Application encountered too many errors and will stop. Please refresh your browser to restart.")
        st.stop()
    st.error(f"An unexpected error occurred while processing your files: {e}\n"
             f"if this occurs multiple times, please contact us")
    if st.button("Restart Workflow", key='error_restart'):
        reset_state()
    st.rerun()
