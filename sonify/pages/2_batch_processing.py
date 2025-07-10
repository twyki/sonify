import streamlit as st
from pathlib import Path
import tempfile
import hashlib
from datetime import timedelta

from sonify.utils.session import init_session
from sonify.transcribe import transcribe_with_cache
from sonify.diarize import diarize_audio

# Constants
AUDIO_TYPES = ["mp3", "wav", "m4a", "flac", "aac", "opus", "ogg"]
MAX_BATCH_FILES = 20  # reasonable limit


def format_hms(seconds: float) -> str:
    hrs, rem = divmod(int(seconds), 3600)
    mins, secs = divmod(rem, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


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
        files = st.session_state.get('batch_files', []) or []
        # Spinner to indicate processing
        with st.spinner("Processing files... Please wait."):
            for up in files:
                st.header(up.name)
                data = up.read()
                content_hash = hashlib.sha256(data).hexdigest()
                suffix = Path(up.name).suffix
                cache_dir = Path(tempfile.gettempdir()) / f"batch_{st.session_state.session_id}"
                cache_dir.mkdir(parents=True, exist_ok=True)
                path = cache_dir / f"{content_hash}{suffix}"
                if not path.exists():
                    path.write_bytes(data)

                st.audio(data, format=f"audio/{suffix[1:]}" )

                # Transcribe & Diarize
                res = transcribe_with_cache(
                    str(path),
                    model_name=st.session_state.cfg['model'],
                    language=st.session_state.cfg['language']
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
                    # Build adjusted transcript using merged segments logic
                    prev_start_time = None
                    prev_speaker = None
                    buffer_msg = ""
                    speaker_txt = ""

                    # Ensure speaker_names dict exists
                    if 'speaker_names' not in st.session_state:
                        st.session_state.speaker_names = {}

                    for t in turns:
                        # Initialize for first turn
                        if prev_start_time is None and prev_speaker is None:
                            stt = timedelta(seconds=int(t["start"]))
                            prev_speaker = st.session_state.speaker_names.get(t["speaker"], t["speaker"])
                            buffer_msg = t['text']

                        current_speaker = st.session_state.speaker_names.get(t["speaker"], t["speaker"])
                        # Same speaker: accumulate text
                        if current_speaker == prev_speaker:
                            buffer_msg += f" {t['text']}"
                        else:
                            ent = timedelta(seconds=int(t["start"]))
                            speaker_txt += f"**{prev_speaker}** [{stt}–{ent}]: {buffer_msg}\n\n"
                            # Reset for new speaker
                            stt = timedelta(seconds=int(t["start"]))
                            prev_speaker = current_speaker
                            buffer_msg = t['text']

                    # Flush last buffer
                    if turns:
                        ent = timedelta(seconds=int(turns[-1]["start"]))
                        speaker_txt += f"**{prev_speaker}** [{stt}–{ent}]: {buffer_msg}\n\n"

                    st.download_button(
                        f"Download .txt", speaker_txt,
                        file_name=f"{up.name}_transcript.txt",
                        icon=":material/download:",
                        type="primary"
                    )
                    st.markdown(speaker_txt)

main()
