import streamlit as st
import uuid


def init_session():
    # Unique ID per user session
    if 'session_id' not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    if 'user_id' not in st.session_state:
        st.session_state.user_id = uuid.uuid4().hex
    if 'cfg' not in st.session_state:
        st.session_state.cfg = {
            'model': 'base',
            'language': 'auto',
            'hf_token': f"{st.secrets.hf_token}",
        }
    # Batch processing state
    if 'batch_phase' not in st.session_state:
        st.session_state.batch_phase = 'start'
    if 'batch_key' not in st.session_state:
        st.session_state.batch_key = 0
    # hold uploaded files list
    if 'batch_files' not in st.session_state:
        st.session_state.batch_files = None

    # Speaker diarization names
    if 'speaker_names' not in st.session_state:
        st.session_state.speaker_names = {}
    # Workflow state
    if 'phase' not in st.session_state:
        st.session_state.phase = "start"
    if 'audio_path' not in st.session_state:
        st.session_state.audio_path = None
    if 'file_id' not in st.session_state:
        st.session_state.file_id = None
    if 'segments' not in st.session_state:
        st.session_state.segments = []
    if 'turns' not in st.session_state:
        st.session_state.turns = []
    if 'file_uploader_key' not in st.session_state:
        st.session_state.file_uploader_key = 0
    if 'speaker_names' not in st.session_state:
        st.session_state.speaker_names = {}


def reset_state():
    for k in ["phase", "audio_path", "file_id", "segments", "turns", 'batch_phase', 'batch_key', 'batch_files', ]:
        st.session_state[k] = None if k not in ("phase", 'batch_phase') else "start"
    init_session()
