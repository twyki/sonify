import streamlit as st
import  uuid

def init_session():
    # Unique ID per user session
    if 'session_id' not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex


    st.session_state.setdefault("cfg", {
        "model": "small",
        "language": "auto",
        "hf_token": f"{st.secrets.hf_token}"
    })
    # Workflow state
    st.session_state.setdefault("phase", "start")
    st.session_state.setdefault("audio_path", None)
    st.session_state.setdefault("file_id", None)
    st.session_state.setdefault("segments", [])
    st.session_state.setdefault("turns", [])
    st.session_state.setdefault("file_uploader_key", 0)
    st.session_state.setdefault("speaker_names", {})


def reset_state():
    for k in ["phase", "audio_path", "file_id", "segments", "turns"]:
        st.session_state[k] = None if k != "phase" else "start"
