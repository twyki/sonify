import streamlit as st


def init_session():
    """Initialize session state and decrypt stored HF token if present."""

    st.session_state.setdefault("cfg", {
        "model": "medium",
        "language": "de",
        "hf_token": st.secrets["hf_token"]
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
