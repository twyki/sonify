import streamlit as st
from sonify.utils.session import reset_state

st.header("Settings")
cfg = st.session_state.cfg
models = ["tiny", "base", "small", "medium", "large"]
sel = st.selectbox("Whisper model", models, index=models.index(cfg["model"]))
if st.session_state.prev_model and sel != st.session_state.prev_model:
    reset_state()
cfg["model"] = sel
st.session_state.prev_model = sel
cfg["language"] = st.text_input("Language code", value=cfg["language"])
if st.session_state.prev_language and sel != st.session_state.prev_language:
    reset_state()
cfg["hf_token"] = st.text_input("HuggingFace token", value=cfg["hf_token"], type="password")