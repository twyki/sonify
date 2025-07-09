import streamlit as st
from sonify.utils.session import init_session
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"Torchaudio's I/O functions now support par-call backend dispatch.*"
)
warnings.filterwarnings(
    "ignore",
    message=r"Module 'speechbrain\.pretrained' was deprecated.*"
)

init_session()
st.set_page_config(page_title="Sonify – Home", layout="wide")

st.title("Sonify CLI App")
st.write(
    """
    Welcome to Sonify!  
    Use the sidebar (left) to navigate:
    1. Transcribe & Diarize  
    3. a Guide how to obtain a hugging face token
    2. Settings  
    """
)
