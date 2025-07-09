# sonify/streamlit_app.py

import streamlit as st
from sonify.utils.session import init_session

init_session()
st.set_page_config(page_title="Sonify – Home", layout="wide")

st.title("Sonify CLI App")
st.write(
    """
    Welcome to Sonify!  
    Use the sidebar (left) to navigate:
    1. Transcribe & Diarize  
    2. Settings  
    """
)
