import streamlit as st
from sonify.utils.session import init_session

init_session()
st.title("How to Obtain and Configure Your Hugging Face Token")
st.markdown(
    """
    **Why you need it**

    Speaker separation in Sonify uses Hugging Face's `pyannote.audio` services,
    which require authentication via an API token.
    """
)

st.header("Step 1: Create a Hugging Face Account")
st.markdown(
    "1. Visit [https://huggingface.co/join](https://huggingface.co/join) and sign up.\n"
    "2. Verify your email address."
)

st.header("Step 2: Generate an Access Token")
st.markdown(
    "1. Go to your profile settings: \n"
    "   [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)\n"
    "2. Click **New token**.\n"
    "3. Give it a descriptive name (e.g., `Sonify Access`).\n"
    "4. Select the **`read`** scope.\n"
    "5. Click **Generate**, and copy the token."
)
st.header("Step 3: Accept Pyannote Model Conditions")
st.markdown(
    "Before you can use the `speaker-diarization-3.1` pipeline, you need to accept its terms:\n"
    "1. navigate to [https://huggingface.co/pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) and accept the user conditions.\n"
    "2. navigate to  [https://huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and accept the user conditions."
)
st.header("Step 4: Configure Sonify to Use Your Token")
token_input = st.text_input("Hugging Face Token", type="password", value=st.session_state.cfg["hf_token"])
if st.button("Save Token"):
    if token_input:
        st.session_state.cfg["hf_token"] = token_input
        st.success("Token saved to session state.")
    else:
        st.error("Please enter a valid token.")
st.info("Once configured, return to 'Transcribe & Diarize' to run speaker separation.")
