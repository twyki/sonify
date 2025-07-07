import streamlit as st
from pathlib import Path
import tempfile
from datetime import timedelta

from .transcribe import cached_wav, transcribe_with_cache
from .diarize import diarize_audio

# Streamlit page config
st.set_page_config(
    page_title="Audio Transcription & Diarization",
    layout="wide"
)


@st.cache_data
def transcribe_cached(wav_path, model, language):
    return transcribe_with_cache(wav_path, model, language)


@st.cache_data
def wav_cached(input_path):
    return cached_wav(input_path)


def main():
    st.title("Audio Transcription & Diarization")

    with st.sidebar:
        st.header("Settings")
        model = st.selectbox(
            "Whisper model size",
            ["tiny", "base", "small", "medium", "large"],
            index=2
        )
        language = st.text_input("Language code", "en")
        do_diar = st.checkbox("Enable speaker diarization")
        hf_token = None
        if do_diar:
            hf_token = st.text_input(
                "HuggingFace token", type="password"
            )

    uploaded = st.file_uploader(
        "Upload audio file (mp3, wav, m4a, etc.)",
        type=["mp3", "wav", "m4a", "flac", "aac"]
    )
    if not uploaded:
        st.info("Please upload an audio file to get started.")
        return

    tmp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=Path(uploaded.name).suffix
    )
    tmp.write(uploaded.getvalue())
    tmp.flush()
    audio_path = tmp.name

    st.audio(audio_path)

    if st.button("Transcribe & Diarize"):
        with st.spinner("Transcribing..."):
            wav_file = wav_cached(audio_path)
            result = transcribe_cached(wav_file, model, language)
        transcript = result.get("text", "")
        st.subheader("Transcript")
        st.text_area("Full Transcript", transcript, height=300)

        if do_diar and hf_token:
            with st.spinner("Diarizing..."):
                diar = diarize_audio(
                    audio_path,
                    result.get("segments", []),
                    hf_token
                )
            st.subheader("Speaker Diarization")
            for turn in diar:
                start = timedelta(seconds=int(turn['start']))
                end = timedelta(seconds=int(turn['end']))
                st.markdown(
                    f"**{turn['speaker']}** [{start} - {end}]: {turn['text']}"
                )


if __name__ == "__main__":
    main()
