# Sonify

Audio transcription and speaker diarization tool, packaged for easy installation.

## Installation

```bash
pip install .
```

## CLI Usage

```bash
sonify path/to/audio.mp3 --model small --lang en --hf_token YOUR_TOKEN
```

## Streamlit App

```bash
sonify-app
```

Or:

```bash
streamlit run -m sonify.streamlit_app
```
