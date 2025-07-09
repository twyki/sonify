**Sonify: Audio Transcription & Speaker Diarization**

A straightforward toolset for transcribing audio, identifying speakers, and launching either a command-line or Streamlit interface. Built on OpenAI Whisper and PyAnnote.

---

## Features

* **High-accuracy transcription** via OpenAI Whisper
* **Speaker diarization** with PyAnnote
* **Chunked processing** for long recordings
* **Cache management** to avoid re-transcribing
* **Dual interfaces**: CLI and Streamlit web app

---

## Acknowledgements

* **PyAnnote Audio** for state-of-the-art speaker diarization, based on the research and implementation by Hervé Bredin and the PyAnnote community.
* **OpenAI Whisper** for robust automatic speech recognition.

---

## Prerequisites

1. **Python 3.10+**
2. **FFmpeg** (for audio conversion)
3. **Hugging Face Access Token** (for PyAnnote models; see `guide_hf_token.md`)

---

## Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/twyki/sonify.git
   cd sonify
   ```

2. **Install system dependencies**

   * **Ubuntu / Debian:**

     ```bash
     sudo apt update && sudo apt install -y ffmpeg
     ```
   * **macOS (Homebrew):**

     ```bash
     brew install ffmpeg
     ```
   * **Windows:**
     Download and install latest build from [FFmpeg.org](https://ffmpeg.org).

3. **Install Python 3.10**

   * **Ubuntu / Debian:**

     ```bash
     sudo apt update && sudo apt install -y software-properties-common
     sudo add-apt-repository ppa:deadsnakes/ppa
     sudo apt update && sudo apt install -y python3.10 python3.10-venv python3.10-dev
     ```

   * **macOS (Homebrew):**

     ```bash
     brew install python@3.10
     ```

   * **Windows:**
     Download and install the Python 3.10 installer from [https://www.python.org/downloads/release/python-3100/](https://www.python.org/downloads/release/python-3100/).

4. **Hugging Face Token & Models**
   Refer to `guide_hf_token.md` for instructions on obtaining an access token and selecting the appropriate PyAnnote models.

5. **Create & activate a virtual environment**

   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   .\.venv\Scripts\activate    # Windows
   ```

6. **Install Python dependencies**

   ```bash
   pip install --upgrade pip
   pip install -e .
   ```

---

## Usage

### Command-Line Interface (CLI)

The `sonify` CLI focuses on simplicity: only the audio file path is required. All other settings use optional flags.

#### Command Syntax

```bash
sonify <audio_file> [OPTIONS]
```

* `<audio_file>` (required): Path to your audio file (e.g., `meeting.mp3`, `recording.wav`).

#### Available Options

* `-m, --model <MODEL_NAME>`
  Whisper model size: `tiny`, `base`, `small`, `medium`, or `large`. Default: `medium`.
* `-l, --lang <LANG_CODE>`
  ISO 639-1 language override (e.g., `en`, `de`, `fr`). Default: `de`.
* `--chunk-size <SECONDS>`
  Split audio into N-second segments before transcription.
* `-hft, --hf-token <TOKEN>`
  Hugging Face token for speaker diarization. If omitted, diarization is skipped.
* `-o, --output-dir <DIR>`
  Directory to save results. Default: `output`.
* `-f, --force`
  Ignore existing cache and re-run everything.
* `-v, --verbose`
  Show detailed logs.
* `-h, --help`
  Display help and exit.

#### Output Structure

By default, outputs are saved under `<output-dir>`:

```
<output-dir>/
├── {file_name}.transcript.txt    # Full transcript text
├── {file_name}.segments.txt      # Time-stamped segment list
└── {file_name}.diarized.txt      # Speaker-diarized transcript (if diarization ran)
```

#### Examples

```bash
# 1. Minimum required: just the file (small model, auto-detect lang, no diarization)
sonify audio_message.opus

# 2. Specify model and language
sonify interview.wav -m medium -l en

# 3. Full pipeline with diarization and chunking
sonify meeting.mp3 -m small -l en -hft $HF_TOKEN --chunk-size 30 -v

# 4. force refresh (full rerun)
sonify lecture.wav -f
```

---

### Streamlit Web App

The `sonify-app` wrapper provides an interactive browser interface.

#### Configuration

1. **Streamlit Secrets**
   Store your Hugging Face token in `.streamlit/secrets.toml`:

   ```toml
   hf_token = "<your_token_here>"
   ```
2. **.streamlit/config.toml (Optional)**
   Customize port or server settings:

   ```toml
   [server]
   port = 8501
   headless = true
   enableCORS = false
   ```

#### Launching the App

```bash
sonify-app
```

#### Interactive Features

* **Model Selection**: Choose Whisper model size from dropdown.
* **Language Override**: Select transcription language.
* **Upload**: Drag-and-drop audio files in the sidebar.
* **Listen along**: Build in Audio Player to listen, while the application transcribes your audio
* **Transcription View**: Live-updating text area with speaker labels and timestamps.
* **Download**: Export transcript and diarization.

## License

MIT © Tom Wysotzki
