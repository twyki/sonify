# Sonify CLI

A simple CLI tool for Whisper transcription with caching, optional speaker diarization, and verbose debug logging.

## Features

* **Model selection**: Choose Whisper model size (`--model`, default `medium`).
* **Language support**: Specify transcription language (`--lang`, default `de`).
* **Caching**:

  * Audio conversions (16 kHz mono WAV) cached under `~/.cache/sonify/wav`.
  * Full transcriptions cached under `~/.cache/sonify/json`.
  * Optional per-chunk caches under `~/.cache/sonify/wav/chunks` when using `--chunk_size`.
* **Chunked transcription**: Split long audio into fixed-length segments (`--chunk_size`) and merge results.
* **Diarization**: Optional speaker diarization via HuggingFace API token (`--hf_token`).
* **Verbose logging**: Enable debug logs for cache hits, conversions, splits, and API calls (`--verbose`).
* **Force refresh**: Re-run and overwrite existing caches (`--force`).
* **Smart skipping**: Automatically skip transcription or diarization if outputs already exist.

## Installation

```bash
pip install sonify
```

## Usage

```bash
sonify <audio-file> [options]
```

### Options

| Flag                 | Description                                                                 |
| -------------------- | --------------------------------------------------------------------------- |
| `-m`, `--model`      | Whisper model size (tiny, base, small, medium, large). Default: `medium`.   |
| `-l`, `--lang`       | Language code (e.g., `en`, `de`, `fr`). Default: `de`.                      |
| `--hf_token`         | HuggingFace token for speaker diarization.                                  |
| `-O`, `--out_dir`    | Output directory for transcripts and diarization files. Default: `output/`. |
| `-c`, `--chunk_size` | Split audio into chunks of given length (in seconds) for per-chunk caching. |
| `-f`, `--force`      | Force re-transcription by clearing existing JSON cache.                     |
| `-v`, `--verbose`    | Enable debug logging (cache hits, conversions, chunk splits, etc.).         |

### Examples

1. **Basic transcription** (defaults: medium, de):

   ```bash
   sonify input.mp3
   ```

2. **English transcription with large model**:

   ```bash
   sonify input.wav -m large -l en
   ```

3. **Chunked transcription & verbose logs**:

   ```bash
   sonify long_audio.wav -c 300 -v
   ```

4. **With speaker diarization**:

   ```bash
   sonify interview.mp3 --hf_token YOUR_TOKEN
   ```

5. **Force refresh**:

   ```bash
   sonify input.wav --force
   ```

## Caching Details

* **Audio cache**: `~/.cache/sonify/wav/{hash}.wav` contains converted WAV.
* **Transcription cache**: `~/.cache/sonify/json/{hash}.json` stores Whisper results.
* **Chunk cache**: When using `--chunk_size`, segments saved to `~/.cache/sonify/wav/chunks/{full_hash}/chunk_NNN.wav` and cached JSON per chunk.

## Logging

* By default logs are at `INFO` level, showing high-level steps (e.g., transcript saved).
* Use `--verbose` (`DEBUG` level) to see detailed messages about cache hits, audio conversions, segmentation, and API calls.

---

# Streamlit App

```bash
sonify-app
```

Or:

```bash
streamlit run -m sonify.streamlit_app
```
