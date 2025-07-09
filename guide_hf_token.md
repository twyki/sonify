# Guide: Hugging Face Token & Model Selection

This guide walks you through obtaining a Hugging Face access token and selecting the appropriate PyAnnote pipelines for speaker diarization in Sonify.

## 1. Create a Hugging Face Account

1. Visit [https://huggingface.co/join](https://huggingface.co/join) and sign up for a free account.
2. Verify your email address.

## 2. Generate an Access Token

1. Log in to [https://huggingface.co](https://huggingface.co).
2. Click your profile avatar (top-right) and select **Settings**.
3. In the sidebar, click **Access Tokens**.
4. Click **New token**, name it (e.g., `sonify-diarization`), and select the **`read`** scope.
5. Click **Generate** and copy the token. Store it securely (e.g., in a password manager).

## 3. Accept PyAnnote Model Conditions

Before using the official PyAnnote diarization pipelines, you must accept their model licenses:

1. Open [https://huggingface.co/pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) and click **Accept**.
2. Open [https://huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and click **Accept**.

## 4. Configure Sonify to Use Your Token

### CLI Usage

When running the CLI, pass the token with the `-hft` flag:

```bash
sonify meeting.mp3 -hft <your_token_here>
```

### Streamlit Usage

In your project root, create `.streamlit/secrets.toml`:

```toml
[hf]
token = "<your_token_here>"
```

Sonify will auto-load this token for diarization.

---

**Security Reminder:**

* Never commit your token or secrets file to version control.
* Treat your token like a password.
