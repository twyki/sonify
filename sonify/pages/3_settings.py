import streamlit as st
from sonify.utils.session import init_session, reset_state

MODELS = ["tiny", "base", "small", "medium", "large"]
LANGUAGES_DICT = {
    "auto detected": "auto",
    "afrikaans": "af",
    "albanian": "sq",
    "amharic": "am",
    "arabic": "ar",
    "armenian": "hy",
    "assamese": "as",
    "azerbaijani": "az",
    "bashkir": "ba",
    "basque": "eu",
    "belarusian": "be",
    "bengali": "bn",
    "bosnian": "bs",
    "breton": "br",
    "bulgarian": "bg",
    "cantonese": "yue",
    "catalan": "ca",
    "croatian": "hr",
    "czech": "cs",
    "danish": "da",
    "dutch": "nl",
    "english": "en",
    "estonian": "et",
    "faroese": "fo",
    "finnish": "fi",
    "french": "fr",
    "galician": "gl",
    "georgian": "ka",
    "german": "de",
    "greek": "el",
    "gujarati": "gu",
    "haitian creole": "ht",
    "hawaiian": "haw",
    "hebrew": "he",
    "hindi": "hi",
    "hungarian": "hu",
    "icelandic": "is",
    "indonesian": "id",
    "irish": "ga",
    "italian": "it",
    "japanese": "ja",
    "javanese": "jw",
    "kazakh": "kk",
    "khmer": "km",
    "korean": "ko",
    "kannada": "kn",
    "latin": "la",
    "latvian": "lv",
    "lithuanian": "lt",
    "luxembourgish": "lb",
    "macedonian": "mk",
    "malagasy": "mg",
    "malay": "ms",
    "malayalam": "ml",
    "maltese": "mt",
    "maori": "mi",
    "marathi": "mr",
    "mongolian": "mn",
    "myanmar": "my",
    "nepali": "ne",
    "norwegian": "no",
    "nynorsk": "nn",
    "occitan": "oc",
    "pashto": "ps",
    "persian": "fa",
    "polish": "pl",
    "portuguese": "pt",
    "punjabi": "pa",
    "romanian": "ro",
    "russian": "ru",
    "sanskrit": "sa",
    "serbian": "sr",
    "shona": "sn",
    "sindhi": "sd",
    "slovak": "sk",
    "slovenian": "sl",
    "somali": "so",
    "spanish": "es",
    "sundanese": "su",
    "swahili": "sw",
    "swedish": "sv",
    "tamil": "ta",
    "tatar": "tt",
    "telugu": "te",
    "thai": "th",
    "tibetan": "bo",
    "turkish": "tr",
    "turkmen": "tk",
    "ukrainian": "uk",
    "urdu": "ur",
    "uzbek": "uz",
    "vietnamese": "vi",
    "welsh": "cy",
    "yiddish": "yi",
    "yoruba": "yo"
}

try:
    cfg = st.session_state.cfg
except AttributeError:
    init_session()  # your function that does st.session_state.cfg = {...}
    cfg = st.session_state.cfg

st.header("Settings")
default_model = cfg.get("model", "small")
if default_model not in MODELS:
    default_model = "small"
default_model_index = MODELS.index(default_model)

current_model = st.selectbox(
    "Whisper model size",
    options=MODELS,
    index=default_model_index,
    key="model-sb",
    help="Larger models give better accuracy but need more RAM/CPU."
)

if current_model in ("medium", "large"):
    st.info(
        "Models above “small” (~1 GB+) are resource-heavy and meant for use locally."
    )

all_langs = sorted([n for n in LANGUAGES_DICT if n != "auto detected"],
                   key=lambda s: s.lower())
language_options = ["auto detected"] + all_langs

default_lang = cfg.get("language", "auto detected")
if default_lang not in language_options:
    default_lang = "auto detected"
default_lang_index = language_options.index(default_lang)

current_lang = LANGUAGES_DICT[st.selectbox(
    "Transcription language",
    options=language_options,
    index=default_lang_index,
    key="language-sb",
    help="Start typing to filter"
)]

current_token = st.text_input(
    "HuggingFace token",
    value=cfg.get("hf_token", ""),
    type="password",
    help="Enter your HF access token.",
    key="hf_token-sb"
)
st.markdown(
    """
    <style>
        [title="Show password text"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

if cfg.get("model") != current_model or cfg.get("language") != current_lang or cfg.get("hf_token") != current_token:
    if st.session_state.phase != "start":
        st.warning("Changes during transcription/diarization will result in a loss of progress.")
    # Create two columns; button lives in the narrow right column
    _, _, button_col = st.columns([1, 6, 1])

    if button_col.button("Save", icon=":material/save:", type="primary"):
        # Persist into cfg
        cfg["model"] = current_model
        cfg["language"] = current_lang
        cfg["hf_token"] = current_token
        st.success("Settings saved. You can now proceed to Transcribe & Diarize.")
        st.session_state.changed_cfg = False
        if st.session_state.phase != "start":
            reset_state()
            st.rerun()
        st.rerun()
        # Optionally reset state so changes take effect immediately:
        # _raw_reset()
