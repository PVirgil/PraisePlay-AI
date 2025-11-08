# PraisePlay AI - Mahomes Mention Tracker (Live Mic + Audio File)

# Requirements:
# - Python 3.9+
# - pip install openai-whisper nltk torch streamlit sounddevice numpy scipy

import whisper
import streamlit as st
import os
import re
import sounddevice as sd
import numpy as np
import tempfile
from scipy.io.wavfile import write
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

# Load Whisper Model
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# Load Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# UI Setup
st.title("🏈 PraisePlay AI: Mahomes Mention Tracker")
st.markdown("Choose an input method to analyze mentions of Patrick Mahomes:")

input_mode = st.radio("Select Input Mode", ["Upload Audio File", "Live Microphone"])

# State variables
if "total_mentions" not in st.session_state:
    st.session_state.total_mentions = 0
if "positive_mentions" not in st.session_state:
    st.session_state.positive_mentions = 0

def analyze_transcript(transcript):
    pattern = re.compile(r"\b(Mahomes|Patrick Mahomes|Chiefs QB)\b", re.IGNORECASE)
    mentions = pattern.finditer(transcript)

    for match in mentions:
        st.session_state.total_mentions += 1
        snippet = transcript[max(0, match.start()-100):match.end()+100]
        sentiment = sia.polarity_scores(snippet)
        if sentiment['compound'] > 0.3:
            st.session_state.positive_mentions += 1
        st.markdown(f"**Snippet:** {snippet}\n\n**Sentiment:** {'Positive' if sentiment['compound'] > 0.3 else 'Neutral/Negative'}")

    st.success(f"📊 Total Mentions: {st.session_state.total_mentions} | Positive Mentions: {st.session_state.positive_mentions}")

# --- Option 1: Upload Audio File ---
if input_mode == "Upload Audio File":
    audio_file = st.file_uploader("Upload an MP3 or WAV file", type=["mp3", "wav"])
    if audio_file:
        st.audio(audio_file)
        st.info("Transcribing uploaded audio...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_file.read())
            result = model.transcribe(f.name)
            transcript = result["text"]
            os.remove(f.name)

        st.subheader("Transcript")
        st.write(transcript)

        st.subheader("Analysis")
        analyze_transcript(transcript)

# --- Option 2: Live Microphone Listening ---
else:
    DURATION = st.slider("Recording Chunk Duration (seconds)", 5, 30, 10)
    start_button = st.button("🎙️ Start Listening")
    stop_button = st.button("🛑 Stop Listening")

    def record_and_process():
        st.info("Recording... Speak near your microphone.")
        while True:
            audio_data = sd.rec(int(DURATION * 44100), samplerate=44100, channels=1, dtype='int16')
            sd.wait()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                write(tmp_wav.name, 44100, audio_data)
                result = model.transcribe(tmp_wav.name)
                transcript = result["text"]
                os.remove(tmp_wav.name)

            st.subheader("Transcript (Live Chunk)")
            st.write(transcript)

            st.subheader("Analysis")
            analyze_transcript(transcript)

    if start_button:
        record_and_process()

    if stop_button:
        st.warning("Stopped listening. Reload to restart.")
