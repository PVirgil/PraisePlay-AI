# PraisePlay AI - MVP Version (Free Tools)

# Requirements:
# - Python 3.9+
# - Install packages: pip install openai-whisper nltk transformers torch streamlit

import whisper
import streamlit as st
import os
import re
from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

# Load Whisper Model
@st.cache_resource
def load_model():
    return whisper.load_model("base")  # Smaller models for speed

model = load_model()

# Load Sentiment Analyzer (VADER for simplicity)
sia = SentimentIntensityAnalyzer()

# Initialize Streamlit UI
st.title("🏈 PraisePlay AI: Mahomes Mention Tracker")
st.markdown("Upload an audio clip (MP3/WAV) from a football game broadcast to analyze positive mentions of Patrick Mahomes.")

audio_file = st.file_uploader("Upload Audio", type=["mp3", "wav"])

if audio_file:
    st.audio(audio_file)
    st.info("Transcribing audio...")
    # Save audio temporarily
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.read())

    result = model.transcribe("temp_audio.wav")
    transcript = result["text"]
    st.subheader("Transcript")
    st.write(transcript)

    # Analyze transcript
    st.subheader("Analyzing Mentions of Patrick Mahomes")
    pattern = re.compile(r"\b(Mahomes|Patrick Mahomes|Chiefs QB)\b", re.IGNORECASE)
    mentions = pattern.finditer(transcript)

    pos_count = 0
    total_mentions = 0
    st.write("### Mentions Detected:")

    for match in mentions:
        total_mentions += 1
        start = max(0, match.start() - 100)
        end = min(len(transcript), match.end() + 100)
        snippet = transcript[start:end]
        sentiment = sia.polarity_scores(snippet)
        label = "Positive" if sentiment['compound'] > 0.3 else "Neutral/Negative"
        if label == "Positive":
            pos_count += 1
        st.markdown(f"**Context**: {snippet}\n\n**Sentiment**: {label}")

    st.success(f"📊 Total Mentions: {total_mentions} | Positive Mentions: {pos_count}")

    # Clean up
    os.remove("temp_audio.wav")
