# PraisePlay AI - Sports Commentary Tracker (Streamlit Cloud Version)

# Requirements:
# - Python 3.9+
# - pip install openai-whisper nltk torch streamlit

import whisper
import streamlit as st
import os
import re
import tempfile
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
st.set_page_config(page_title="PraisePlay AI", page_icon="🏈", layout="centered")
st.markdown("""
    <style>
        .main {background-color: #0B0C10; color: #C5C6C7; font-family: 'Segoe UI', sans-serif;}
        .css-18e3th9 {background-color: #1F2833; color: #C5C6C7;}
        .css-1d391kg {background-color: #1F2833;}
        .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    </style>
""", unsafe_allow_html=True)

st.title("🏈 PraisePlay AI")
st.subheader("Analyze Sports Commentary for Positive Player Mentions")
st.markdown("Upload audio clips (MP3/WAV) from sports broadcasts to detect and track praise for your favorite players.")

# Player input
player_name = st.text_input("🔍 Enter Player's Name to Track", "Patrick Mahomes")

# Audio upload
audio_file = st.file_uploader("🎧 Upload an Audio Clip (MP3/WAV)", type=["mp3", "wav"])

# State variables
if "total_mentions" not in st.session_state:
    st.session_state.total_mentions = 0
if "positive_mentions" not in st.session_state:
    st.session_state.positive_mentions = 0

def analyze_transcript(transcript, player):
    st.markdown("---")
    st.subheader(f"📋 Transcript Snippets mentioning '{player}'")
    pattern = re.compile(rf"\b({re.escape(player)})\b", re.IGNORECASE)
    mentions = pattern.finditer(transcript)

    for match in mentions:
        st.session_state.total_mentions += 1
        snippet = transcript[max(0, match.start()-100):match.end()+100]
        sentiment = sia.polarity_scores(snippet)
        if sentiment['compound'] > 0.3:
            st.session_state.positive_mentions += 1
        st.markdown(f"**🎙️ Snippet:** {snippet}\n\n**📈 Sentiment:** {'Positive' if sentiment['compound'] > 0.3 else 'Neutral/Negative'}")

    st.markdown("---")
    st.success(f"🏆 Total Mentions of '{player}': {st.session_state.total_mentions} | Positive Mentions: {st.session_state.positive_mentions}")

if audio_file:
    st.audio(audio_file)
    st.info("Transcribing audio... please wait.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_file.read())
        result = model.transcribe(f.name)
        transcript = result["text"]
        os.remove(f.name)

    st.subheader("📝 Transcript")
    st.write(transcript)

    analyze_transcript(transcript, player_name)
