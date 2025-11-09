# PraisePlay AI - Sports Commentary Tracker (Groq API Only)

# Requirements:
# - Python 3.9+
# - pip install streamlit nltk requests

import streamlit as st
import os
import re
import tempfile
import requests
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

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

def transcribe_via_groq(file_path):
    api_key = st.secrets["GROQ_API_KEY"]
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    files = {"file": open(file_path, "rb")}
    data = {"model": "whisper-large-v3-turbo", "response_format": "text"}
    response = requests.post(url, headers=headers, files=files, data=data)
    response.raise_for_status()
    return response.text

if audio_file:
    st.audio(audio_file)
    st.info("Transcribing audio via Groq API... please wait.")
    file_extension = audio_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as f:
        f.write(audio_file.read())
        f.flush()
        transcript = transcribe_via_groq(f.name)
        os.remove(f.name)

    st.subheader("📝 Transcript")
    st.write(transcript)

    analyze_transcript(transcript, player_name)


