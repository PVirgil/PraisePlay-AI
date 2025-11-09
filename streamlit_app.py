# PraisePlay AI - In-Depth Game Summary + Smart Mentions (Groq API)

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
st.subheader("Analyze Sports Commentary for Positive Player Mentions and Game Highlights")
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

positive_keywords = [
    "amazing", "brilliant", "excellent", "great", "incredible", "unbelievable",
    "fantastic", "phenomenal", "strong", "perfect", "outstanding", "dominant"
]

highlight_keywords = [
    "touchdown", "field goal", "interception", "pass", "run", "score", "throws",
    "catches", "breakaway", "drive", "sack", "pick"
]

def analyze_transcript(transcript, player):
    st.markdown("---")
    st.subheader(f"📋 Notable Mentions of '{player}'")
    pattern = re.compile(rf"\b({re.escape(player)})\b", re.IGNORECASE)
    mentions = pattern.finditer(transcript)

    for match in mentions:
        snippet = transcript[max(0, match.start()-100):match.end()+100]
        sentiment = sia.polarity_scores(snippet)
        is_positive = sentiment['compound'] > 0.3 or any(word in snippet.lower() for word in positive_keywords)

        st.session_state.total_mentions += 1
        if is_positive:
            st.session_state.positive_mentions += 1
        st.markdown(f"**🎙️ Snippet:** {snippet}\n\n**📈 Sentiment:** {'Positive' if is_positive else 'Neutral/Negative'}")

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

def generate_detailed_summary(text):
    touchdowns = len(re.findall(r"touchdown", text, re.IGNORECASE))
    field_goals = len(re.findall(r"field goal", text, re.IGNORECASE))
    interceptions = len(re.findall(r"interception", text, re.IGNORECASE))
    big_plays = touchdowns + field_goals + interceptions

    player_impact = re.findall(rf"{player_name}.*?(touchdown|pass|score|throw|run|drive|highlight)", text, re.IGNORECASE)
    player_praise = len(player_impact)

    summary = "This game showcased "
    if big_plays > 0:
        summary += f"a high level of intensity with {touchdowns} touchdowns, {field_goals} field goals, and {interceptions} interceptions. "
    else:
        summary += "a more defensive battle with fewer headline plays. "

    if player_praise > 0:
        summary += f"{player_name} had a strong presence throughout the game, drawing attention during key moments such as {player_praise} highlighted actions. "
    else:
        summary += f"{player_name} was mentioned but didn't dominate the commentary with standout plays. "

    summary += "Overall, the broadcast captured a dynamic matchup with momentum shifts and key contributions from notable players."
    return summary

if audio_file:
    st.audio(audio_file)
    st.info("Transcribing audio via Groq API... please wait.")
    file_extension = audio_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as f:
        f.write(audio_file.read())
        f.flush()
        transcript = transcribe_via_groq(f.name)
        os.remove(f.name)

    st.subheader("📝 Game Summary")
    summary = generate_detailed_summary(transcript)
    st.write(summary)

    analyze_transcript(transcript, player_name)
