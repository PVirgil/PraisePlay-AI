# PraisePlay AI - Multi-Player Sports Highlight Tracker (Video Uploads)

# Requirements:
# - Python 3.9+
# - pip install openai-whisper nltk torch streamlit moviepy

import whisper
import streamlit as st
import os
import re
import tempfile
from moviepy.editor import VideoFileClip
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
st.title("🏈 PraisePlay AI: Sports Highlight Commentary Tracker")
st.markdown("Upload an NFL or sports video highlight and track how often any player's name is mentioned positively.")

# Player input
player_name = st.text_input("Enter Player's Name to Track (e.g., Patrick Mahomes)", "Patrick Mahomes")

# Video upload
video_file = st.file_uploader("Upload a video clip (MP4/MOV)", type=["mp4", "mov"])

# State variables
if "total_mentions" not in st.session_state:
    st.session_state.total_mentions = 0
if "positive_mentions" not in st.session_state:
    st.session_state.positive_mentions = 0

def analyze_transcript(transcript, player):
    pattern = re.compile(rf"\b({re.escape(player)})\b", re.IGNORECASE)
    mentions = pattern.finditer(transcript)

    for match in mentions:
        st.session_state.total_mentions += 1
        snippet = transcript[max(0, match.start()-100):match.end()+100]
        sentiment = sia.polarity_scores(snippet)
        if sentiment['compound'] > 0.3:
            st.session_state.positive_mentions += 1
        st.markdown(f"**Snippet:** {snippet}\n\n**Sentiment:** {'Positive' if sentiment['compound'] > 0.3 else 'Neutral/Negative'}")

    st.success(f"📊 Total Mentions: {st.session_state.total_mentions} | Positive Mentions: {st.session_state.positive_mentions}")

if video_file:
    st.video(video_file)
    st.info("Extracting audio and analyzing commentary...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
        tmp_video.write(video_file.read())
        tmp_video_path = tmp_video.name

    audio_path = tmp_video_path.replace(".mp4", ".wav")
    try:
        clip = VideoFileClip(tmp_video_path)
        clip.audio.write_audiofile(audio_path, verbose=False, logger=None)

        result = model.transcribe(audio_path)
        transcript = result["text"]

        st.subheader("Transcript")
        st.write(transcript)

        st.subheader(f"Analysis for: {player_name}")
        analyze_transcript(transcript, player_name)

    except Exception as e:
        st.error(f"Error processing video: {e}")

    finally:
        if os.path.exists(tmp_video_path):
            os.remove(tmp_video_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)

