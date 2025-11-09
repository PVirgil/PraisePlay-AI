# 🏈 PraisePlay AI

**PraisePlay AI** is an intelligent sports audio analyzer that detects and counts positive mentions of a specific player during a game broadcast. Designed for sports fans, analysts, and athlete agents, it transcribes audio commentary and generates simple yet insightful game summaries — all powered by the Groq API and Streamlit.

## 🔍 What It Does

- 🎧 Upload MP3 or WAV audio clips of game commentary
- 📋 Automatically transcribes the audio using Groq's Whisper model
- 🗣 Detects how many times your chosen player is mentioned
- 📈 Identifies which of those mentions are **positive**
- 📝 Produces a simple highlight-style game summary (no external AI needed!)

## 🚀 Live Demo

Deployed on [Streamlit Cloud](https://praiseplay-ai.streamlit.app) 

## 🎯 Use Cases

- Athlete brand monitoring
- Agent/player reputation tracking
- Fan media analysis
- Post-game promotional content
- Sports data journalism

## 🧰 Tech Stack

- **Python 3.9+**
- **Streamlit** for the front-end UI
- **Groq Whisper API** for fast transcription
- **NLTK (VADER)** for sentiment analysis
- **Regex** and keyword heuristics for game insight

## 📁 File Structure

```
├── streamlit_app.py         # Main Streamlit app logic
├── requirements.txt         # Python dependencies
└── .streamlit
    └── secrets.toml         # Store your API keys securely here
```

## ⚙️ How It Works

1. Upload an audio file from a game broadcast
2. App sends it to Groq’s `whisper-large-v3-turbo` for transcription
3. It parses the text and identifies:
   - Total mentions of your selected player
   - How many are positive (based on sentiment + praise keywords)
   - Summary of gameplay based on detected terms like touchdowns/interceptions

## 📊 Example Output

- Mentions of *Patrick Mahomes*: **9**
- Positive Mentions: **6**
- Summary:
  > The game featured 3 touchdowns, 1 field goal, and 2 interceptions. Patrick Mahomes had a strong presence with multiple impactful plays.

---

**PraisePlay AI** — Because praise matters. Now you can measure it.
