import os
import base64
import streamlit as st
import torch
from faster_whisper import WhisperModel
from transformers import pipeline
import yt_dlp
from deep_translator import GoogleTranslator
import tempfile

from yt_dlp import YoutubeDL

ydl_opts = {
    'format': 'bestaudio/best',
    'noplaylist': True,
    'quiet': True,
    'http_headers': {
        'User-Agent': 'Mozilla/5.0'
    }
}

with YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info("VIDEO_URL", download=False)
    audio_url = info['url']


# ---- Set Page Config ----
st.set_page_config(page_title="Video Transcript Summarizer", layout="wide")

# ---- Function to Set Background Image ----
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded_image = base64.b64encode(image.read()).decode()
    
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-image: url("data:image/jpeg;base64,{encoded_image}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call function to set background
set_background("background.jpeg")

# ---- UI Title & Description ----
st.title("🎥 Video Transcript Summarizer")

st.write("Enter a YouTube video link to extract and summarize the transcript.")

# ---- Sidebar: Language & Summary Format Selection ----
st.sidebar.header("🌍 Language Options")
languages = {
    "English": "en",
    "Telugu": "te",
    "Hindi": "hi",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Chinese": "zh"
}
transcript_lang = st.sidebar.selectbox("📜 Select Transcript Language", list(languages.keys()))
summary_lang = st.sidebar.selectbox("📄 Select Summary Language", list(languages.keys()))

st.sidebar.header("📝 Summary Format")
summary_format = st.sidebar.selectbox("📌 Choose Summary Format", ["Paragraph", "Bullet Points", "Key Highlights"])

# ---- Video URL Input ----
video_url = st.text_input("🔗 Enter YouTube Video URL:")
process_button = st.button("▶ Process Video")  # Process Button

if process_button and video_url:
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_filename = "temp_audio.wav"
        audio_path = os.path.join(temp_dir, audio_filename)
        
        with st.spinner("🔊 Extracting audio from video..."):
            try:
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'wav',
                        'preferredquality': '192',
                    }],
                    'outtmpl': os.path.join(temp_dir, '%(id)s.%(ext)s'),
                    'quiet': True,
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(video_url, download=True)
                    downloaded_filename = os.path.join(temp_dir, f"{info['id']}.wav")
                    if os.path.exists(downloaded_filename):
                        os.rename(downloaded_filename, audio_path)
                    else:
                        raise FileNotFoundError("Downloaded audio file not found.")
                st.success("✅ Audio extraction completed!")
            except Exception as e:
                st.error(f"❌ Error extracting audio: {e}")
                st.stop()

        # ---- Step 2: Transcribe Audio ----
        with st.spinner("📝 Transcribing audio in English..."):
            device = "cpu"  # Optimized for CPU
            model_size = "small"

            model = WhisperModel(model_size, device=device, compute_type="int8")
            
            if os.path.exists(audio_path):
                segments, _ = model.transcribe(audio_path, beam_size=2, language="en")
                transcript_text = " ".join(segment.text for segment in segments).strip()
                
                if transcript_text:
                    st.success("✅ Transcription completed!")
                    st.text_area("📜 Full Transcript (English):", transcript_text, height=200)
                else:
                    st.error("❌ No text transcribed from the audio.")
                    st.stop()
            else:
                st.error("❌ Audio file not found.")
                st.stop()

        # ---- Step 3: Translate Transcript ----
        with st.spinner(f"🌍 Translating transcript to {transcript_lang}..."):
            translated_transcript = GoogleTranslator(source="auto", target=languages[transcript_lang]).translate(transcript_text)
            st.success("✅ Transcript translation completed!")
            st.text_area(f"📜 Transcript in {transcript_lang}:", translated_transcript, height=200)

        # ---- Step 4: Summarize Transcript ----
        with st.spinner("📄 Summarizing transcript..."):
            summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)

            def chunk_text(text, max_words=400):
                words = text.split()
                return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)] if words else []

            chunks = chunk_text(transcript_text, max_words=400)
            summary_text = ""
            
            for chunk in chunks:
                summary_result = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
                summary_text += summary_result[0]["summary_text"] + " "

            if summary_format == "Bullet Points":
                summary_text = "\n".join([f"- {sentence}" for sentence in summary_text.split(". ") if sentence])
            elif summary_format == "Key Highlights":
                summary_text = "\n".join([f"✔ {sentence}" for sentence in summary_text.split(". ")[:5]])

            st.success("✅ Summary generated!")
            st.text_area(f"📌 Summary ({summary_format}):", summary_text, height=150)

        # ---- Step 5: Translate Summary ----
        with st.spinner(f"🌍 Translating summary to {summary_lang}..."):
            translated_summary = GoogleTranslator(source="auto", target=languages[summary_lang]).translate(summary_text)
            st.success("✅ Summary translation completed!")
            st.text_area(f"📌 Summary in {summary_lang}:", translated_summary, height=150)
