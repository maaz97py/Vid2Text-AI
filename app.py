import os
import base64
import streamlit as st
import torch
from faster_whisper import WhisperModel
from transformers import pipeline
import yt_dlp
from deep_translator import GoogleTranslator
import tempfile

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

set_background("background.jpeg")

st.title("🎥 Video Transcript Summarizer")
st.write("Enter a YouTube video link to extract and summarize the transcript.")

# ---- Sidebar Options ----
languages = {"English": "en", "Telugu": "te", "Hindi": "hi"}
transcript_lang = st.sidebar.selectbox("📜 Transcript Language", list(languages.keys()))
summary_lang = st.sidebar.selectbox("📄 Summary Language", list(languages.keys()))
summary_format = st.sidebar.selectbox("📌 Summary Format", ["Paragraph", "Bullet Points", "Key Highlights"])

video_url = st.text_input("🔗 Enter YouTube Video URL:")
process_button = st.button("▶ Process Video")

if process_button and video_url:
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = os.path.join(temp_dir, "audio.m4a")
        
        with st.spinner("🔊 Extracting audio from video..."):
            try:
                ydl_opts = {
                    'format': 'bestaudio[ext=m4a]',
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'm4a',
                    }],
                    'outtmpl': os.path.join(temp_dir, '%(id)s.%(ext)s'),
                    'quiet': True,
                    'nocheckcertificate': True,  # Bypass certificate issues
                    'geo_bypass': True,  # Bypass geographical restrictions
                    'http_headers': {  # Spoof as a real browser
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                        'Accept-Language': 'en-US,en;q=0.9',
                        'Referer': 'https://www.youtube.com/',
                    },
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(video_url, download=True)
                    downloaded_filename = os.path.join(temp_dir, f"{info['id']}.m4a")
                    if os.path.exists(downloaded_filename):
                        os.rename(downloaded_filename, audio_path)
                    else:
                        raise FileNotFoundError("Downloaded audio file not found.")
                
                st.success("✅ Audio extraction completed!")
            except Exception as e:
                st.error(f"❌ Error extracting audio: {e}")
                st.stop()

        # ---- Transcribe Audio ----
        with st.spinner("📝 Transcribing audio..."):
            model = WhisperModel("small", device="cpu", compute_type="int8")
            segments, _ = model.transcribe(audio_path, beam_size=2, language="en")
            transcript_text = " ".join(segment.text for segment in segments).strip()
            st.success("✅ Transcription completed!")
            st.text_area("📜 Full Transcript:", transcript_text, height=200)

        # ---- Translate Transcript ----
        with st.spinner(f"🌍 Translating transcript to {transcript_lang}..."):
            translated_transcript = GoogleTranslator(source="auto", target=languages[transcript_lang]).translate(transcript_text)
            st.success("✅ Translation completed!")
            st.text_area(f"📜 Transcript in {transcript_lang}:", translated_transcript, height=200)

        # ---- Summarize Transcript ----
        with st.spinner("📄 Summarizing transcript..."):
            summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
            chunks = [transcript_text[i:i+400] for i in range(0, len(transcript_text), 400)]
            summary_text = " ".join([summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]["summary_text"] for chunk in chunks])

            if summary_format == "Bullet Points":
                summary_text = "\n".join([f"- {s}" for s in summary_text.split(". ") if s])
            elif summary_format == "Key Highlights":
                summary_text = "\n".join([f"✔ {s}" for s in summary_text.split(". ")[:5]])

            st.success("✅ Summary generated!")
            st.text_area("📌 Summary:", summary_text, height=150)

        # ---- Translate Summary ----
        with st.spinner(f"🌍 Translating summary to {summary_lang}..."):
            translated_summary = GoogleTranslator(source="auto", target=languages[summary_lang]).translate(summary_text)
            st.success("✅ Summary translation completed!")
            st.text_area("📌 Summary in {summary_lang}:", translated_summary, height=150)
