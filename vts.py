import os
import base64
import streamlit as st
import torch
from faster_whisper import WhisperModel
from transformers import pipeline
import yt_dlp
from deep_translator import GoogleTranslator
import subprocess

# Ensure PyTorch is installed at runtime
try:
    import torch
except ModuleNotFoundError:
    subprocess.run(["pip", "install", "torch torchvision torchaudio -f https://download.pytorch.org/whl/cpu.html"])
    import torch


# Set Page Config
st.set_page_config(page_title="Vid2Text-AI | Video Transcript Summarizer", layout="wide")

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
            .title {{
                font-size: 40px;
                font-weight: bold;
                text-align: center;
                color: #1E90FF;
            }}
            .subtitle {{
                font-size: 22px;
                text-align: center;
                color: #808080;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call function to set background
set_background("background.jpeg")

# ---- UI Title & Description ----
st.markdown('<p class="title">Vid2Text-AI</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Video Transcript Summarizer</p>', unsafe_allow_html=True)

st.write("Enter a YouTube video link to extract and summarize the transcript.")

# ---- Language Selection ----
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

# ---- Video URL Input ----
video_url = st.text_input("🔗 Enter YouTube Video URL:")
process_button = st.button("▶ Process Video")  # Process Button

audio_path = "temp_audio.mp3"

if process_button and video_url:
    with st.spinner("🔊 Extracting audio from video..."):
        try:
            ydl_opts = {
                'format': 'bestaudio/best',
                'extract_audio': True,
                'audio_format': 'mp3',
                'outtmpl': audio_path,
                'quiet': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            st.success("✅ Audio extraction completed!")
        except Exception as e:
            st.error(f"❌ Error extracting audio: {e}")

    # ---- Step 2: Transcribe Audio (English Only) ----
    with st.spinner("📝 Transcribing audio in English..."):
        device = "cpu"  # Optimized for CPU
        model_size = "small"

        model = WhisperModel(model_size, device=device, compute_type="int8")
        segments, _ = model.transcribe(audio_path, beam_size=2, language="en")  # Force English for accuracy

        transcript_text = " ".join(segment.text for segment in segments).strip()
        if transcript_text:
            st.success("✅ Transcription completed!")
            st.text_area("📜 Full Transcript (English):", transcript_text, height=200)
        else:
            st.error("❌ No text transcribed from the audio.")

    # ---- Step 3: Translate Transcript ----
    with st.spinner(f"🌍 Translating transcript to {transcript_lang}..."):
        if transcript_text.strip():
            translated_transcript = GoogleTranslator(source="auto", target=languages[transcript_lang]).translate(transcript_text)
            st.success("✅ Transcript translation completed!")
            st.text_area(f"📜 Transcript in {transcript_lang}:", translated_transcript, height=200)
        else:
            st.error("❌ Error: No transcript available for translation.")
            translated_transcript = ""

    # ---- Step 4: Summarize Transcript ----
    with st.spinner("📄 Summarizing transcript..."):
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)  # Faster model

        def chunk_text(text, max_words=400):
            words = text.split()
            return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)] if words else []

        if transcript_text.strip():
            chunks = chunk_text(transcript_text, max_words=400)

            if not chunks:
                st.error("❌ Error: Failed to split transcript into chunks.")
                summary_text = ""
            else:
                summary_text = ""
                try:
                    for chunk in chunks:
                        summary_result = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
                        summary_text += summary_result[0]["summary_text"] + " "

                    st.success("✅ Summary generated!")
                    st.text_area("📌 Summary (English):", summary_text, height=150)

                except Exception as e:
                    st.error(f"❌ Error summarizing: {e}")
                    summary_text = ""

    # ---- Step 5: Translate Summary ----
    with st.spinner(f"🌍 Translating summary to {summary_lang}..."):
        if summary_text.strip():
            translated_summary = GoogleTranslator(source="auto", target=languages[summary_lang]).translate(summary_text)
            st.success("✅ Summary translation completed!")
            st.text_area(f"📌 Summary in {summary_lang}:", translated_summary, height=150)
        else:
            st.error("❌ Error: No summary available for translation.")

    # ---- Clean Up ----
    os.remove(audio_path)
