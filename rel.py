import streamlit as st
import subprocess
import tempfile
import os
import json
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BACKEND = st.sidebar.selectbox("Transcription Backend", ["whisper", "faster_whisper"])
MODEL_NAME = st.sidebar.selectbox("Model Size", ["tiny", "base", "small", "medium", "large"])
st.title("🎙️ Instagram Reel Transcriber")

# ─────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────
url = st.text_input("Paste Instagram Reel URL")
start_button = st.button("Transcribe Reel")

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def download_video(url, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp",
        "-f", "mp4",
        "--no-playlist",
        "--restrict-filenames",
        "--write-info-json",
        "-o", str(out_dir / "%(uploader)s_%(id)s.%(ext)s"),
        url,
    ]
    subprocess.run(cmd, check=True)
    mp4s = sorted(out_dir.glob("*_*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not mp4s:
        raise RuntimeError("Download failed: no MP4 found.")
    mp4 = mp4s[0]
    info_path = mp4.with_suffix(".info.json")
    meta = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    return mp4, meta

def extract_audio(video_path, wav_path):
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        str(wav_path),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def transcribe_whisper(audio_path):
    import whisper
    model = whisper.load_model(MODEL_NAME)
    result = model.transcribe(str(audio_path))
    return {
        "text": result.get("text", "").strip(),
        "segments": [
            {
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "text": seg.get("text", "").strip(),
            }
            for seg in result.get("segments", [])
        ],
        "language": result.get("language")
    }

def transcribe_faster_whisper(audio_path):
    from faster_whisper import WhisperModel
    model = WhisperModel(MODEL_NAME, compute_type="int8")
    segments, info = model.transcribe(str(audio_path))
    segs = []
    text_concat = []
    for s in segments:
        segs.append({"start": s.start, "end": s.end, "text": s.text.strip()})
        text_concat.append(s.text.strip())
    return {
        "text": " ".join(text_concat).strip(),
        "segments": segs,
        "language": getattr(info, "language", None),
    }

def format_srt_time(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def generate_srt(transcript):
    lines = []
    for i, seg in enumerate(transcript["segments"], 1):
        lines.append(str(i))
        lines.append(f"{format_srt_time(seg['start'])} --> {format_srt_time(seg['end'])}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)

# ─────────────────────────────────────────────
# ACTION
# ─────────────────────────────────────────────
if start_button and url:
    with st.spinner("Downloading Reel..."):
        with tempfile.TemporaryDirectory() as tmpd:
            tmpdir = Path(tmpd)
            try:
                mp4, meta = download_video(url, tmpdir)
                audio = tmpdir / "audio.wav"
                extract_audio(mp4, audio)

                with st.spinner("Transcribing audio..."):
                    if BACKEND == "faster_whisper":
                        transcript = transcribe_faster_whisper(audio)
                    else:
                        transcript = transcribe_whisper(audio)

                st.success("✅ Transcription complete!")
                st.markdown("### 📝 Transcript")
                st.text_area("Full Text", transcript["text"], height=300)

                srt_data = generate_srt(transcript)
                st.download_button("📥 Download .srt", srt_data, file_name="transcript.srt")
                st.download_button("📥 Download .txt", transcript["text"], file_name="transcript.txt")

            except Exception as e:
                st.error(f"❌ Error: {e}")
