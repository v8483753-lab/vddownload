#!/usr/bin/env python3
# Filename: transcribe_reels.py

import os
import sys
import subprocess
import shlex
from pathlib import Path
import argparse
import tempfile
import json

# Choose backend: "whisper" (openai-whisper) or "faster_whisper"
BACKEND = os.getenv("TRANSCRIPT_BACKEND", "whisper")  # or "faster_whisper"
MODEL_NAME = os.getenv("WHISPER_MODEL", "base")       # tiny, base, small, medium, large

def ensure_deps():
    # Check ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        print("ffmpeg not found on PATH. Please install ffmpeg.", file=sys.stderr)
        sys.exit(1)
    # Check yt-dlp
    try:
        subprocess.run(["yt-dlp", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        print("yt-dlp not found. Install with: pip install yt-dlp", file=sys.stderr)
        sys.exit(1)
    # Check whisper backend
    try:
        if BACKEND == "faster_whisper":
            import faster_whisper  # noqa: F401
        else:
            import whisper  # noqa: F401
    except Exception:
        if BACKEND == "faster_whisper":
            print("faster-whisper not installed. Install with: pip install faster-whisper", file=sys.stderr)
        else:
            print("openai-whisper not installed. Install with: pip install openai-whisper torch", file=sys.stderr)
        sys.exit(1)

def read_urls(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

def safe_name(title):
    keep = "-._() []"
    return "".join(c if c.isalnum() or c in keep else "_" for c in title)[:120]

def download_video(url, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Template will create a safe filename; capture info JSON to get title/id
    cmd = [
        "yt-dlp",
        "-f", "mp4",
        "--no-playlist",
        "--restrict-filenames",
        "--write-info-json",
        "-o", str(out_dir / "%(uploader)s_%(id)s.%(ext)s"),
        url,
    ]
    print(f"[download] {url}")
    subprocess.run(cmd, check=True)
    # Find the downloaded file
    mp4s = sorted(out_dir.glob("*_*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not mp4s:
        raise RuntimeError("Download failed: no MP4 found.")
    mp4 = mp4s[0]
    info_path = mp4.with_suffix(".info.json")
    meta = {}
    if info_path.exists():
        meta = json.loads(info_path.read_text(encoding="utf-8"))
    return mp4, meta

def extract_audio(video_path, wav_path):
    wav_path.parent.mkdir(parents=True, exist_ok=True)
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
    # result has 'text' and 'segments'
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
    model = WhisperModel(MODEL_NAME, compute_type="int8" if os.name != "posix" else "int8")
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

def write_outputs(base_path, transcript):
    txt_path = base_path.with_suffix(".txt")
    srt_path = base_path.with_suffix(".srt")
    # TXT
    txt_path.write_text(transcript["text"] + "\n", encoding="utf-8")
    # SRT
    lines = []
    for i, seg in enumerate(transcript["segments"], 1):
        lines.append(str(i))
        lines.append(f"{format_srt_time(seg['start'])} --> {format_srt_time(seg['end'])}")
        lines.append(seg["text"])
        lines.append("")  # blank line
    srt_path.write_text("\n".join(lines), encoding="utf-8")
    return txt_path, srt_path

def main():
    parser = argparse.ArgumentParser(description="Transcribe public Instagram Reels to text/SRT.")
    parser.add_argument("--urls", default="reels.txt", help="Path to file containing Reel URLs (one per line).")
    parser.add_argument("--out", default="transcripts", help="Output folder.")
    args = parser.parse_args()

    ensure_deps()
    urls = read_urls(args.urls)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for url in urls:
        try:
            with tempfile.TemporaryDirectory() as tmpd:
                tmpdir = Path(tmpd)
                mp4, meta = download_video(url, tmpdir)
                title = meta.get("title") or mp4.stem
                base = safe_name(title)
                audio = tmpdir / "audio.wav"
                extract_audio(mp4, audio)

                if BACKEND == "faster_whisper":
                    transcript = transcribe_faster_whisper(audio)
                else:
                    transcript = transcribe_whisper(audio)

                # Build base output path using uploader+id if present
                uid = meta.get("id") or mp4.stem
                uploader = meta.get("uploader") or "instagram"
                base_out = out_dir / f"{uploader}_{uid}"

                txt_path, srt_path = write_outputs(base_out, transcript)
                print(f"[done] {url}\n  TXT: {txt_path}\n  SRT: {srt_path}\n")
        except Exception as e:
            print(f"[error] {url} -> {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
