"""Video processing and FAISS vector store creation for YouTube RAG Chatbot."""

import json
import os
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

DATA_DIR = Path("data")
VECTOR_DIR = DATA_DIR / "vectorstores"
METADATA_DIR = DATA_DIR / "metadata"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "nomic-embed-text:latest"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats."""
    url = url.strip()
    parsed = urlparse(url)

    if parsed.hostname in ("youtu.be", "www.youtu.be"):
        return parsed.path.lstrip("/").split("/")[0]

    if parsed.hostname in ("youtube.com", "www.youtube.com", "m.youtube.com"):
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]
        match = re.match(r"^/(embed|v|shorts)/([^/?]+)", parsed.path)
        if match:
            return match.group(2)

    if re.fullmatch(r"[\w-]{11}", url):
        return url

    raise ValueError(f"Could not extract video ID from URL: {url}")


def fetch_transcript(video_id: str) -> tuple[str, list[dict]]:
    """Fetch transcript text and timestamped segments for a YouTube video."""
    api = YouTubeTranscriptApi()
    try:
        transcript_list = api.list(video_id)
        try:
            transcript = transcript_list.find_transcript(["en", "en-US", "en-GB"])
        except NoTranscriptFound:
            transcript = next(iter(transcript_list))
            if transcript.is_translatable:
                transcript = transcript.translate("en")
        fetched = transcript.fetch()
        segments = [{"start": s.start, "text": s.text} for s in fetched]
    except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable):
        segments = _fetch_transcript_ytdlp(video_id)

    if not segments:
        raise ValueError("No transcript available for this video.")

    lines = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if text:
            start = seg.get("start", 0)
            lines.append(f"[{format_timestamp(start)}] {text}")

    return "\n".join(lines), segments


def _fetch_transcript_ytdlp(video_id: str) -> list[dict]:
    """Fallback transcript fetch using yt-dlp when the API has no captions."""
    import yt_dlp

    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    subtitles = info.get("subtitles") or info.get("automatic_captions") or {}
    en_subs = subtitles.get("en") or subtitles.get("en-US") or subtitles.get("en-GB")
    if not en_subs:
        raise ValueError("No English subtitles found for this video.")

    sub_url = en_subs[0]["url"]
    import urllib.request

    with urllib.request.urlopen(sub_url) as response:
        content = response.read().decode("utf-8")

    return _parse_vtt(content)


def _parse_vtt(vtt_content: str) -> list[dict]:
    """Parse WebVTT subtitle content into segment dicts."""
    segments = []
    lines = vtt_content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "-->" in line:
            start_str = line.split("-->")[0].strip()
            start = _vtt_time_to_seconds(start_str)
            i += 1
            text_parts = []
            while i < len(lines) and lines[i].strip() and "-->" not in lines[i]:
                text_parts.append(re.sub(r"<[^>]+>", "", lines[i].strip()))
                i += 1
            text = " ".join(text_parts).strip()
            if text:
                segments.append({"start": start, "text": text})
        else:
            i += 1
    return segments


def _vtt_time_to_seconds(time_str: str) -> float:
    parts = time_str.replace(",", ".").split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(parts[0])


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)


def process_video(url: str) -> dict:
    """
    Process a YouTube video: fetch transcript, chunk, embed, and store in FAISS.
    Returns metadata dict with video_id, title, and transcript preview.
    """
    video_id = extract_video_id(url)
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    store_path = VECTOR_DIR / video_id
    meta_path = METADATA_DIR / f"{video_id}.json"

    if store_path.exists() and meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)

    transcript_text, segments = fetch_transcript(video_id)
    title = _fetch_video_title(video_id)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(transcript_text)

    embeddings = get_embeddings()
    vectorstore = FAISS.from_texts(
        chunks,
        embeddings,
        metadatas=[{"video_id": video_id, "chunk_index": i} for i in range(len(chunks))],
    )
    vectorstore.save_local(str(store_path))

    metadata = {
        "video_id": video_id,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "title": title,
        "transcript_preview": transcript_text[:500] + ("..." if len(transcript_text) > 500 else ""),
        "segment_count": len(segments),
        "chunk_count": len(chunks),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def _fetch_video_title(video_id: str) -> str:
    """Fetch video title via yt-dlp."""
    import yt_dlp

    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {"skip_download": True, "quiet": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return info.get("title", video_id)


def load_vectorstore(video_id: str) -> FAISS:
    """Load an existing FAISS vector store for a video."""
    store_path = VECTOR_DIR / video_id
    if not store_path.exists():
        raise FileNotFoundError(f"No vector store found for video {video_id}. Process it first.")
    return FAISS.load_local(str(store_path), get_embeddings(), allow_dangerous_deserialization=True)


def list_processed_videos() -> list[dict]:
    """Return metadata for all processed videos."""
    if not METADATA_DIR.exists():
        return []
    videos = []
    for meta_file in METADATA_DIR.glob("*.json"):
        with open(meta_file) as f:
            videos.append(json.load(f))
    return sorted(videos, key=lambda v: v.get("title", ""))


def get_full_transcript(video_id: str) -> str:
    """Reconstruct full transcript text from all FAISS chunks."""
    vectorstore = load_vectorstore(video_id)
    docs = list(vectorstore.docstore._dict.values())
    docs.sort(key=lambda d: d.metadata.get("chunk_index", 0))
    return "\n".join(doc.page_content for doc in docs)
