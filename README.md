# RAG AI Teaching Assistant

A retrieval-augmented generation system that helps students find specific topics within video course content. Uses embeddings and semantic search to locate relevant video segments and timestamps.

## Overview

This tool transcribes video lectures, chunks them into searchable segments, and uses RAG to answer questions like "where is heading and paragraph taught?" by pointing to specific videos and timestamps.

## Requirements

- Python 3.x
- ffmpeg
- Ollama (running locally with bge-m3 and llama3.2 models)
- whisper, pandas, scikit-learn, joblib, requests

## Setup

1. Install dependencies:
```bash
pip install whisper pandas scikit-learn joblib requests
```

2. Make sure Ollama is running on localhost:11434 with these models:
```bash
ollama pull bge-m3
ollama pull llama3.2
```

3. Create required folders:
```bash
mkdir videos audios jsons newjsons
```

## Usage

### Step 1: Add Videos
Place your course videos in the `videos` folder.

### Step 2: Extract Audio
```bash
python video_to_mp3.py
```
Converts videos to mp3 files using ffmpeg.

### Step 3: Transcribe Audio
```bash
python mp3_to_json.py
```
Uses Whisper large-v2 to transcribe audio and create timestamped JSON files.

### Step 4: Merge and Embed Chunks
```bash
python merge_chunks.py
python preprocess_json.py
```
Groups transcription chunks and generates embeddings using bge-m3. Saves as `embeddings.joblib`.

### Step 5: Query the System
```bash
python process_incoming.py
```
Ask questions about course content. The system finds relevant video segments and responds with video numbers and timestamps.

## How It Works

1. Videos are transcribed with timestamps
2. Transcripts are chunked and embedded using bge-m3
3. User questions are embedded and compared using cosine similarity
4. Top matching chunks are sent to llama3.2 with context
5. LLM generates a natural response with video references

## Files

- `video_to_mp3.py` - Audio extraction
- `mp3_to_json.py` - Whisper transcription
- `merge_chunks.py` - Chunk grouping (5 segments per chunk)
- `preprocess_json.py` - Embedding generation
- `process_incoming.py` - Query interface
- `embeddings.joblib` - Stored embeddings database

## Notes

- Transcription uses Hindi language with English translation
- Adjust chunk size by changing `n` in merge_chunks.py
- Model can be switched in process_incoming.py (currently uses llama3.2)
- Top 5 results are used for context generation
