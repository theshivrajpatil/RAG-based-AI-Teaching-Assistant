RAG AI Teaching Assistant
A retrieval-augmented generation system that helps students find specific topics within video course content. Uses embeddings and semantic search to locate relevant video segments and timestamps.
Overview
This tool transcribes video lectures, chunks them into searchable segments, and uses RAG to answer questions like "where is heading and paragraph taught?" by pointing to specific videos and timestamps.
Requirements

Python 3.x
ffmpeg
Ollama (running locally with bge-m3 and llama3.2 models)
whisper, pandas, scikit-learn, joblib, requests

Setup

Install dependencies:

bashpip install whisper pandas scikit-learn joblib requests

Make sure Ollama is running on localhost:11434 with these models:

bashollama pull bge-m3
ollama pull llama3.2

Create required folders:

bashmkdir videos audios jsons newjsons
Usage
Step 1: Add Videos
Place your course videos in the videos folder.
Step 2: Extract Audio
bashpython video_to_mp3.py
Converts videos to mp3 files using ffmpeg.
Step 3: Transcribe Audio
bashpython mp3_to_json.py
Uses Whisper large-v2 to transcribe audio and create timestamped JSON files.
Step 4: Merge and Embed Chunks
bashpython merge_chunks.py
python preprocess_json.py
Groups transcription chunks and generates embeddings using bge-m3. Saves as embeddings.joblib.
Step 5: Query the System
bashpython process_incoming.py
Ask questions about course content. The system finds relevant video segments and responds with video numbers and timestamps.
How It Works

Videos are transcribed with timestamps
Transcripts are chunked and embedded using bge-m3
User questions are embedded and compared using cosine similarity
Top matching chunks are sent to llama3.2 with context
LLM generates a natural response with video references

Files

video_to_mp3.py - Audio extraction
mp3_to_json.py - Whisper transcription
merge_chunks.py - Chunk grouping (5 segments per chunk)
preprocess_json.py - Embedding generation
process_incoming.py - Query interface
embeddings.joblib - Stored embeddings database

Notes

Transcription uses Hindi language with English translation
Adjust chunk size by changing n in merge_chunks.py
Model can be switched in process_incoming.py (currently uses llama3.2)
Top 5 results are used for context generation