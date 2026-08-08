# YouTube RAG Chatbot

An AI-powered Retrieval-Augmented Generation (RAG) application that lets users chat with any YouTube video. Simply paste a YouTube URL to generate AI-powered notes, ask questions, summarize videos, and interact with the content through natural language.

## Features

- Process any YouTube video using its URL
- AI-powered question answering
- Automatic video summarization
- Generate structured notes
- Multi-chat conversation history
- Search previous conversations
- Delete chat history
- Fast semantic search using FAISS
- Context-aware answers using Retrieval-Augmented Generation (RAG)

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| LLM | Ollama — `mistral:latest  ` |
| Embeddings | Ollama — `nomic-embed-text:latest` |
| Vector Database | FAISS |
| Framework | LangChain |
| Data Source | YouTube Transcript API, yt-dlp |

## Project Structure

```
YouTube-RAG-Chatbot/
├── app.py           # Streamlit UI
├── backend.py       # Video processing & FAISS creation
├── chat_engine.py   # RAG chain
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd RAG-based-AI-Teaching-Assistant
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install and pull Ollama models

Install [Ollama](https://ollama.com) and pull the required models:

```bash
ollama pull llama3.2:latest
ollama pull nomic-embed-text:latest
```

Make sure Ollama is running (default: `http://localhost:11434`).

Optional — set a custom Ollama URL in `.env`:

```
OLLAMA_BASE_URL=http://localhost:11434
```

### 3. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Usage

1. **Process a video** — Paste a YouTube URL in the sidebar and click **Process Video**.
2. **Ask questions** — Type questions in the chat input about the video content.
3. **Summarize** — Click **Summarize** in the sidebar for an AI-generated summary.
4. **Generate notes** — Click **Notes** for structured study notes.
5. **Manage history** — Previous chats appear in the sidebar. Search by title or delete old chats.

## How It Works

1. The app extracts the video transcript via the YouTube Transcript API (with yt-dlp fallback).
2. The transcript is split into chunks and embedded using Ollama `nomic-embed-text:latest`.
3. Embeddings are stored in a local FAISS vector index for fast semantic search.
4. When you ask a question, the most relevant chunks are retrieved and sent to Ollama `llama3.2:latest`.
5. The LLM generates a context-aware answer based on the retrieved transcript segments.

## Requirements

- Python 3.10+
- Ollama running locally with `llama3.2:latest` and `nomic-embed-text:latest`
- Internet connection for YouTube transcript fetching

> **Note:** If you previously indexed videos with a different embedding model, delete the `data/vectorstores/` folder and re-process those videos.

## License

MIT
