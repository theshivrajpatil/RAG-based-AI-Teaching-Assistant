"""Streamlit UI for YouTube RAG Chatbot."""

import json
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from backend import extract_video_id, list_processed_videos, process_video
from chat_engine import ask_question, generate_notes, summarize_video

CHATS_DIR = Path("data/chats")
CHATS_DIR.mkdir(parents=True, exist_ok=True)


def load_chats() -> list[dict]:
    chats = []
    for chat_file in CHATS_DIR.glob("*.json"):
        with open(chat_file) as f:
            chats.append(json.load(f))
    return sorted(chats, key=lambda c: c.get("updated_at", ""), reverse=True)


def save_chat(chat: dict) -> None:
    chat_path = CHATS_DIR / f"{chat['id']}.json"
    with open(chat_path, "w") as f:
        json.dump(chat, f, indent=2)


def delete_chat(chat_id: str) -> None:
    chat_path = CHATS_DIR / f"{chat_id}.json"
    if chat_path.exists():
        chat_path.unlink()


def get_chat(chat_id: str) -> dict | None:
    chat_path = CHATS_DIR / f"{chat_id}.json"
    if chat_path.exists():
        with open(chat_path) as f:
            return json.load(f)
    return None


def init_session_state() -> None:
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "video_id" not in st.session_state:
        st.session_state.video_id = None
    if "video_title" not in st.session_state:
        st.session_state.video_title = None


def start_new_chat(video_id: str, title: str) -> str:
    chat_id = str(uuid.uuid4())[:8]
    chat = {
        "id": chat_id,
        "video_id": video_id,
        "title": title,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "messages": [],
    }
    save_chat(chat)
    st.session_state.current_chat_id = chat_id
    st.session_state.messages = []
    st.session_state.video_id = video_id
    st.session_state.video_title = title
    return chat_id


def load_existing_chat(chat: dict) -> None:
    st.session_state.current_chat_id = chat["id"]
    st.session_state.messages = chat.get("messages", [])
    st.session_state.video_id = chat["video_id"]
    st.session_state.video_title = chat.get("title", chat["video_id"])


def append_message(role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})
    if st.session_state.current_chat_id:
        chat = get_chat(st.session_state.current_chat_id)
        if chat:
            chat["messages"] = st.session_state.messages
            chat["updated_at"] = datetime.now().isoformat()
            save_chat(chat)


def render_sidebar() -> None:
    with st.sidebar:
        st.title("YouTube RAG Chatbot")
        st.markdown("Chat with any YouTube video using AI")

        st.divider()
        st.subheader("New Video")

        url = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...")
        if st.button("Process Video", type="primary", use_container_width=True):
            if url:
                with st.spinner("Fetching transcript and building index..."):
                    try:
                        metadata = process_video(url)
                        start_new_chat(metadata["video_id"], metadata["title"])
                        st.success(f"Ready: {metadata['title']}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Enter a YouTube URL first.")

        st.divider()
        st.subheader("Quick Actions")
        if st.session_state.video_id:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Summarize", use_container_width=True):
                    with st.spinner("Summarizing..."):
                        try:
                            summary = summarize_video(st.session_state.video_id)
                            append_message("assistant", f"**Video Summary**\n\n{summary}")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
            with col2:
                if st.button("Notes", use_container_width=True):
                    with st.spinner("Generating notes..."):
                        try:
                            notes = generate_notes(st.session_state.video_id)
                            append_message("assistant", f"**Structured Notes**\n\n{notes}")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
        else:
            st.caption("Process a video to enable quick actions.")

        st.divider()
        st.subheader("Chat History")

        search = st.text_input("Search chats", placeholder="Search by title...")
        chats = load_chats()
        if search:
            chats = [c for c in chats if search.lower() in c.get("title", "").lower()]

        for chat in chats[:20]:
            col1, col2 = st.columns([5, 1])
            with col1:
                label = chat.get("title", chat["video_id"])[:30]
                if st.button(label, key=f"load_{chat['id']}", use_container_width=True):
                    load_existing_chat(chat)
                    st.rerun()
            with col2:
                if st.button("🗑", key=f"del_{chat['id']}"):
                    delete_chat(chat["id"])
                    if st.session_state.current_chat_id == chat["id"]:
                        st.session_state.current_chat_id = None
                        st.session_state.messages = []
                        st.session_state.video_id = None
                    st.rerun()

        st.divider()
        processed = list_processed_videos()
        if processed:
            st.caption(f"{len(processed)} video(s) indexed")


def render_chat() -> None:
    if st.session_state.video_title:
        st.header(st.session_state.video_title)
        st.caption(f"Video ID: {st.session_state.video_id}")
    else:
        st.header("YouTube RAG Chatbot")
        st.markdown(
            "Paste a **YouTube URL** in the sidebar to get started. "
            "Ask questions, summarize videos, and generate notes — all powered by RAG."
        )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about the video..."):
        if not st.session_state.video_id:
            st.warning("Process a YouTube video first using the sidebar.")
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = ask_question(st.session_state.video_id, prompt)
                    st.markdown(answer)
                    append_message("assistant", answer)
                except Exception as e:
                    st.error(str(e))


def main() -> None:
    st.set_page_config(
        page_title="YouTube RAG Chatbot",
        page_icon="🎥",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
