"""RAG chat engine powered by Ollama and LangChain."""

import os
from typing import Generator

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from backend import get_full_transcript, load_vectorstore

LLM_MODEL = "llama3.2:latest"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
RETRIEVAL_K = 5

QA_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant that answers questions about YouTube video content. "
        "Use ONLY the provided transcript context to answer. "
        "If the answer is not in the context, say you cannot find it in the video. "
        "When referencing specific moments, include timestamps if available in the context.",
    ),
    ("human", "Context from the video transcript:\n\n{context}\n\nQuestion: {input}"),
])

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert at summarizing video content. "
        "Create a clear, well-structured summary of the video based on the transcript.",
    ),
    (
        "human",
        "Summarize this YouTube video transcript. Include:\n"
        "1. A brief overview (2-3 sentences)\n"
        "2. Key points (bullet list)\n"
        "3. Main takeaways\n\n"
        "Transcript:\n\n{context}",
    ),
])

NOTES_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert note-taker. Generate structured study notes from video transcripts.",
    ),
    (
        "human",
        "Create detailed structured notes from this YouTube video transcript. "
        "Organize with headings, subheadings, bullet points, and key definitions. "
        "Make the notes easy to review and study from.\n\n"
        "Transcript:\n\n{context}",
    ),
])


def _get_llm() -> ChatOllama:
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.3,
    )


def _build_rag_chain(video_id: str):
    vectorstore = load_vectorstore(video_id)
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    llm = _get_llm()
    doc_chain = create_stuff_documents_chain(llm, QA_PROMPT)
    return create_retrieval_chain(retriever, doc_chain)


def ask_question(video_id: str, question: str) -> str:
    """Answer a question about a video using RAG."""
    chain = _build_rag_chain(video_id)
    result = chain.invoke({"input": question})
    return result["answer"]


def ask_question_stream(video_id: str, question: str) -> Generator[str, None, None]:
    """Stream an answer to a question (non-streaming fallback for compatibility)."""
    answer = ask_question(video_id, question)
    yield answer


def summarize_video(video_id: str) -> str:
    """Generate a summary of the video transcript."""
    context = get_full_transcript(video_id)
    llm = _get_llm()
    chain = SUMMARY_PROMPT | llm
    response = chain.invoke({"context": context})
    return response.content


def generate_notes(video_id: str) -> str:
    """Generate structured study notes from the video."""
    context = get_full_transcript(video_id)
    llm = _get_llm()
    chain = NOTES_PROMPT | llm
    response = chain.invoke({"context": context})
    return response.content
