import streamlit as st
import os
import openai
import whisper
import tempfile
import subprocess

from dotenv import load_dotenv
from pytubefix import YouTube
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS  # or Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI

import ssl
ssl._create_default_https_context = ssl._create_stdlib_context

############################################
# Streamlit App
############################################
def main():
    st.title("YouTube Q&A")

    # 1) Get the OpenAI API key from the user
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if openai_api_key:
        openai.api_key = openai_api_key  # Set the key for openai

    # 2) Get YouTube link
    youtube_url = st.text_input("Enter the YouTube URL to process:")

    # 3) Initialize session state
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 4) Button to download and transcribe
    if youtube_url and openai_api_key:
        if st.button("Download & Transcribe with Whisper"):
            with st.spinner("Downloading video and transcribing..."):
                transcript = download_and_transcribe_youtube(youtube_url)
                if transcript is None:
                    st.error("Could not retrieve transcript. Check logs for details.")
                else:
                    st.success("Transcription completed!")
                    st.info("Sample of transcript:\n\n" + transcript[:200] + "...")

                    # 5) Build QA chain
                    st.session_state.qa_chain = build_qa_chain(transcript, openai_api_key)
                    st.success("Vector store + chain built! You can now ask questions below.")

    # 6) Ask questions once chain is built
    if st.session_state.qa_chain is not None:
        question = st.text_input("Ask a question about the video content:")
        if question:
            with st.spinner("Thinking..."):
                response = st.session_state.qa_chain(
                    {"question": question, "chat_history": st.session_state.chat_history}
                )
                answer = response["answer"]
                st.session_state.chat_history.append((question, answer))
                st.write(f"**Answer**: {answer}")

                # Optionally display sources
                if "source_documents" in response:
                    sources = [doc.metadata.get("source") for doc in response["source_documents"] if doc.metadata]
                    st.write("**Sources**:", sources)


def download_and_transcribe_youtube(youtube_url: str) -> str:
    """
    Downloads the YouTube video using pytube, extracts audio, 
    and uses OpenAI's Whisper to transcribe it.
    Returns the transcript as a string.
    """
    try:
        # 1) Download the video to a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            yt = YouTube(youtube_url)
            video = yt.streams.filter(only_audio=True).first()
            if video is None:
                raise ValueError("Could not find audio-only stream. Trying fallback...")

            audio_path = video.download(output_path=tmpdir)
            base, ext = os.path.splitext(audio_path)

            # 2) Convert to WAV (optional but often more reliable for whisper)
            wav_audio_path = os.path.join(tmpdir, "audio.wav")
            # use ffmpeg to convert 
            subprocess.run([
                "ffmpeg",
                "-i", audio_path,
                "-ar", "16000",  # sample rate for whisper
                "-ac", "1",      # mono channel
                wav_audio_path
            ], capture_output=True, text=True, check=True)

            # 3) Load Whisper model (choose the size, e.g. "base", "small", "medium", etc.)
            #    Larger models are more accurate but slower.
            model = whisper.load_model("base")

            # 4) Transcribe the audio
            result = model.transcribe(wav_audio_path)
            transcript = result["text"]

            return transcript.strip() if transcript else None

    except Exception as e:
        print("Error in download_and_transcribe_youtube:", e)
        return None


def build_qa_chain(transcript: str, openai_api_key: str):
    """
    Build a retrieval-based QA chain using:
      1) Embedding the transcript
      2) Storing and creating a retriever
      3) Creating a ConversationalRetrievalChain
    """
    # 1) Split text
    text_splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(transcript)

    # 2) Create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # 3) Build a vector store (FAISS, Chroma, etc.). Here we use FAISS in-memory
    vectorstore = FAISS.from_texts(chunks, embeddings)

    # 4) Create the retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # 5) Create the LLM
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

    # 6) Build the chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain


if __name__ == "__main__":
    main()
