import streamlit as st
import os
from dotenv import load_dotenv
from pathlib import Path

# ============================
# FORCE LOAD .env FILE (Critical Fix for Codespaces)
# ============================
env_path = Path('.') / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f".env file found and loaded from {env_path}")  # Debug line
else:
    print(".env file NOT found in project root!")

# NOW check the key
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    print(f"GOOGLE_API_KEY loaded successfully: {api_key[:6]}...{api_key[-4:]}")
else:
    st.error("🚨 GOOGLE_API_KEY is MISSING! Check your .env file and restart the app.")
    st.stop()  # Stop execution if no key

# ============================
# IMPORTS AFTER ENV IS LOADED
# ============================
import easyocr
from PIL import Image
import io
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from transformers import pipeline
import librosa
import json

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Temporary debug
print("GOOGLE API KEY:", os.getenv("GOOGLE_API_KEY")[:5] + "..." if os.getenv("GOOGLE_API_KEY") else "MISSING")

load_dotenv()

# ============================
# PHASE 3: RAG SETUP (Runs once)
# ============================
@st.cache_resource(show_spinner="Loading knowledge base and building vector store...")
def load_vector_store():
    from langchain_community.embeddings import HuggingFaceEmbeddings

    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Fast, local, free

        loader = DirectoryLoader("knowledge_base/", glob="*.txt", loader_cls=TextLoader)
        docs = loader.load()

        if not docs:
            st.warning("No documents found in 'knowledge_base/' folder. Add some .txt files.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store.as_retriever(search_kwargs={"k": 3})

    except Exception as e:
        st.error(f"Error loading knowledge base: {str(e)}")
        return None

retriever = load_vector_store()

# ============================
# APP UI STARTS HERE
# ============================
st.title("Math Mentor")

# Input mode selector
input_mode = st.selectbox("Choose input mode:", ["Text", "Image", "Audio"])

# Session state initialization
if "parsed_problem" not in st.session_state:
    st.session_state.parsed_problem = None
if "rag_context" not in st.session_state:
    st.session_state.rag_context = ""
if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

# ============================
# PARSER FUNCTION
# ============================
def parse_problem(text: str):
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            convert_system_message_to_human=True
        )

        prompt = PromptTemplate(
            input_variables=["text"],
            template="""
You are a precise math problem parser. Clean and structure the following math problem.

Problem: {text}

Respond ONLY with valid JSON in this exact format:
{{
  "problem_text": "cleaned and readable version of the problem",
  "topic": "algebra" or "probability" or "calculus" or "linear_algebra",
  "variables": ["x", "y", ...],
  "constraints": ["x > 0", "n is integer", ...],
  "needs_clarification": true or false
}}

Do not add explanations. Output only JSON.
"""
        )

        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(text=text)

        # Clean any extra text around JSON
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            response = response[json_start:json_end]

        return json.loads(response)
    except Exception as e:
        return {"error": "Parsing failed", "details": str(e)}

# ============================
# RETRIEVAL FUNCTION
# ============================
def retrieve_context(problem_text: str):
    if retriever is None:
        return {"context": "", "sources": [], "error": "Vector store not loaded."}
    
    try:
        relevant_docs = retriever.get_relevant_documents(problem_text)
        context = "\n\n---\n\n".join([doc.page_content.strip() for doc in relevant_docs])
        sources = [doc.metadata.get("source", "unknown.txt").split("/")[-1] for doc in relevant_docs]
        return {
            "context": context,
            "sources": sources,
            "num_docs": len(relevant_docs)
        }
    except Exception as e:
        return {"context": "", "sources": [], "error": str(e)}

# ============================
# TEXT INPUT MODE
# ============================
if input_mode == "Text":
    problem_text = st.text_area("Type your math problem:", height=150)
    if st.button("Parse and Proceed"):
        with st.spinner("Parsing with Gemini..."):
            st.session_state.parsed_problem = parse_problem(problem_text)
            st.json(st.session_state.parsed_problem)

        if st.session_state.parsed_problem.get("needs_clarification", False):
            st.warning("Problem is ambiguous! Please clarify and try again.")
        elif st.session_state.parsed_problem.get("error"):
            st.error("Parsing failed – cannot retrieve context.")
        else:
            with st.spinner("Retrieving relevant knowledge..."):
                rag_result = retrieve_context(st.session_state.parsed_problem["problem_text"])
                st.session_state.rag_context = rag_result.get("context", "")
                st.session_state.rag_sources = rag_result.get("sources", [])

            st.success("Relevant knowledge retrieved!")
            if st.session_state.rag_context:
                st.text_area("Retrieved Knowledge:", value=st.session_state.rag_context, height=250, disabled=True)
            else:
                st.info("No relevant knowledge found.")

# ============================
# IMAGE INPUT MODE
# ============================
elif input_mode == "Image":
    uploaded_image = st.file_uploader("Upload image (JPG/PNG):", type=["jpg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Extract Text (OCR)"):
            with st.spinner("Running OCR..."):
                reader = easyocr.Reader(['en'], gpu=False)
                result = reader.readtext(uploaded_image.getvalue())

                extracted_text = " ".join([text for _, text, _ in result])
                confidences = [conf for _, _, conf in result]
                avg_conf = sum(confidences) / len(confidences) if confidences else 0

                st.write(f"**Extracted Text** (Confidence: {avg_conf:.2f})")
                st.text_area("Edit/correct the text below:", value=extracted_text, height=150, key="edited_ocr")

                if avg_conf < 0.8:
                    st.warning("Low confidence OCR – please review and edit!")

        if st.button("Parse and Proceed"):
            problem_text = st.session_state.get("edited_ocr", extracted_text if 'extracted_text' in locals() else "")
            if not problem_text.strip():
                st.error("No text available to parse. Run OCR first.")
            else:
                with st.spinner("Parsing with Gemini..."):
                    st.session_state.parsed_problem = parse_problem(problem_text)
                    st.json(st.session_state.parsed_problem)

                if st.session_state.parsed_problem.get("error"):
                    st.error("Parsing failed.")
                elif st.session_state.parsed_problem.get("needs_clarification", False):
                    st.warning("Ambiguous problem.")
                else:
                    with st.spinner("Retrieving relevant knowledge..."):
                        rag_result = retrieve_context(st.session_state.parsed_problem["problem_text"])
                        st.session_state.rag_context = rag_result.get("context", "")
                        st.session_state.rag_sources = rag_result.get("sources", [])

                    st.success("Knowledge retrieved!")
                    if st.session_state.rag_context:
                        st.text_area("Retrieved Knowledge:", value=st.session_state.rag_context, height=250, disabled=True)

# ============================
# AUDIO INPUT MODE
# ============================
elif input_mode == "Audio":
    uploaded_audio = st.file_uploader("Upload audio (MP3/WAV):", type=["mp3", "wav"])
    if uploaded_audio:
        if st.button("Transcribe (ASR)"):
            with st.spinner("Transcribing audio with Whisper..."):
                audio_bytes = uploaded_audio.read()
                audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
                pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
                result = pipe({"raw": audio, "sampling_rate": sr})
                transcript = result["text"]

                st.write("**Transcript:**")
                st.text_area("Edit if needed:", value=transcript, height=150, key="edited_asr")

                if len(transcript.strip()) < 10:
                    st.warning("Short or unclear transcription – please check!")

        if st.button("Parse and Proceed"):
            problem_text = st.session_state.get("edited_asr", transcript if 'transcript' in locals() else "")
            if not problem_text.strip():
                st.error("No transcript available. Run transcription first.")
            else:
                with st.spinner("Parsing with Gemini..."):
                    st.session_state.parsed_problem = parse_problem(problem_text)
                    st.json(st.session_state.parsed_problem)

                if st.session_state.parsed_problem.get("error"):
                    st.error("Parsing failed.")
                elif st.session_state.parsed_problem.get("needs_clarification", False):
                    st.warning("Ambiguous problem.")
                else:
                    with st.spinner("Retrieving relevant knowledge..."):
                        rag_result = retrieve_context(st.session_state.parsed_problem["problem_text"])
                        st.session_state.rag_context = rag_result.get("context", "")
                        st.session_state.rag_sources = rag_result.get("sources", [])

                    st.success("Knowledge retrieved!")
                    if st.session_state.rag_context:
                        st.text_area("Retrieved Knowledge:", value=st.session_state.rag_context, height=250, disabled=True)

# ============================
# SIDEBAR
# ============================
st.sidebar.title("Agent Trace")
st.sidebar.write("1. Input → OCR/ASR → HITL Edit\n2. Parser Agent (Gemini)\n3. RAG Retrieval (FAISS + Gemini Embeddings)")

st.sidebar.title("Retrieved Context")
if st.session_state.rag_context:
    st.sidebar.text_area("Knowledge", st.session_state.rag_context, height=200, key="sidebar_context")
    st.sidebar.write("**Sources:**")
    for source in st.session_state.rag_sources:
        st.sidebar.caption(f"📄 {source}")
else:
    st.sidebar.info("No context retrieved yet.")

st.sidebar.title("Status")
if st.session_state.parsed_problem:
    if st.session_state.parsed_problem.get("error"):
        st.sidebar.error("Parsing failed")
    else:
        st.sidebar.success("Problem parsed successfully!")
        st.sidebar.json(st.session_state.parsed_problem)
else:
    st.sidebar.info("Waiting for input...")
