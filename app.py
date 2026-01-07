import streamlit as st
import os
import json
import io
import librosa
import easyocr
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
from transformers import pipeline

# --- Modern LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- Import your custom agent logic ---
# Ensure these files use the 'invoke' method rather than 'run'
from agents.supervisor import run_multi_agent_system

# ============================
# 1. ENVIRONMENT & CONFIG
# ============================
load_dotenv()
st.set_page_config(page_title="Math Mentor", layout="wide")

# Verify API Key
if not os.getenv("GOOGLE_API_KEY"):
    st.error("🚨 GOOGLE_API_KEY is MISSING! Please check your .env file.")
    st.stop()

# ============================
# 2. RAG SETUP (Vector Store)
# ============================
@st.cache_resource(show_spinner="Building Math Knowledge Base...")
def get_retriever():
    try:
        # Create folder if missing
        if not os.path.exists("knowledge_base"):
            os.makedirs("knowledge_base")
            with open("knowledge_base/sample.txt", "w") as f:
                f.write("The quadratic formula is x = (-b ± √(b² - 4ac)) / 2a")

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        loader = DirectoryLoader("knowledge_base/", glob="*.txt", loader_cls=TextLoader)
        docs = loader.load()

        if not docs:
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.error(f"RAG Init Error: {e}")
        return None

# Global Retriever Instance
retriever = get_retriever()

# ============================
# 3. CORE LOGIC FUNCTIONS
# ============================

def parse_problem(text: str):
    """Uses Gemini to clean and structure raw input text into JSON."""
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        prompt = PromptTemplate.from_template("""
        You are a precise math problem parser. 
        Clean and structure the following input: {text}
        
        Return ONLY valid JSON:
        {{
          "problem_text": "cleaned version",
          "topic": "algebra|probability|calculus|geometry|other",
          "variables": [],
          "constraints": [],
          "needs_clarification": false
        }}
        """)
        
        # LCEL Chain: Prompt piped to LLM
        chain = prompt | llm
        response = chain.invoke({"text": text})
        
        # Handle JSON parsing from LLM string
        content = response.content
        start = content.find("{")
        end = content.rfind("}") + 1
        return json.loads(content[start:end])
    except Exception as e:
        return {"error": str(e), "needs_clarification": True}

def retrieve_context(problem_text: str):
    """Fetches relevant formulas from the FAISS vector store."""
    if not retriever:
        return {"context": "No specialized knowledge found.", "sources": []}
    try:
        docs = retriever.invoke(problem_text)
        context = "\n".join([d.page_content for d in docs])
        sources = [d.metadata.get("source", "unknown") for d in docs]
        return {"context": context, "sources": sources}
    except Exception as e:
        return {"context": f"Error: {e}", "sources": []}

# ============================
# 4. STREAMLIT UI
# ============================
st.title("🧠 Multimodal Math Mentor")

# State Management
if "parsed_problem" not in st.session_state:
    st.session_state.parsed_problem = None
if "rag_context" not in st.session_state:
    st.session_state.rag_context = ""

# --- INPUT SECTION ---
input_mode = st.radio("Input Method:", ["Text", "Image", "Audio"], horizontal=True)
raw_text = ""

if input_mode == "Text":
    raw_text = st.text_area("Enter your math problem:", height=100)

elif input_mode == "Image":
    up_img = st.file_uploader("Upload Problem Image", type=["png", "jpg", "jpeg"])
    if up_img:
        st.image(up_img, width=300)
        if st.button("Extract Text"):
            reader = easyocr.Reader(['en'])
            ocr_result = reader.readtext(up_img.getvalue(), detail=0)
            raw_text = " ".join(ocr_result)
            st.text_input("OCR Result (Verify/Edit):", value=raw_text, key="ocr_edit")

elif input_mode == "Audio":
    up_aud = st.file_uploader("Upload Audio", type=["wav", "mp3"])
    if up_aud:
        if st.button("Transcribe"):
            pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
            # Convert to numpy for librosa if needed, or pass directly to pipeline
            audio, sr = librosa.load(io.BytesIO(up_aud.read()), sr=16000)
            raw_text = pipe({"raw": audio, "sampling_rate": sr})["text"]
            st.text_input("Transcription:", value=raw_text, key="asr_edit")

# --- ACTION BUTTON ---
if st.button("Step 1: Parse & Search Knowledge"):
    input_to_parse = st.session_state.get("ocr_edit") or st.session_state.get("asr_edit") or raw_text
    
    if input_to_parse:
        with st.spinner("Analyzing..."):
            # 1. Parse
            st.session_state.parsed_problem = parse_problem(input_to_parse)
            # 2. Retrieve
            rag_data = retrieve_context(st.session_state.parsed_problem.get("problem_text", ""))
            st.session_state.rag_context = rag_data["context"]
            st.session_state.rag_sources = rag_data["sources"]
            st.rerun()

# --- RESULTS SECTION ---
if st.session_state.parsed_problem:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Parsed Structure")
        st.json(st.session_state.parsed_problem)
    with col2:
        st.subheader("Retrieved Formulas")
        st.info(st.session_state.rag_context if st.session_state.rag_context else "No matching formulas found.")

    # --- MULTI-AGENT SOLVER ---
    if st.button("Step 2: Solve with Multi-Agent System", type="primary"):
        with st.spinner("Agents are collaborating..."):
            results = run_multi_agent_system(
                parsed_problem=st.session_state.parsed_problem,
                rag_context=st.session_state.rag_context
            )
            
            st.divider()
            st.subheader("Final Solution")
            st.write(results.get("final_solution", "Error generating solution."))
            
            with st.expander("View Step-by-Step Explanation"):
                st.write(results.get("explanation"))
            
            with st.expander("View Verifier Feedback"):
                v = results.get("verification", {})
                st.write(f"Correct: {v.get('is_correct')}")
                st.write(v.get("feedback"))

# --- SIDEBAR TRACE ---
st.sidebar.title("Agent Trace")
if st.session_state.parsed_problem:
    st.sidebar.success("✅ Parser Agent Done")
    if st.session_state.rag_context:
        st.sidebar.success("✅ RAG Retrieval Done")