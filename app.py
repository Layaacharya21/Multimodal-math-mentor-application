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

# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- Custom imports ---
from agents.supervisor import run_multi_agent_system
from memory import init_db, save_solution, find_similar_solution  # Phase 5

# ============================
# 1. ENVIRONMENT & CONFIG
# ============================
load_dotenv()
st.set_page_config(page_title="Math Mentor", layout="wide")

# Check for Google API key
if not os.getenv("GOOGLE_API_KEY"):
    st.error("🚨 GOOGLE_API_KEY is MISSING! Please add it to your .env file.")
    st.stop()

# Initialize memory database
init_db()

# ============================
# 2. RAG SETUP (Vector Store with Local Embeddings)
# ============================
@st.cache_resource(show_spinner="Building Math Knowledge Base...")
def get_retriever():
    try:
        if not os.path.exists("knowledge_base"):
            os.makedirs("knowledge_base")
            with open("knowledge_base/sample.txt", "w") as f:
                f.write("The quadratic formula is x = (-b ± √(b² - 4ac)) / 2a\n"
                        "Common mistake: forgetting the sign of b or dividing incorrectly.")

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        loader = DirectoryLoader("knowledge_base/", glob="*.txt", loader_cls=TextLoader)
        docs = loader.load()

        if not docs:
            st.warning("No documents in knowledge_base/. RAG will be limited.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.error(f"RAG Init Error: {e}")
        return None

retriever = get_retriever()

# ============================
# 3. CORE LOGIC FUNCTIONS
# ============================
def parse_problem(text: str):
    """Uses Gemini 1.5 Flash to parse and structure the math problem."""
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",  # Reliable free-tier model with high quota
            temperature=0,
            convert_system_message_to_human=True
        )
        
        prompt = PromptTemplate.from_template("""
        You are a precise math problem parser. Clean and structure the following input:
        
        {text}
        
        Return ONLY valid JSON in this exact format:
        {{
          "problem_text": "cleaned and readable version of the problem",
          "topic": "algebra|probability|calculus|geometry|other",
          "variables": [],
          "constraints": [],
          "needs_clarification": false
        }}
        
        Do not add any explanation. Output only JSON.
        """)
        
        chain = prompt | llm
        response = chain.invoke({"text": text})
        
        content = response.content.strip()
        start = content.find("{")
        end = content.rfind("}") + 1
        if start == -1 or end <= start:
            raise ValueError("No valid JSON found in response")
        
        return json.loads(content[start:end])
    
    except Exception as e:
        return {"error": f"Parsing failed: {str(e)}", "needs_clarification": True}

def retrieve_context(problem_text: str):
    if not retriever:
        return {"context": "No knowledge base loaded.", "sources": []}
    try:
        docs = retriever.invoke(problem_text)
        context = "\n\n".join([d.page_content.strip() for d in docs])
        sources = [os.path.basename(d.metadata.get("source", "unknown")) for d in docs]
        return {"context": context, "sources": sources}
    except Exception as e:
        return {"context": f"Retrieval error: {e}", "sources": []}

# ============================
# 4. STREAMLIT UI
# ============================
st.title("🧠 Multimodal Math Mentor")

# Session state initialization
for key in ["parsed_problem", "rag_context", "rag_sources", "ocr_text", "asr_text"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "rag_sources" else "" if key == "rag_context" else None

# Default solution variables for sidebar
final_solution = None
explanation = None
verification = {}

# --- INPUT SECTION ---
input_mode = st.radio("Input Method:", ["Text", "Image", "Audio"], horizontal=True)

if input_mode == "Text":
    st.text_area("Enter your math problem:", height=120, key="text_input")

elif input_mode == "Image":
    up_img = st.file_uploader("Upload Problem Image", type=["png", "jpg", "jpeg"])
    if up_img:
        st.image(up_img, caption="Uploaded Image", width=400)
        if st.button("Extract Text (OCR)"):
            with st.spinner("Extracting text from image..."):
                reader = easyocr.Reader(['en'], gpu=False)
                result = reader.readtext(up_img.getvalue(), detail=0)
                extracted = " ".join(result)
                st.session_state.ocr_text = extracted
                st.rerun()

        if st.session_state.ocr_text:
            st.text_area("Edit OCR Result (HITL):", value=st.session_state.ocr_text, height=120, key="ocr_edit_final")

elif input_mode == "Audio":
    up_aud = st.file_uploader("Upload Audio", type=["wav", "mp3"])
    if up_aud:
        if st.button("Transcribe Audio"):
            with st.spinner("Transcribing audio..."):
                audio_bytes = up_aud.read()
                audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
                pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
                result = pipe({"raw": audio, "sampling_rate": sr})
                transcript = result["text"]
                st.session_state.asr_text = transcript
                st.rerun()

        if st.session_state.asr_text:
            st.text_area("Edit Transcription (HITL):", value=st.session_state.asr_text, height=120, key="asr_edit_final")

# --- STEP 1: PARSE & RETRIEVE ---
if st.button("Step 1: Parse Problem & Retrieve Knowledge", type="secondary"):
    # Get correct input text
    if input_mode == "Image" and "ocr_edit_final" in st.session_state:
        input_text = st.session_state.ocr_edit_final
    elif input_mode == "Audio" and "asr_edit_final" in st.session_state:
        input_text = st.session_state.asr_edit_final
    elif input_mode == "Text" and "text_input" in st.session_state:
        input_text = st.session_state.text_input
    else:
        input_text = ""

    if not input_text.strip():
        st.warning("Please provide text via input, image, or audio.")
    else:
        with st.spinner("Parsing problem and retrieving knowledge..."):
            st.session_state.parsed_problem = parse_problem(input_text)
            if "error" not in st.session_state.parsed_problem:
                rag_data = retrieve_context(st.session_state.parsed_problem["problem_text"])
                st.session_state.rag_context = rag_data["context"]
                st.session_state.rag_sources = rag_data["sources"]
            st.rerun()

# --- RESULTS DISPLAY ---
if st.session_state.parsed_problem:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Parsed Problem")
        st.json(st.session_state.parsed_problem)
        
        if st.session_state.parsed_problem.get("needs_clarification"):
            st.warning("Problem is ambiguous — please clarify and re-parse.")
        if st.session_state.parsed_problem.get("error"):
            st.error(st.session_state.parsed_problem["error"])

    with col2:
        st.subheader("Retrieved Knowledge")
        if st.session_state.rag_context:
            st.info(st.session_state.rag_context)
            st.caption("Sources:")
            for src in st.session_state.rag_sources:
                st.caption(f"📄 {src}")
        else:
            st.info("No relevant knowledge found.")

    # --- PHASE 5: MULTI-AGENT SOLVING + MEMORY + FEEDBACK ---
    if st.button("Step 2: Solve with Multi-Agent System", type="primary"):
        if st.session_state.parsed_problem.get("error") or st.session_state.parsed_problem.get("needs_clarification"):
            st.error("Cannot solve until parsing is successful.")
            st.stop()

        problem_text = st.session_state.parsed_problem["problem_text"]

        # Memory reuse
        reused = find_similar_solution(problem_text)
        if reused:
            st.success("🎉 Reusing verified correct solution from memory!")
            final_solution = reused
            explanation = "Reused from previous correct solution."
            verification = {"is_correct": True, "feedback": "Memory reuse"}
        else:
            with st.spinner("Agents collaborating on solution..."):
                results = run_multi_agent_system(
                    parsed_problem=st.session_state.parsed_problem,
                    rag_context=st.session_state.rag_context
                )
            final_solution = results.get("final_solution", "No solution generated.")
            explanation = results.get("explanation", "")
            verification = results.get("verification", {})

        st.divider()
        st.subheader("Final Answer")
        st.markdown(final_solution)

        with st.expander("Step-by-Step Explanation"):
            st.write(explanation or "No explanation provided.")

        with st.expander("Verifier Report"):
            verdict = "Correct" if verification.get("is_correct") else "Uncertain"
            st.write(f"**Verdict:** {verdict}")
            st.write(verification.get("feedback", "No feedback"))

        # --- FEEDBACK (HITL + Self-Learning) ---
        st.markdown("---")
        st.subheader("Was this solution correct?")
        c1, c2 = st.columns(2)
        
        if c1.button("✅ Correct", type="primary", use_container_width=True):
            save_solution(problem_text, st.session_state.parsed_problem, final_solution, "correct")
            st.success("Saved as correct — will be reused in future!")

        if c2.button("❌ Incorrect", type="secondary", use_container_width=True):
            corrected = st.text_area("Enter the correct solution:", height=200)
            if st.button("Submit Correction"):
                save_solution(problem_text, st.session_state.parsed_problem, final_solution, "incorrect", corrected)
                st.success("Correction saved — thank you for improving the system!")

# --- SIDEBAR ---
st.sidebar.title("Agent Trace")
st.sidebar.write("1. Input → Extraction → HITL Edit")
st.sidebar.write("2. Parser → Structured Problem")
if st.session_state.parsed_problem and "error" not in st.session_state.parsed_problem:
    st.sidebar.success("✅ Parser Complete")
    if st.session_state.rag_context:
        st.sidebar.success("✅ RAG Retrieval Complete")
    if final_solution:
        st.sidebar.success("✅ Multi-Agent Solving Complete")
        if verification.get("is_correct"):
            st.sidebar.success("✅ Verified Correct")
        else:
            st.sidebar.warning("⚠️ Verification Uncertain")

st.sidebar.title("Memory")
if os.path.exists("math_mentor_memory.db"):
    size = os.path.getsize("math_mentor_memory.db")
    st.sidebar.info(f"Active ({size:,} bytes stored)")
else:
    st.sidebar.info("Ready")
    