import streamlit as st
from dotenv import load_dotenv
import easyocr
from PIL import Image
import io
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from transformers import pipeline
import librosa
import json
# Add at the very top of app.py (temporary debug)
import os
print("GEMINI KEY:", os.getenv("GEMINI_API_KEY")[:5] + "..." if os.getenv("GEMINI_API_KEY") else "MISSING")

load_dotenv()

st.title("Math Mentor")

# Input mode selector
input_mode = st.selectbox("Choose input mode:", ["Text", "Image", "Audio"])

# Shared session state to store parsed problem later
if "parsed_problem" not in st.session_state:
    st.session_state.parsed_problem = None

# Parser function using Gemini
def parse_problem(text: str):
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",  # Fast, great for math, supports JSON mode
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

If the problem is ambiguous or incomplete, set needs_clarification to true.
Do not add explanations. Output only JSON.
"""
        )

        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(text=text)

        # Clean response in case of extra text
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            response = response[json_start:json_end]

        return json.loads(response)
    except Exception as e:
        return {"error": "Parsing failed", "details": str(e), "raw_response": response if 'response' in locals() else ""}

# Text Input
if input_mode == "Text":
    problem_text = st.text_area("Type your math problem:", height=150)
    if st.button("Parse and Proceed"):
        with st.spinner("Parsing with Gemini..."):
            st.session_state.parsed_problem = parse_problem(problem_text)
            st.json(st.session_state.parsed_problem)
            if st.session_state.parsed_problem.get("needs_clarification", False):
                st.warning("Problem is ambiguous! Please clarify and try again.")

# Image Input
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
                edited_text = st.text_area("Edit/correct the text below:", value=extracted_text, height=150, key="edited_ocr")

                if avg_conf < 0.8:
                    st.warning("Low confidence OCR – please review and edit!")

            if st.button("Parse and Proceed"):
                problem_text = st.session_state.get("edited_ocr", extracted_text)
                with st.spinner("Parsing with Gemini..."):
                    st.session_state.parsed_problem = parse_problem(problem_text)
                    st.json(st.session_state.parsed_problem)

# Audio Input
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
                edited_transcript = st.text_area("Edit if needed:", value=transcript, height=150, key="edited_asr")

                if len(transcript.strip()) < 10:
                    st.warning("Short or unclear transcription – please check!")

        if st.button("Parse and Proceed"):
            problem_text = st.session_state.get("edited_asr", transcript)
            with st.spinner("Parsing with Gemini..."):
                st.session_state.parsed_problem = parse_problem(problem_text)
                st.json(st.session_state.parsed_problem)

# Sidebar
st.sidebar.title("Agent Trace")
st.sidebar.write("Phase 1: Input → Extraction → Editing (HITL) → Parsing")

st.sidebar.title("Retrieved Context")
st.sidebar.info("RAG context will appear here in Phase 3")

st.sidebar.title("Status")
if st.session_state.parsed_problem:
    st.sidebar.success("Problem parsed successfully!")
    st.sidebar.json(st.session_state.parsed_problem)
else:
    st.sidebar.info("No problem parsed yet.")