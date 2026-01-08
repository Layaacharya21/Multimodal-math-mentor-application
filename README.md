# 🧠 Multimodal Math Mentor – Reliable JEE-Style Math Solver

**A complete end-to-end AI application built for the AI Planet AI Engineer Assignment**

This Math Mentor helps students solve JEE-level math problems using **text, image (photo/screenshot), or audio input**. It combines **RAG, multi-agent reasoning, human-in-the-loop correction, memory, and self-learning** to provide accurate, step-by-step solutions that improve over time.

**Live Deployed App**: [Insert your Streamlit/HuggingFace link here after deployment]

**Demo Video** (4 minutes): [Insert YouTube/Unlisted link here]

## 🚀 Features

- **Multimodal Input**
  - Type text directly
  - Upload image/screenshot of handwritten or printed math problems (OCR with EasyOCR + HITL editing)
  - Upload audio of spoken math questions (Whisper ASR + HITL editing)

- **Intelligent Parsing**
  - Parser Agent (Gemini 1.5 Flash) converts raw input into structured JSON
  - Detects ambiguity and triggers HITL if needed

- **RAG Pipeline**
  - Curated knowledge base (10+ TXT files with formulas, templates, common mistakes)
  - FAISS vector store with local HuggingFace embeddings
  - Retrieves relevant knowledge and displays sources

- **Multi-Agent System**
  - Supervisor orchestrates 5+ agents: Parser, Router, Solver, Verifier, Explainer
  - Uses RAG context for grounded reasoning
  - Verifier checks correctness and confidence

- **Human-in-the-Loop (HITL)**
  - Editable OCR/transcription before solving
  - Feedback buttons (Correct / Incorrect + correction input)
  - Corrections stored for self-improvement

- **Memory & Self-Learning**
  - SQLite database stores past interactions
  - Reuses verified correct solutions for similar problems
  - Learns from user corrections over time

- **UI Features**
  - Agent trace in sidebar
  - Retrieved context panel with sources
  - Confidence indicator via verifier
  - Clean step-by-step explanation

## 🏗️ Architecture Diagram

<img width="2547" height="5895" alt="Untitled diagram-2026-01-08-173250" src="https://github.com/user-attachments/assets/6226eb85-1131-4d6b-9f33-f42e40d14ea9" />

