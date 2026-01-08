# 🧠 Multimodal Math Mentor – Reliable JEE-Style Math Solver

**A complete end-to-end AI application built for the AI Planet AI Engineer Assignment**

This Math Mentor helps students solve JEE-level math problems using **text, image (photo/screenshot), or audio input**. It combines **RAG, multi-agent reasoning, human-in-the-loop correction, memory, and self-learning** to provide accurate, step-by-step solutions that improve over time.

**Live Deployed App**: [[Insert your Streamlit/HuggingFace link here after deployment]](https://multimodal-math-mentor-application-htcsnrnevzwopdtadgz6hb.streamlit.app/)

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

Here is a clean, focused `LOCAL_SETUP.md` file you can add to your repository. It specifically addresses the errors you faced earlier (like the `libGL` issue) to ensure a smooth run for anyone cloning your repo.

---

```markdown
# 🛠️ Local Setup & Run Guide

Follow these steps to get **Math Mentor** running on your local machine.

## Prerequisites
* **Python 3.10+** installed.
* **Git** installed.
* A **Google Gemini API Key** (Get it [here](https://aistudio.google.com/app/apikey)).

---

## 🚀 Installation Steps

### 1. Clone the Repository
Open your terminal and run:
```bash
git clone [https://github.com/YOUR_USERNAME/math-mentor.git](https://github.com/YOUR_USERNAME/math-mentor.git)
cd math-mentor

```

### 2. Create a Virtual Environment (Recommended)

It is best practice to use a virtual environment to avoid conflicts.

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate

```

**Mac / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate

```

### 3. Install System Dependencies (Linux Users Only)

If you are running on **Linux** (Ubuntu/Debian) or **WSL**, you must install the graphics libraries required by OpenCV. If you are on Windows or Mac, you can usually skip this.

```bash
sudo apt-get update
sudo apt-get install libgl1

```

*(If you see an `ImportError: libGL.so.1` later, this step is the fix!)*

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt

```

### 5. Configure Environment Variables

1. Create a file named `.env` in the root directory.
2. Open it and paste your Google API Key:
```env
GOOGLE_API_KEY=AIzaSy...YourKeyHere...

```



---

## ▶️ Running the Application

Once installation is complete, start the Streamlit server:

```bash
streamlit run app.py

```

* The app should automatically open in your browser at `http://localhost:8501`.
* If it doesn't open, strictly check the terminal for the URL.

---

