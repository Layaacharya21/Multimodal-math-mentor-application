<pre>
graph TD
    %% Main Entry and UI
    User[User (Student)] -->|Uploads Image/Audio/Text| UI[Streamlit UI]
    UI -->|Raw Input Data| InputProcessor[Multimodal Input Handler]
    
    %% Layer 1: Input Processing
    subgraph "Input Processing Layer"
        InputProcessor -->|Image Path| OCR[OCR Service (e.g., PaddleOCR)]
        InputProcessor -->|Audio Path| ASR[ASR Service (e.g., Whisper)]
        InputProcessor -->|Text Path| TextNorm[Text Normalizer]
    end
    
    %% Data Convergence
    OCR & ASR & TextNorm -->|Cleaned Raw Text| ParserAgent[Parser Agent]
    
    %% Layer 2: Agent Orchestration & Reasoning
    subgraph "Agent Orchestration Layer"
        ParserAgent -->|Structured JSON Problem| RouterAgent[Intent Router Agent]
        RouterAgent -->|Route to Topic Expert| SolverAgent[Solver Agent]
        
        %% Solver Interaction with RAG
        SolverAgent <-->|Query & Retrieve Context| RAG[RAG Pipeline]
        
        SolverAgent -->|Proposed Solution| VerifierAgent[Verifier/Critic Agent]
        
        %% Verification Logic branching
        VerifierAgent -- "Confident & Correct" --> ExplainerAgent[Explainer Agent]
        VerifierAgent -- "Unsure or Incorrect" --> HITL[Human-in-the-Loop Interface]
    end
    
    %% Layer 3: Knowledge & Memory
    subgraph "Knowledge & Memory Infrastructure"
        RAG <-->|Vector Search| VectorDB[(Vector DB - FAISS/Chroma)]
        VectorDB <-->|Ingest Documents| KB[Knowledge Base (Formulas/Templates)]
        
        SolverAgent <-->|Read Past similar solutions| Memory[(Long-term Memory Store)]
        HITL -->|Write Corrections/Feedback| Memory
    end
    
    %% Outputs back to UI
    ExplainerAgent -->|Final Step-by-Step Explanation| UI
    HITL -->|Manual Correction Output| UI

</pre>
