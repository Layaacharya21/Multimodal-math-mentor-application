from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import create_retriever_tool
from langchain import hub

def get_retriever_executor(retriever):
    """
    Creates a ReAct agent capable of using the vector store to look up math formulas.
    """
    # 1. Safety Check
    if not retriever:
        raise ValueError("Retriever object is None. Cannot create retriever agent.")

    # 2. Define the LLM (Updated to Gemini 1.5 Flash)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest", 
        temperature=0,
        convert_system_message_to_human=True
    )

    # 3. Define the Tool
    retriever_tool = create_retriever_tool(
        retriever,
        "math_knowledge_retriever",
        "Searches the math knowledge base for relevant formulas and tips."
    )
    tools = [retriever_tool]

    # 4. Get the ReAct Prompt
    # Pulls the standard prompt from LangChain Hub
    prompt = hub.pull("hwchase17/react")

    # 5. Create Agent
    agent = create_react_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True  # Critical for Gemini stability
    )

def run_retriever_agent(retriever, query: str) -> str:
    """
    Standalone function to run retrieval via an agent (Thought -> Action -> Observation).
    """
    try:
        executor = get_retriever_executor(retriever)
        result = executor.invoke({"input": f"Retrieve relevant math context for: {query}"})
        return result['output']
    except Exception as e:
        return f"Retriever Agent Error: {str(e)}"