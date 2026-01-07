from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import create_retriever_tool
from langchain import hub  # Required to get the prompt

def get_retriever_executor(retriever):
    """
    Creates the agent executor using the retriever passed from app.py
    """
    # 1. Check if retriever exists
    if not retriever:
        raise ValueError("Retriever object is None. Cannot create retriever agent.")

    # 2. Define the LLM (Ensure you have OPENAI_API_KEY in .env)
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.1)

    # 3. Create the Tool
    retriever_tool = create_retriever_tool(
        retriever,
        "math_knowledge_retriever",
        "Searches the math knowledge base for relevant formulas and tips."
    )
    tools = [retriever_tool]

    # 4. Get the Reference Prompt (Fixes the 'missing argument' error)
    # This pulls the standard ReAct prompt from LangChain Hub
    prompt = hub.pull("hwchase17/react")

    # 5. Create Agent & Executor
    agent = create_react_agent(llm, tools, prompt)
    
    # Return the executor so it can be used/invoked
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# Helper function to run it easily
def run_retriever_agent(retriever, query: str) -> str:
    try:
        executor = get_retriever_executor(retriever)
        result = executor.invoke({"input": f"Retrieve context for: {query}"})
        return result['output']
    except Exception as e:
        return f"Retriever Agent Error: {str(e)}"