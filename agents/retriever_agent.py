from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import create_retriever_tool
llm = ChatOpenAI(temperature=0.1)

# Assume retriever from app.py is passed or global
retriever_tool = create_retriever_tool(
    retriever,
    "math_knowledge_retriever",
    "Searches the math knowledge base for relevant formulas and tips."
)
tools = [retriever_tool]
retriever_agent = create_react_agent(llm, tools)
retriever_executor = AgentExecutor(agent=retriever_agent, tools=tools, verbose=True)

def run_retriever(query: str) -> str:
    result = retriever_executor.invoke({"input": f"Retrieve context for: {query}"})
    return result['output']