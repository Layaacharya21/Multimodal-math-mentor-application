from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool

llm = ChatOpenAI(temperature=0.1)

@tool
def check_accuracy(solution: str, problem: str) -> str:
    """Verify if solution matches problem."""
    # Simple check; expand with sympy eval
    return "Valid" if "error" not in solution.lower() else "Invalid"

tools = [check_accuracy]
verifier_agent = create_react_agent(llm, tools)
verifier_executor = AgentExecutor(agent=verifier_agent, tools=tools, verbose=True)

def run_verifier(solution: str, problem: str) -> str:
    result = verifier_executor.invoke({"input": f"Verify solution for {problem}: {solution}"})
    return result['output']