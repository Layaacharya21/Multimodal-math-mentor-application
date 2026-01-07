from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
import sympy  # For symbolic math; add to requirements.txt if needed: pip install sympy

llm = ChatOpenAI(temperature=0.3)

@tool
def solve_equation(equation: str) -> str:
    """Solve a math equation using sympy."""
    try:
        expr = sympy.sympify(equation)
        solution = sympy.solve(expr)
        return str(solution)
    except Exception as e:
        return f"Error: {e}"

tools = [solve_equation]  # Add more tools like calculator if needed
solver_agent = create_react_agent(llm, tools)
solver_executor = AgentExecutor(agent=solver_agent, tools=tools, verbose=True)

def run_solver(problem: str, context: str) -> str:
    input_prompt = f"Solve this problem using context: {problem}\nContext: {context}"
    result = solver_executor.invoke({"input": input_prompt})
    return result['output']