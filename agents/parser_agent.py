from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool

llm = ChatOpenAI(temperature=0.2)  # Low temp for accuracy

@tool
def clean_math_text(text: str) -> str:
    """Clean and standardize math problem text."""
    # Simple cleaning: remove noise, format equations
    cleaned = text.strip().replace('\n', ' ').lower()
    # Add more logic if needed, e.g., regex for math symbols
    return cleaned

tools = [clean_math_text]
prompt = hub.pull("hwchase17/react")  # ReAct prompt template
parser_agent = create_react_agent(llm, tools, prompt)
parser_executor = AgentExecutor(agent=parser_agent, tools=tools, verbose=True)

def run_parser(input_text: str) -> str:
    result = parser_executor.invoke({"input": f"Parse and clean this math problem: {input_text}"})
    return result['output']