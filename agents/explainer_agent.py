from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor

llm = ChatOpenAI(temperature=0.5)  # Higher for natural explanations

tools = []  # No tools needed, just generation
explainer_agent = create_react_agent(llm, tools)
explainer_executor = AgentExecutor(agent=explainer_agent, tools=tools, verbose=True)

def run_explainer(solution: str, context: str) -> str:
    input_prompt = f"Explain this solution step-by-step with tips: {solution}\nContext: {context}"
    result = explainer_executor.invoke({"input": input_prompt})
    return result['output']