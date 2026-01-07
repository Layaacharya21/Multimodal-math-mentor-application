from agents import parser_agent
from agents import explainer_agent
from agents import retriever_agent
from agents import verifier_agent
from agents import solver_agent

def run_multi_agent_system(input_text: str) -> dict:
    cleaned = parser_agent.run_parser(input_text)
    context = retriever_agent.run_retriever(cleaned)
    solution = solver_agent.run_solver(cleaned, context)
    verification = verifier_agent.run_verifier(solution, cleaned)
    explanation = explainer_agent.run_explainer(solution, context) if "valid" in verification.lower() else "Solution needs review."
    return {
        "cleaned": cleaned,
        "context": context,
        "solution": solution,
        "verification": verification,
        "explanation": explanation
    }