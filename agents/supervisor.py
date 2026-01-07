from agents import solver_agent
from agents import verifier_agent
from agents import explainer_agent

def run_multi_agent_system(parsed_problem: dict, rag_context: str) -> dict:
    """
    Orchestrates the Solver -> Verifier -> Explainer flow.
    Uses data already parsed in Step 1 of app.py.
    """
    
    # 1. Run Solver
    # We pass the dictionary and the context string
    print("--- 🤖 Running Solver ---")
    solution_data = solver_agent.solver_agent(parsed_problem, rag_context)
    solution_text = solution_data.get("solution_text", "Error in solution generation.")

    # 2. Run Verifier
    print("--- 🕵️ Running Verifier ---")
    verification = verifier_agent.verifier_agent(parsed_problem, solution_text)

    # 3. Handle Verification Results
    final_solution = solution_text
    if not verification.get("is_correct"):
        final_solution = f"ORIGINAL SOLUTION:\n{solution_text}\n\n⚠️ VERIFICATION NOTE:\n{verification.get('feedback')}"

    # 4. Run Explainer
    print("--- 🧑‍🏫 Running Explainer ---")
    explanation = explainer_agent.explainer_agent(final_solution, rag_context)

    return {
        "final_solution": final_solution,
        "verification": verification,
        "explanation": explanation
    }