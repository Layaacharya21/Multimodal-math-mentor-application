def run_multi_agent_system(input_text: str) -> dict:
    cleaned = run_parser(input_text)
    context = run_retriever(cleaned)
    solution = run_solver(cleaned, context)
    verification = run_verifier(solution, cleaned)
    explanation = run_explainer(solution, context) if "valid" in verification.lower() else "Solution needs review."
    return {
        "cleaned": cleaned,
        "context": context,
        "solution": solution,
        "verification": verification,
        "explanation": explanation
    }