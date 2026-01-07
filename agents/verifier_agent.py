from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import json

def verifier_agent(parsed_problem, solution_text):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
        
        prompt = PromptTemplate.from_template("""
        You are a Math Verifier. Check the solution below for the given problem.
        
        Problem: {problem}
        Proposed Solution: {solution}
        
        1. Is the logic correct?
        2. Are the calculations correct?
        
        Return ONLY valid JSON:
        {{
            "is_correct": true,
            "feedback": "Short explanation."
        }}
        """)
        
        chain = prompt | llm
        
        if isinstance(parsed_problem, dict):
            problem_text = parsed_problem.get("problem_text", "")
        else:
            problem_text = str(parsed_problem)
        
        response = chain.invoke({
            "problem": problem_text,
            "solution": solution_text
        })
        
        # JSON Cleaning
        content = response.content
        start = content.find("{")
        end = content.rfind("}") + 1
        return json.loads(content[start:end])
        
    except Exception as e:
        return {"is_correct": False, "feedback": f"Verification failed: {str(e)}"}