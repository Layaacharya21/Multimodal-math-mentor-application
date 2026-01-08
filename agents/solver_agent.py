from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

def solver_agent(parsed_problem, rag_context):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0)
        
        prompt = PromptTemplate.from_template("""
        You are an expert Math Solver. 
        Solve the problem below using the provided context formulas.
        
        Problem: {problem_text}
        Topic: {topic}
        Context/Formulas: {rag_context}
        
        Show your work step-by-step. Return the final answer clearly.
        """)
        
        chain = prompt | llm
        
        # Handle if parsed_problem is dict or string
        if isinstance(parsed_problem, dict):
            problem_text = parsed_problem.get("problem_text", "")
            topic = parsed_problem.get("topic", "General Math")
        else:
            problem_text = str(parsed_problem)
            topic = "General Math"
        
        response = chain.invoke({
            "problem_text": problem_text, 
            "topic": topic,
            "rag_context": rag_context
        })
        
        return {"solution_text": response.content}
        
    except Exception as e:
        return {"solution_text": f"Solver Error: {str(e)}"}