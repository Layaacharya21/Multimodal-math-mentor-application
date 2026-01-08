from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

def explainer_agent(solution_text, problem_context):
    try:
        # Temperature 0.7 for more natural/friendly tone
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)
        
        prompt = PromptTemplate.from_template("""
        You are a helpful and encouraging Math Mentor.
        
        Problem Context: {context}
        Technical Solution: {solution}

        Please explain this solution step-by-step in simple English. 
        - Break down the logic.
        - Explain "why" we took each step.
        - Be encouraging!

        Explanation:
        """)
        
        chain = prompt | llm
        response = chain.invoke({"context": problem_context, "solution": solution_text})
        return response.content
        
    except Exception as e:
        return f"Explainer Error: {str(e)}"