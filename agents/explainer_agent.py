from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

def explainer_agent(solution_text, problem_context):
    try:
        # Use a model with slightly higher temperature for more natural language
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
        
        prompt = PromptTemplate.from_template("""
        You are a helpful and encouraging Math Mentor.
        
        The student has the following math problem and solution:
        Problem Context: {context}
        Technical Solution: {solution}

        Please explain this solution step-by-step in simple English. 
        - Break down the logic.
        - Explain "why" we took each step.
        - If there are formulas, explain what they mean.
        - Be encouraging!

        Explanation:
        """)
        
        # LCEL Chain
        chain = prompt | llm
        response = chain.invoke({"context": problem_context, "solution": solution_text})
        return response.content
        
    except Exception as e:
        return f"Explainer Error: {str(e)}"