from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import json

def parser_agent(text: str):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        prompt = PromptTemplate.from_template("""
        You are a precise math problem parser.
        Input: {text}
        Return ONLY valid JSON:
        {{
          "problem_text": "cleaned problem statement",
          "topic": "algebra|probability|calculus|geometry|other",
          "variables": [],
          "constraints": [],
          "needs_clarification": false
        }}
        """)
        chain = prompt | llm 
        response = chain.invoke({"text": text})
        
        content = response.content
        start = content.find("{")
        end = content.rfind("}") + 1
        return json.loads(content[start:end])
    except Exception as e:
        return {"error": str(e)}