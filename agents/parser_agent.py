from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
# No LLMChain import needed!
import json

def parse_problem(text: str):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
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
        # MODERN WAY: Using LCEL (LangChain Expression Language)
        chain = prompt | llm 
        
        # .invoke replaces .run
        response = chain.invoke({"text": text})
        
        # ChatModels return a BaseMessage object, so we access .content
        content = response.content

        start = content.find("{")
        end = content.rfind("}") + 1
        json_str = content[start:end]
        return json.loads(json_str)
    except Exception as e:
        return {"error": str(e)}