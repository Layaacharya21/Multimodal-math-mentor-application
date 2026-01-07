# Already good if you moved your parse_problem function here
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json

def parse_problem(text: str):
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
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(text=text)

        start = response.find("{")
        end = response.rfind("}") + 1
        json_str = response[start:end]
        return json.loads(json_str)
    except Exception as e:
        return {"error": str(e)}