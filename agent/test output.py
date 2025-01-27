import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

api_key = dotenv.get_key(key_to_get='GOOGLE_API_KEY', dotenv_path='../.env')
genai.configure(api_key=api_key)
model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

chain = LLMChain(
    llm=model, prompt="""Given the following user question, corresponding SQL query, and the SQL result, answer the user question.

            Question: {question}
            SQL Query: {query}
            SQL Result: {result}

            Answer: """+"How to check count of workers at office plaza_3?"
)

respose = chain.run()
print(respose)