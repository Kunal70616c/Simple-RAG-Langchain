from dotenv import load_dotenv , find_dotenv
load_dotenv(find_dotenv(), override = True)
import os
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

# Lists All The Model 
for model in genai.list_models():
    print(model.name)

llm = ChatGoogleGenerativeAI(model= 'gemini-1.5-flash', temperature= 1)


print("\n\n- - - - - - - - Talk With AI - - - - - - - -\n")

query = input("Ask AI: ")
# Sending Prompt
response = llm.invoke(query)

# Print Response
print(f"AI 🤖 : {response.content}")
