from dotenv import load_dotenv , find_dotenv
load_dotenv(find_dotenv(), override = True)
import os
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

# Lists All The Model 
for model in genai.list_models():
    print(model.name)

llm = ChatGoogleGenerativeAI(model= 'gemini-1.5-flash', temperature= 1)

# Sending Prompt
response = llm.invoke('Write a Short story about a broken pencil')

# Print Response
print(response.content)
print(response.response_metadata)