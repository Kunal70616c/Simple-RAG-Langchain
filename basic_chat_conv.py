from dotenv import load_dotenv , find_dotenv
load_dotenv(find_dotenv(), override = True)
import os
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.schema import AIMessage, HumanMessage, SystemMessage

#create Model
model = ChatGoogleGenerativeAI(model= 'gemini-1.5-flash', temperature= 1)

chat_history = [] # Stores Chat history

# Setting Initial System message
sys_msg = SystemMessage(content='Useful AI Assistant')
chat_history.append(sys_msg) # Also adding sys msg to the history

print("\n\n- - - - - - - - Talk With AI - - - - - - - -\n")
# Chat LOOP
while True:
    query = input("YOU 😎 : ")
    if query.lower() == "exit" :
        break
    chat_history.append(HumanMessage(content=query)) # Add User message to history

    # Gen AI Response
    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response)) # Add AI message to history

    print(f"AI 🤖 : {response}")

print("- - - - - - - - History - - - - - - - -")
print(chat_history)



