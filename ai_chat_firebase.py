from dotenv import load_dotenv , find_dotenv
load_dotenv(find_dotenv(), override = True)
import os
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory

# Setup FireBase Firestore
PROJECT_ID = "langchain-ai-basic"
SESSION_ID = "cov_with_kunal_new"  # This could be a username or a unique ID
COLLECTION_NAME = "chat_history"

# Initialize Firestore Client
print("Initializing Firestore Client......")
client = firestore.Client(project=PROJECT_ID)

# Initialize Firestore Chat History
print("Initializing Firebase Chat History ......")
chat_history = FirestoreChatMessageHistory(
    session_id = SESSION_ID,
    collection = COLLECTION_NAME,
    client = client,
)
print("Chat History Restored")
print("- - - - - - - - Current Chat History - - - - - - - -\n\n", chat_history.messages)

# Initialize Chat Model
model = ChatGoogleGenerativeAI(model= 'gemini-1.5-flash', temperature= 1)

print("\n\n- - - - - - - - Talk With AI - - - - - - - -\n")

while True:
    human_input = input("YOU 😎 : ")
    if human_input.lower() == "exit" :
        break
    chat_history.add_user_message(human_input)
    
    ai_response = model.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)
    
    print(f"AI 🤖 : {ai_response.content}")



