import os

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from dotenv import load_dotenv , find_dotenv
load_dotenv(find_dotenv(), override = True)
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.schema import AIMessage, HumanMessage, SystemMessage


model_name = "BAAI/bge-base-en-v1.5"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}

# define Path
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

# loading embeddings
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
    ) 

#load the existing vector database
db = Chroma(persist_directory=persistent_directory, 
            embedding_function=embeddings)

# User Query
query = "What is python's use ?"

# retrieve relevant docs based on query
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
) 
relevant_docs = retriever.invoke(query)

# display the result

print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    
# Combine the query and the relevant document we found .

combined_input = (
    "Here are some documents that might help answer the question:"
    + query
    + "\n\nRelevant Documents:"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\nPlease provide answer based on provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
)

# Create LLM Model 
model = ChatGoogleGenerativeAI(model= 'gemini-1.5-flash', temperature= 1)

# Defining messages for this model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# Invoke
result = model.invoke(messages)

# Display the results
print("\n-----Generated Response-----")
print("Content ONLY ::")
print(result.content)

