import os 
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# Setup for BAAI Embeddings HF module
model_name = "BAAI/bge-base-en-v1.5"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}

from langchain.text_splitter import (
    TextSplitter,
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TokenTextSplitter,
)

# Setting the directory where the file is
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "data", "romeo.txt")
db_dir = os.path.join(current_dir, "db")

# Check if the file path exists
if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"The file {file_path} does not exist!"
    )

# Read the content
loader = TextLoader(file_path)
documents = loader.load()

# loading embeddings
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
    ) 

# Function for creating vector store 
def create_vector_store(docs, store_name):
    persistent_dir = os.path.join(db_dir,store_name)
    if not os.path.exists(persistent_dir):
        print(f"\n-----Creating Vector Store 🌀 -----")
        db = Chroma.from_documents(
            docs, embeddings, persist_directory = persistent_dir
        )
        print(f"\n-----Creating Vector Store > {store_name} < is Complete ✅ -----")
    else:
        print(f"\n Vector Store {store_name} already exists. No need to initialize 🔰 ")

# Use Of Different Text Splitters : :

# 1 > Character based Splitting
# Splits The text into chunks based on a specified number of chunks
# Useful For consistent chunk size regardless of the file / content structure 
print("----- Using ⭕️ Character-based Splitter -----")
char_split = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
char_docs = char_split.split_documents(documents)
create_vector_store(char_docs, "chroma_db_char")

# 2 > Sentence based Splitting
# Splits The text into chunks based on sentences, ensuring chunks end at the sentence boundaries
# Useful For maintaining semantic coherence within chunks
print("----- Using ⭕️ Sentence-based Splitter -----")
sen_split = SentenceTransformersTokenTextSplitter(chunk_size = 1000)
sen_docs = sen_split.split_documents(documents)
create_vector_store(sen_docs, "chroma_db_sent")

# 3 > Token based Splitting
# Splits The text into chunks based on tokens (words or sub-words).
# Useful For transformer models with strict token limits
print("----- Using ⭕️ Token-based Splitter -----")
token_split = TokenTextSplitter(chunk_overlap=0, chunk_size = 512)
token_docs = token_split.split_documents(documents)
create_vector_store(token_docs, "chroma_db_token")

# 4 > Recursive based Splitting
# Attempts to split text at natural boundaries (sentences, paragraphs) within character limit.
# Balances between maintaining coherence and adhering to character limits
print("----- Using ⭕️ Recursive-based Splitter -----")
rec_char_split = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 100)
rec_char_docs = rec_char_split.split_documents(documents)
create_vector_store(rec_char_docs, "chroma_db_rec_char")

# 5 > Custom Splitting
# Allows creating custom splitting logic based on specific requirements.
# Useful documents with unique structure that standard splitters can't handle
print("----- Using ⭕️ Custom Splitter -----")


class CustomTextSplitter(TextSplitter):
    def split_text(self, text):
        return text.split("\n\n")

custom_splitter = CustomTextSplitter()
custom_docs = custom_splitter.split_documents(documents)
create_vector_store(custom_docs, "chroma_db_custom")

# Function to query a vector store
def query_vector_store(store_name, query):
    persistent_dir = os.path.join(db_dir, store_name)
    if os.path.exists(persistent_dir):
        print(f"\n--- Querying Vector Store > {store_name} < 🔄 ---") 
        db = Chroma(
            persist_directory=persistent_dir, embedding_function= embeddings
        )
        retriever = db.as_retriever(
            search_type = "similarity_score_threshold",
            search_kwargs={"k":1, "score_threshold": 0.1} 
        )
        relevant_docs = retriever.invoke(query)
        # Display The relevant results with metadata
        print(f"\n--- Relevant Documents For > {store_name} <  ---")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}: \n {doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source','Unknown')}\n")
    else:
        print(f"Vector Store {store_name} does not exist 🔴") 

# Define the user's question
query = "How Did Juliet Die?"

# Query each vector store.
query_vector_store("chroma_db_char", query)
query_vector_store("chroma_db_sent", query)
query_vector_store("chroma_db_token", query)
query_vector_store("chroma_db_rec_char", query)
query_vector_store("chroma_db_custom", query)