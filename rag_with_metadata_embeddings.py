import os
from dotenv import load_dotenv , find_dotenv
load_dotenv(find_dotenv(), override = True)
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

model_name = "BAAI/bge-base-en-v1.5"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}

# loading embeddings
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
    ) 

# define Path
current_dir = os.path.dirname(os.path.abspath(__file__))
docs_dir = os.path.join(current_dir,"data")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

print(f"Document Directory:{docs_dir}")
print(f"Persistent Directory:{persistent_directory}")

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")
    
    # Ensure the text file exists
    if not os.path.exists(docs_dir):
        raise FileNotFoundError(
            f"The Directory {docs_dir} does not exist. Please check the path."
        )
# List all text files in the directory
    docs_files = [f for f in os.listdir(docs_dir) if f.endswith(".md")]
    
    # Read the text content from each file and store it with metadata
    documents = []
    for book_file in docs_files:
        file_path = os.path.join(docs_dir, book_file)
        loader = TextLoader(file_path)
        book_docs = loader.load()
        for doc in book_docs:
            # Add metadata to each document indicating its source
            doc.metadata = {"source": book_file}
            documents.append(doc)
    
    # Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    # Create embeddings
    print("\n--- Creating embeddings ---")
    embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
    ) 
    print("\n--- Finished creating embeddings ---")

    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating vector store ---")

else:
    print("Vector store already exists. No need to initialize.")

