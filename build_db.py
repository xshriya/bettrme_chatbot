import os
import shutil
import streamlit as st  # <-- THE NEW, IMPORTANT IMPORT
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- Load API Key ---
# Now this will work, because 'st' is imported above.
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    print("ERROR: GOOGLE_API_KEY not found in .streamlit/secrets.toml")
    exit() # Stop the script if the key is missing

print("API key loaded.")

# --- Define Paths ---
DB_DIR = "db"
DATA_DIR = "knowledge"

def main():
    # --- 1. Clean up old database ---
    if os.path.exists(DB_DIR):
        print(f"Deleting old database directory: {DB_DIR}")
        shutil.rmtree(DB_DIR)

    # --- 2. Load Documents ---
    print(f"Loading documents from: {DATA_DIR}")
    loader = DirectoryLoader(DATA_DIR, glob="*.txt", show_progress=True)
    documents = loader.load()
    if not documents:
        print("No documents found. Please check your 'knowledge' folder.")
        return

    print(f"Loaded {len(documents)} documents.")

    # --- 3. Split Documents ---
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} text chunks.")

    # --- 4. Initialize Embeddings ---
    print("Initializing Google embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # --- 5. Create and Save Vector Database (Chroma) ---
    print(f"Creating and saving vector database to: {DB_DIR}")
    db = Chroma.from_documents(
        texts, 
        embeddings, 
        persist_directory=DB_DIR
    )
    
    print("\n--- Database build complete! ---")
    print(f"Database is saved in the '{DB_DIR}' folder.")

# --- Run the main function when the script is executed ---
if __name__ == "__main__":
    main()