import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
import os
import chromadb


# Initialize the Sentence Transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Directory to store the SQLite database file
db_directory = './ResumeEmbeddings'  # Change this to your actual path
os.makedirs(db_directory, exist_ok=True)
sqlite_db_path = os.path.join(db_directory, 'chromadb.sqlite')

# Initialize ChromaDB persistent client and collection
client = chromadb.PersistentClient(path=sqlite_db_path,settings=Settings())
collection = client.get_or_create_collection(name="resume_embeddings")

# Function to extract text from PDFs
def extract_text_from_pdfs(pdf_files):
    texts = []
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        texts.append((pdf_file.name, text))  # Store file name along with text
    return texts

# Function to chunk text
def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

# Function to create embeddings
def create_embeddings(text_chunks):
    embeddings = model.encode(text_chunks, show_progress_bar=True)
    print(embeddings)
    return embeddings

# Function to save embeddings to Chroma DB
def save_embeddings_to_chroma(embeddings, text_chunks, file_name):
    for i, embedding in enumerate(embeddings):
        collection.upsert(
            ids=[f"{file_name}chunk{i}"],
            metadatas=[{"file_name": file_name, "text": text_chunks[i]}],
            embeddings=[embedding.tolist()]
        )

# Function to query embeddings using ChromaDB
def query_embeddings(query, max_results=10):
    query_embedding = model.encode([query])[0].tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=max_results)
    
    # Ensure results['documents'] always exists even if no results are returned
    ids= results.get('ids', [[]])[0]
    distances = results.get('distances', [[]])[0]
    metadatas = results.get('metadatas', [[]])[0]

    # Sort by distance (ascending) and take the top 3 results
    sorted_results = sorted(zip(ids, distances, metadatas), key=lambda x: x[1])[:3]
    
    return sorted_results

# Streamlit UI
def main():
    st.title("Resume Query")
    
    # Upload PDF files
    pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Process PDFs"):
        if pdf_files:
            # Extract and chunk text
            texts = extract_text_from_pdfs(pdf_files)
            for file_name, text in texts:
                chunks = chunk_text(text)
                
                # Create and save embeddings
                embeddings = create_embeddings(chunks)
                save_embeddings_to_chroma(embeddings, chunks, file_name)
            
            st.success("PDFs processed and embeddings created successfully!")
        else:
            st.warning("Please upload PDF files.")
    
    # Query system
    query = st.text_input("Enter your query")
    if query:
        results = query_embeddings(query)
        print(results)
        st.subheader("Top 3 Results by Distance:")
        for ids, distance, metadata in results:
            st.write(metadata["file_name"])
            st.write(distance)
            st.write(metadata['text'])
            st.write("---")

if __name__== '__main__':
    main()