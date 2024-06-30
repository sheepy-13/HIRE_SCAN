import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
import os
import chromadb
import re

# Initialize the Sentence Transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Directory to store the SQLite database file
db_directory = './ResumeEmbeddings'
os.makedirs(db_directory, exist_ok=True)
sqlite_db_path = os.path.join(db_directory, 'chromadb.sqlite')

# Initialize ChromaDB persistent client and collection
client = chromadb.PersistentClient(path=sqlite_db_path, settings=Settings())
collection = client.get_or_create_collection(name="resume_embeddings")

# Function to extract emails
def extract_emails(text):
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return re.findall(email_pattern, text)

# Function to extract text from PDFs
def extract_text_from_pdfs(pdf_files):
    texts = []
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        emails = extract_emails(text)
        texts.append((pdf_file.name, text, emails))  # Store file name, text, and emails
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
def save_embeddings_to_chroma(embeddings, text_chunks, file_name, emails):
    emails_str = ", ".join(emails)  # Convert email list to a comma-separated string
    for i, embedding in enumerate(embeddings):
        collection.upsert(
            ids=[f"{file_name}chunk{i}"],
            metadatas=[{
                "file_name": file_name,
                "text": text_chunks[i],
                "emails": emails_str  # Store emails as a concatenated string
            }],
            embeddings=[embedding.tolist()]
        )

# Function to query embeddings using ChromaDB
def query_embeddings(query, max_results=10):
    query_embedding = model.encode([query])[0].tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=max_results)
    
    # Ensure results['documents'] always exists even if no results are returned
    ids = results.get('ids', [[]])[0]
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
            for file_name, text, emails in texts:
                chunks = chunk_text(text)
                
                # Create and save embeddings
                embeddings = create_embeddings(chunks)
                save_embeddings_to_chroma(embeddings, chunks, file_name, emails)
            
            st.success("PDFs processed and embeddings created successfully!")
        else:
            st.warning("Please upload PDF files.")
    
    # Query system
    # Query system
    query = st.text_input("Enter your query")
    if query:
        results = query_embeddings(query)
        
        # Use a set to keep track of displayed file names
        displayed_files = set()
        
        st.subheader("Top 3 Results by Distance:")
        count = 0
        for ids, distance, metadata in results:
            file_name = metadata.get('file_name', 'Unknown file')
            
            # Check if this file has already been displayed
            if file_name not in displayed_files:
                text = metadata.get('text', 'No text available')
                emails = metadata.get('emails', 'No emails found')

                st.write(f"File Name: {file_name}")
                st.write(f"Distance: {distance}")
                st.write(f"Text: {text}")
                st.write(f"Emails: {emails}")  # Displaying emails, if available
                st.write("---")

                # Mark this file as displayed
                displayed_files.add(file_name)
                count += 1

                # Stop after displaying 3 unique files
                if count >= 3:
                    break

if __name__ == '__main__':
        main()
