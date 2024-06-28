# HIRE_SCAN
Overview
The Resume Query Project is a powerful Streamlit application designed to facilitate the querying and searching of resumes stored in PDF format. By leveraging advanced natural language processing techniques and embedding models, this project allows users to upload resumes, process them, create embeddings, and query these embeddings for relevant information.

Features
PDF Upload: Users can upload multiple PDF files containing resumes.
Text Extraction: Text is extracted from the uploaded PDFs using PyPDF2.
Text Chunking: Extracted text is chunked into smaller pieces for efficient processing.
Embedding Creation: SentenceTransformer is used to create embeddings for the text chunks.
Persistent Storage: Embeddings and metadata are stored persistently using ChromaDB.
Query Interface: Users can input queries to search the embedded resumes and retrieve the most relevant sections.
Requirements
The project requires the following libraries:

streamlit
PyPDF2
sentence-transformers
chromadb
os
Installation
To install the required dependencies, run:

Diff
Copy
Insert
New
pip install streamlit PyPDF2 sentence-transformers chromadb
Make sure you have all necessary Python packages installed before running the application.

Usage
Clone the Repository: Clone this repository to your local machine.

Run the Application: Navigate to the directory where the repository is cloned and run the following command:

Diff
Copy
Insert
New
streamlit run app.py
Upload Resumes: Use the Streamlit interface to upload PDF files containing resumes.

Process Resumes: Click the "Process PDFs" button to extract text, create embeddings, and store them in ChromaDB.

Query Resumes: Enter a query in the provided input field to search the stored resumes. The top 3 relevant results will be displayed along with their distances.
