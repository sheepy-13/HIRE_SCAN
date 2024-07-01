import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
import os
import chromadb
import asyncio
from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from dotenv import load_dotenv

# Initialize the Sentence Transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Directory to store the SQLite database file
db_directory = './ResumeEmbeddings'
os.makedirs(db_directory, exist_ok=True)
sqlite_db_path = os.path.join(db_directory, 'chromadb.sqlite')

# Initialize ChromaDB persistent client and collection
client = chromadb.PersistentClient(path=sqlite_db_path, settings=Settings())
collection = client.get_or_create_collection(name="resume_embeddings")

# Function to extract text from PDFs
def extract_text_from_pdfs(pdf_files):
    texts = []
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        texts.append((pdf_file.name, text))
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
    
    ids = results.get('ids', [[]])[0]
    distances = results.get('distances', [[]])[0]
    metadatas = results.get('metadatas', [[]])[0]

    sorted_results = sorted(zip(ids, distances, metadatas), key=lambda x: x[1])[:3]
    return sorted_results

# Function to get the response from Semantic Kernel
async def get_response_from_semantic_kernel(query, results_texts):
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    model_id = os.getenv("OPENAI_MODEL_ID")

    if not openai_api_key or not model_id:
        raise ValueError("OpenAI API Key and Model ID must be set in the .env file.")

    kernel = Kernel()

    service_id = "openai_chat_completion"

    chat_completion = OpenAIChatCompletion(service_id=service_id, api_key=openai_api_key, ai_model_id=model_id)
    kernel.add_service(chat_completion)

    req_settings = kernel.get_prompt_execution_settings_from_service_id(service_id)
    req_settings.max_tokens = 2000
    req_settings.temperature = 0.7
    req_settings.top_p = 0.8

    summarize = kernel.add_function(
        function_name="summarize_function",
        plugin_name="summarize_plugin",
        prompt="{{$input}}\n\nAnswer the user's query based on the information provided.",
        prompt_template_settings=req_settings,
    )

    results_combined_text = "\n".join(results_texts)
    final_input_prompt = f"Query: {query}\n\nInformation: {results_combined_text}"

    response = await kernel.invoke(summarize, input=final_input_prompt)
    return response

# Streamlit Chat UI
def main():
    st.title("Resume Query Chatbot")

    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Process PDFs"):
        if pdf_files:
            texts = extract_text_from_pdfs(pdf_files)
            for file_name, text in texts:
                chunks = chunk_text(text)
                embeddings = create_embeddings(chunks)
                save_embeddings_to_chroma(embeddings, chunks, file_name)
            st.success("PDFs processed and embeddings created successfully!")
        else:
            st.warning("Please upload PDF files.")
    
    query_input = st.text_input("Enter your query")

    if st.button("Submit Query"):
        if query_input:
            results = query_embeddings(query_input)
            results_texts = [metadata['text'] for _, _, metadata in results]
            response = asyncio.run(get_response_from_semantic_kernel(query_input, results_texts))
            st.session_state.conversation.append({"query": query_input, "response": response})

    # Display conversation history
    conversation_style = """
    <style>
        .message-row {
            display: flex;
            margin-bottom: 10px;
        }
        .user-message, .bot-message {
            padding: 10px;
            border-radius: 10px;
            max-width: 75%;
            font-family: Arial, sans-serif;
            color: black;
        }
        .user-message {
            background-color: #DCF8C6;
            margin-left: auto;
        }
        .bot-message {
            background-color: #F1F0F0;
            margin-right: auto;
        }
    </style>
    """

    st.markdown(conversation_style, unsafe_allow_html=True)

    for idx, entry in enumerate(st.session_state.conversation):
        st.markdown(f"""
        <div class='message-row'>
            <div class='user-message'>
                <strong>You:</strong> {entry['query']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='message-row'>
            <div class='bot-message'>
                <strong>Bot:</strong> {entry['response']}
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
