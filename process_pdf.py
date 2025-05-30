import os
import pickle
import requests
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
from setup import HUGGING_FACE_INFERENCE_API_KEY

# Directory to store session data
SESSION_DATA_DIR = 'session_data'
os.makedirs(SESSION_DATA_DIR, exist_ok=True)

HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
HF_API_KEY = HUGGING_FACE_INFERENCE_API_KEY
HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}


def get_embedding(text):
    response = requests.post(HF_API_URL, headers=HF_HEADERS, json={"inputs": text})
    response.raise_for_status()
    return response.json()


def process_pdf(file_path: str, session_id: str) -> str:
    # Read PDF
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    # Chunk text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunk_list = splitter.split_text(text)

    # Embed chunks using Hugging Face API
    embeddings = [get_embedding(chunk)[0] for chunk in chunk_list]
    embeddings = np.array(embeddings).astype('float32')

    # Create FAISS index
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings)

    # Save FAISS index and chunks to disk
    faiss_file = os.path.join(SESSION_DATA_DIR, f'{session_id}_faiss.index')
    faiss.write_index(faiss_index, faiss_file)
    chunks_file = os.path.join(SESSION_DATA_DIR, f'{session_id}_chunks.pkl')
    with open(chunks_file, 'wb') as f:
        pickle.dump(chunk_list, f)

    return "PDF processed and indexed successfully."


def query_pdf(user_prompt: str, session_id: str, top_k: int = 2):
    # Load FAISS index and chunks from disk
    faiss_file = os.path.join(SESSION_DATA_DIR, f'{session_id}_faiss.index')
    chunks_file = os.path.join(SESSION_DATA_DIR, f'{session_id}_chunks.pkl')
    if not os.path.exists(faiss_file) or not os.path.exists(chunks_file):
        return []
    faiss_index = faiss.read_index(faiss_file)
    with open(chunks_file, 'rb') as f:
        chunks = pickle.load(f)
    query_vec = np.array(get_embedding(user_prompt)).astype('float32')
    D, I = faiss_index.search(query_vec, top_k)
    # Clean up RAM
    del faiss_index
    del query_vec
    return [chunks[i] for i in I[0]]
