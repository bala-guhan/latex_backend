import os
import pickle
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Directory to store session data
SESSION_DATA_DIR = 'session_data'
os.makedirs(SESSION_DATA_DIR, exist_ok=True)

MODEL_NAME = 'all-MiniLM-L6-v2'


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

    # Embed chunks
    embedding_model = SentenceTransformer(MODEL_NAME)
    embeddings = embedding_model.encode(chunk_list, show_progress_bar=True)
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

    # Clean up RAM
    del faiss_index
    del embedding_model
    del embeddings

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
    embedding_model = SentenceTransformer(MODEL_NAME)
    query_vec = embedding_model.encode([user_prompt]).astype('float32')
    D, I = faiss_index.search(query_vec, top_k)
    # Clean up RAM
    del faiss_index
    del embedding_model
    del query_vec
    return [chunks[i] for i in I[0]]
