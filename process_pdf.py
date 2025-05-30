import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Globals for FAISS and chunks
faiss_index = None
chunks = []
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def process_pdf(file_path: str) -> str:
    global faiss_index, chunks
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
    chunks = chunk_list

    # Embed chunks
    embeddings = embedding_model.encode(chunk_list, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    # Create FAISS index
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings)

    return "PDF processed and indexed successfully."


def query_pdf(user_prompt: str, top_k: int = 2):
    global faiss_index, chunks
    if faiss_index is None or not chunks:
        return []
    # Embed the user prompt
    query_vec = embedding_model.encode([user_prompt]).astype('float32')
    # Search FAISS
    D, I = faiss_index.search(query_vec, top_k)
    # Return top_k chunks
    return [chunks[i] for i in I[0]]
