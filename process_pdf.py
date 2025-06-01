import os
import pickle
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
from setup import HUGGING_FACE_INFERENCE_API_KEY
from huggingface_hub import InferenceClient

# Directory to store session data
SESSION_DATA_DIR = 'session_data'
os.makedirs(SESSION_DATA_DIR, exist_ok=True)

# Initialize the HuggingFace client
client = InferenceClient(token=HUGGING_FACE_INFERENCE_API_KEY)

def get_embedding(text):
    try:
        # Use the client to get embeddings with specific model
        embeddings = client.feature_extraction(
            text,
            model="sentence-transformers/all-MiniLM-L6-v2"  # Specify the model explicitly
        )
        # Ensure we get a single embedding vector
        if isinstance(embeddings, list):
            embeddings = embeddings[0]
        # Convert to numpy array and ensure it's 1D
        embeddings = np.array(embeddings, dtype=np.float32).flatten()
        print(f"Embedding shape: {embeddings.shape}")  # Debug print
        return embeddings
    except Exception as e:
        print(f"Error getting embedding: {str(e)}")
        raise

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
    embeddings = []
    valid_chunks = []
    
    for i, chunk in enumerate(chunk_list):
        try:
            embedding = get_embedding(chunk)
            # Only add if embedding has a valid shape
            if embedding.size > 0:
                embeddings.append(embedding)
                valid_chunks.append(chunk)
            else:
                print(f"Skipping chunk {i} due to empty embedding")
        except Exception as e:
            print(f"Error processing chunk {i}: {str(e)}")
            continue
    
    if not embeddings:
        raise Exception("No embeddings were generated successfully")
    
    # Check shapes before stacking
    shapes = [emb.shape for emb in embeddings]
    print(f"All embedding shapes: {shapes}")  # Debug print
    
    # Ensure all embeddings have the same shape
    target_shape = embeddings[0].shape
    valid_embeddings = []
    for i, emb in enumerate(embeddings):
        if emb.shape == target_shape:
            valid_embeddings.append(emb)
        else:
            print(f"Skipping embedding {i} with shape {emb.shape} (expected {target_shape})")
    
    if not valid_embeddings:
        raise Exception("No valid embeddings after shape validation")
    
    # Convert to numpy array and ensure all embeddings have the same shape
    embeddings = np.stack(valid_embeddings)
    print(f"Final embeddings shape: {embeddings.shape}")  # Debug print
    
    # Create FAISS index
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings)

    # Save FAISS index and chunks to disk
    faiss_file = os.path.join(SESSION_DATA_DIR, f'{session_id}_faiss.index')
    faiss.write_index(faiss_index, faiss_file)
    chunks_file = os.path.join(SESSION_DATA_DIR, f'{session_id}_chunks.pkl')
    with open(chunks_file, 'wb') as f:
        pickle.dump(valid_chunks, f)  # Save only the valid chunks

    return "PDF processed and indexed successfully."


def query_pdf(user_prompt: str, session_id: str, top_k: int = 2):
    try:
        # Load FAISS index and chunks from disk
        faiss_file = os.path.join(SESSION_DATA_DIR, f'{session_id}_faiss.index')
        chunks_file = os.path.join(SESSION_DATA_DIR, f'{session_id}_chunks.pkl')
        
        print(f"Checking files: {faiss_file} and {chunks_file}")  # Debug print
        
        if not os.path.exists(faiss_file) or not os.path.exists(chunks_file):
            print(f"Files not found: faiss={os.path.exists(faiss_file)}, chunks={os.path.exists(chunks_file)}")
            return []
        
        print("Loading FAISS index...")  # Debug print
        faiss_index = faiss.read_index(faiss_file)
        print(f"FAISS index dimension: {faiss_index.d}")
        
        print("Loading chunks...")  # Debug print
        with open(chunks_file, 'rb') as f:
            chunks = pickle.load(f)
        print(f"Loaded {len(chunks)} chunks")
        
        print("Getting embedding for query...")  # Debug print
        query_vec = np.array(get_embedding(user_prompt)).astype('float32')
        print(f"Query vector shape: {query_vec.shape}")
        
        # Reshape query vector if needed
        if len(query_vec.shape) == 1:
            query_vec = query_vec.reshape(1, -1)
        print(f"Reshaped query vector shape: {query_vec.shape}")
        
        print("Performing FAISS search...")  # Debug print
        D, I = faiss_index.search(query_vec, top_k)
        print(f"Search results - Distances: {D}, Indices: {I}")
        
        # Clean up RAM
        del faiss_index
        del query_vec
        
        print("Returning results...")  # Debug print
        results = [chunks[i] for i in I[0]]
        print(f"Returning {len(results)} results")
        return results
        
    except Exception as e:
        print(f"Error in query_pdf: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise
