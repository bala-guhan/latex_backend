from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time
from setup import GEMINI_API_KEY
import google.generativeai as genai
import os
from fastapi.responses import HTMLResponse
import uuid
from process_pdf import process_pdf, query_pdf
from typing import Dict
import asyncio
from pydantic import BaseModel
import shutil
from pathlib import Path

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Local dev frontend
        "https://latex-frontend-self.vercel.app",  # Production frontend
        "http://127.0.0.1:5173", # Local dev frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSION_STATE = {}

# Store processing status
processing_status: Dict[str, dict] = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str

@app.get("/hello")
async def root():
    return HTMLResponse("Hello! This is the Backend site of the PDF Chat Application.")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        print(f"Received upload request for file: {file.filename}")  # Debug log
        
        if not file.filename.endswith('.pdf'):
            print(f"Invalid file type: {file.filename}")  # Debug log
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Create a unique session ID
        session_id = str(uuid.uuid4())
        print(f"Generated session ID: {session_id}")  # Debug log
        
        # Create session directory
        session_dir = 'session_data'
        os.makedirs(session_dir, exist_ok=True)
        print(f"Created/verified session directory: {session_dir}")  # Debug log
        
        # Save the uploaded file
        file_path = os.path.join(session_dir, f"{session_id}.pdf")
        try:
            content = await file.read()
            print(f"Read file content, size: {len(content)} bytes")  # Debug log
            
            with open(file_path, "wb") as buffer:
                buffer.write(content)
            print(f"Saved file to: {file_path}")  # Debug log
        except Exception as e:
            print(f"Error saving file: {str(e)}")  # Debug log
            raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
        
        # Start processing in background
        try:
            asyncio.create_task(process_pdf_async(file_path, session_id))
            print(f"Started background processing for session: {session_id}")  # Debug log
        except Exception as e:
            print(f"Error starting background processing: {str(e)}")  # Debug log
            # Clean up the uploaded file if background processing fails
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=500, detail=f"Error starting processing: {str(e)}")
        
        return {"session_id": session_id}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in upload endpoint: {str(e)}")  # Debug log
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

async def process_pdf_async(file_path: str, session_id: str):
    try:
        print(f"Starting PDF processing for session: {session_id}")  # Debug log
        
        # Update status to processing
        processing_status[session_id] = {
            "status": "processing",
            "message": "Reading PDF..."
        }
        
        # Process the PDF
        print(f"Calling process_pdf for file: {file_path}")  # Debug log
        process_pdf(file_path, session_id)
        print(f"PDF processing completed for session: {session_id}")  # Debug log
        
        # Update status to completed
        processing_status[session_id] = {
            "status": "completed",
            "message": "Processing completed"
        }
    except Exception as e:
        print(f"Error in process_pdf_async: {str(e)}")  # Debug log
        # Update status to error and clean up
        processing_status[session_id] = {
            "status": "error",
            "message": str(e)
        }
        cleanup_session(session_id)

@app.get("/processing-status/{session_id}")
async def get_processing_status(session_id: str):
    if session_id not in processing_status:
        raise HTTPException(status_code=404, detail="Session not found")
    return processing_status[session_id]

def cleanup_session(session_id: str):
    """Clean up all files and data associated with a session"""
    try:
        # Remove PDF file
        pdf_path = f"session_data/{session_id}.pdf"
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        
        # Remove FAISS index
        faiss_path = f"session_data/{session_id}_faiss.index"
        if os.path.exists(faiss_path):
            os.remove(faiss_path)
        
        # Remove chunks file
        chunks_path = f"session_data/{session_id}_chunks.pkl"
        if os.path.exists(chunks_path):
            os.remove(chunks_path)
        
        # Remove from processing status
        if session_id in processing_status:
            del processing_status[session_id]
            
        print(f"Cleaned up session data for {session_id}")
    except Exception as e:
        print(f"Error cleaning up session {session_id}: {str(e)}")

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Get relevant chunks from the PDF
        relevant_chunks = query_pdf(request.message, request.session_id)
        
        if not relevant_chunks:
            # Clean up if no relevant chunks found
            cleanup_session(request.session_id)
            return {"answer": "I couldn't find any relevant information in the PDF to answer your question."}
        
        # Create a prompt template with the context and user's question
        prompt = f"""You are a helpful AI assistant. Use the following context from a PDF document to answer the user's question.
        If the context doesn't contain enough information to answer the question, say so.

        Context from PDF:
        {' '.join(relevant_chunks)}

        User's question: {request.message}

        Please provide a clear and concise answer based on the context:"""

        # Generate response using Gemini
        response = model.generate_content(prompt)
        
        if not response.text:
            # Clean up if no response generated
            cleanup_session(request.session_id)
            return {"answer": "I apologize, but I couldn't generate a response. Please try rephrasing your question."}
        
        # Clean up after successful response
        cleanup_session(request.session_id)
        return {"answer": response.text}
        
    except Exception as e:
        # Clean up on error
        if 'session_id' in locals():
            cleanup_session(request.session_id)
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add a cleanup endpoint for manual cleanup if needed
@app.delete("/cleanup/{session_id}")
async def cleanup_endpoint(session_id: str):
    try:
        cleanup_session(session_id)
        return {"message": "Session cleaned up successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
