from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
import time
from setup import GEMINI_API_KEY
import google.generativeai as genai
import os
from fastapi.responses import HTMLResponse
import uuid
from process_pdf import process_pdf, query_pdf

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSION_STATE = {}

@app.get("/hello")
async def root():
    return HTMLResponse("Hello! This is the Backend site of the PDF Chat Application.")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        session_id = str(uuid.uuid4())
        file_path = f"uploaded_{session_id}.pdf"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        # Process the PDF and store state
        msg = process_pdf(file_path, session_id)
        SESSION_STATE[session_id] = file_path
        return {"message": msg, "session_id": session_id}
    except Exception as e:
        return {"error": str(e)}

@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        message = data.get("message")
        session_id = data.get("session_id")
        if not message:
            return {"error": "No message provided"}
        if not session_id or session_id not in SESSION_STATE:
            return {"error": "Invalid or missing session_id. Please upload a PDF first."}
        # Query the PDF for relevant context
        top_chunks = query_pdf(message, session_id, top_k=2)
        context = "\n\n".join(top_chunks)
        prompt = f"Context from PDF:\n{context}\n\nUser question: {message}\n\nAnswer:"
        response = model.generate_content(prompt)
        answer = response.text
        # Optionally, clear session after use to free memory
        del SESSION_STATE[session_id]
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}
