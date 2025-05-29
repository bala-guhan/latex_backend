from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time
from setup import GEMINI_API_KEY
import google.generativeai as genai

genai.configure(api_key=GEMINI_API_KEY)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api_hello")
def read_root():
    return {"message" : "Ellaam maaya, Ellaam chaaya!"}


@app.post("/convert")
def convert(data: dict):
    try:
        data = data.get("input")
        prompt = f"""
        Convert the following latex code to markdown:
        And only return the markdown code, nothing else.
        {data}
        """
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return {"output": response.text}
    except Exception as e:
        return {"error": str(e)}
