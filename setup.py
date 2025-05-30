from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HUGGING_FACE_INFERENCE_API_KEY = os.getenv("HUGGING_FACE_INFERENCE_API_KEY")

