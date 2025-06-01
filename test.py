from huggingface_hub import InferenceClient
from setup import HUGGING_FACE_INFERENCE_API_KEY

client = InferenceClient(token=HUGGING_FACE_INFERENCE_API_KEY)
print(client.feature_extraction("Hi, who are you?"))