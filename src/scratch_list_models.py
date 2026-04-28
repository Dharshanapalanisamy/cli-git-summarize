import os
from google import genai
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env") # Make sure to point to the correct .env
api_key = os.getenv("GCM_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
print(f"Using key: {api_key[:10]}...")
client = genai.Client(api_key=api_key)

print("Available models:")
for model in client.models.list():
    print(f"  {model.name}")
