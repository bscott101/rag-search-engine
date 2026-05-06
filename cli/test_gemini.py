import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash-001"
model = "gemini-3-flash-preview"

if __name__ == "__main__":
    prompt = "If I have two dollary doos and give you one, how many do I have?"
    response = client.models.generate_content(model=model, contents=prompt)
    assert response.usage_metadata is not None
    print(f"Prompt Tokens: {response.usage_metadata.prompt_token_count}")
    print(f"Response Tokens: {response.usage_metadata.candidates_token_count}")
    print(response.text)

# havent made a request an im over the qouta - ffs
# Time to make your own wagie
