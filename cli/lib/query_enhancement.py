import os
from typing import Optional

from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("gemini_api_key")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"
model = "gemini-3-flash-preview"


def spell_correct(query: str) -> str:
    prompt = f"""Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
Corrected:"""

    response = client.models.generate_content(model=model, contents=prompt)
    corrected = (response.text or "").strip().strip('"')
    return corrected if corrected else query


def enhance_query(query: str, method: Optional[str] = None) -> str:
    match method:
        case "spell":
            return spell_correct(query)
        case _:
            return query
