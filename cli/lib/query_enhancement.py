import os
import time
import torch
from typing import Optional
from transformers import (
    AutoTokenizer,
    Gemma3ForCausalLM,
    AutoModelForCausalLM,
    pipeline,
)
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("gemini_api_key")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"
model = "gemini-3-flash-preview"
model = "gemini-2.0-flash-lite-001"


class QueryEnhancementGemini:
    def __init__(self, model_path: str = "data/models/gemma-3-1b"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = Gemma3ForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=self.device
        )

    def generate_content(self, prompt: str):
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **model_inputs, max_new_tokens=50, do_sample=False
            )
            generation = generation[0][input_len:]

        decoded = self.tokenizer.decode(generation, skip_special_tokens=True)
        print(f"This is the decoded output: \n{decoded}\n\nEND OF BLOCK")
        return decoded


class LlamaModel:
    def __init__(self, model_path: str = "data/models/llama-3.2-1b"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # self.model = AutoModelForCausalLM.from_pretrained(
        #    model_path, torch_dtype=torch.float16, device_map=self.device
        # )
        self.pipe = pipeline(
            "text-generation",
            model=model_path,
            dtype=torch.bfloat16,
            device_map=self.device,
        )

    def generate_content(self, prompt: str):
        messages = [
            # {
            #    "role": "system",
            #    "content": "You are to follow the requests of the user carefully",
            # },
            {"role": "user", "content": prompt},
        ]
        outputs = self.pipe(messages, max_new_tokens=256)
        return outputs[0]["generated_text"][-1]["content"].strip()


# MODEL = QueryEnhancementGemini()
MODEL = LlamaModel("data/models/gemma-2-2b-it")


def spell_correct(query: str) -> str:
    prompt = f"""Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
Corrected:"""

    response = client.models.generate_content(model=model, contents=prompt)
    corrected = (response.text or "").strip().strip('"')
    return corrected if corrected else query


def rewrite_query(query: str) -> str:
    prompt = f"""Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:"""
    # response = client.models.generate_content(model=model, contents=prompt)
    response = MODEL.generate_content(prompt)
    corrected = (response or "").strip().strip('"')
    return corrected if corrected else query


def expand_query(query: str) -> str:
    prompt = f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{query}"
"""
    response = client.models.generate_content(model=model, contents=prompt)
    corrected = (response.text or "").strip().strip('"')
    return corrected if corrected else query


def enhance_query(query: str, method: Optional[str] = None) -> str:
    match method:
        case "spell":
            return spell_correct(query)
        case "rewrite":
            return rewrite_query(query)
        case "expand":
            return expand_query(query)
        case _:
            return query


def rerank_query(query: str, doc: dict) -> str:
    prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""
    # response = client.models.generate_content(model=model, contents=prompt)
    # corrected = (response.text or "").strip().strip('"')
    # return corrected if corrected else query
    response = MODEL.generate_content(prompt)
    corrected = (response or "").strip().strip('"')
    return corrected if corrected else query


def rerank_method(query: str, results: list[dict]) -> list[dict]:
    for doc in results:
        score = rerank_query(query, doc)
        doc["rerank_score"] = float(score)

    return sorted(results, key=lambda x: x["rerank_score"], reverse=True)
