import os
import json
import torch
from dotenv import load_dotenv
from google import genai
from transformers import AutoTokenizer, Gemma3ForCausalLM, pipeline

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


class LLMModel:
    def __init__(
        self, model_path: str = "data/models/llama-3.2-1b", complex: bool = False
    ):
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
        self.complex = complex

    def generate_content(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        temp: float = 0.3,
    ):
        messages = [
            {"role": "user", "content": prompt},
        ]
        if self.complex:
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]

        outputs = self.pipe(messages, max_new_tokens=200, temperature=temp)
        return outputs[0]["generated_text"][-1]["content"].strip()


# MODEL = QueryEnhancementGemini("data/models/gemma-2-2b-it")
MODEL = LLMModel("data/models/gemma-3-4b-it", True)


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


def rerank_batch_method(query: str, results: list[dict]) -> list[dict]:
    doc_list_str = []

    for id, doc in enumerate(results):
        print(f'id: {id} title: {doc["title"]}')
        doc_list_str.append(
            # f"ID: {id} Title: {doc['title']} Description: {doc['document'][:400]}"
            f"{doc['document'][:300]}"
        )

    prompt = f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[1, 3, 4, 2]
"""
    # surely having an AI rank other AI items is not a good idea
    response = MODEL.generate_content(
        prompt,
        system_prompt="You are a movie critic",
    )
    res = json.loads(response)

    if len(res) > len(results):
        res = res[: len(results)]

    if len(res) < len(results):
        diff = len(results) - len(res)
        res.extend([9999 for _ in range(0, diff)])

    for id, score in enumerate(res):
        results[id]["rerank_batch"] = score

    return sorted(results, key=lambda x: x["rerank_batch"])
