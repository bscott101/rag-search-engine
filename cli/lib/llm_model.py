from annotated_types import Ge
import torch
import base64
import os
import requests
from PIL import Image
from transformers import pipeline
from dotenv import load_dotenv
from lib.schemas import GenerateContent

load_dotenv()
MODEL_SERVING = os.environ.get("MODEL_SERVING")


class LLMModel:
    def __init__(self, model_path: str = "./data/models/gemma-3-4b-it"):
        self.device = self._get_device()
        self.pipe = pipeline(
            "image-text-to-text",
            model=model_path,
            dtype=torch.bfloat16,
            device_map=self.device,
            use_fast=True,
        )

    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.mps.is_available():
            return "mps"

        return "cpu"

    def generate_content(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        temp: float = 0.3,
        max_new_tokens=300,
        input_image: str | None = None,
    ):
        messages = []

        messages.append(
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        )

        user_prompt = {"role": "user", "content": [{"type": "text", "text": prompt}]}

        if input_image:
            raw_image = Image.open(input_image).convert("RGB")
            user_prompt["content"].append({"type": "image"})
            messages.append(user_prompt)

            outputs = self.pipe(
                images=raw_image,
                text=messages,
                max_new_tokens=max_new_tokens,
                temperature=temp,
            )
        else:
            messages.append(user_prompt)
            outputs = self.pipe(
                text=messages, max_new_tokens=max_new_tokens, temperature=temp
            )

        return outputs[0]["generated_text"][-1]["content"].strip()


def llm_inference(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    temp: float = 0.3,
    max_new_tokens=300,
    input_image: str | None = None,
):
    if MODEL_SERVING == "LOCAL":
        model = LLMModel()
        return model.generate_content(
            prompt, system_prompt, temp, max_new_tokens, input_image
        )

    model_endpoint = os.environ.get("MODEL_ENDPOINT")
    gemma_path = os.environ.get("MODEL_GENERATE_CONTENT")
    if model_endpoint is None:
        raise Exception(
            "Model inference not set to local and no MODEL_ENDPOINT environment set"
        )
    if gemma_path is None:
        raise Exception(
            "Model inference not set to local and no MODEL_GENERATE_CONTENT environment set"
        )

    if input_image:
        input_image = base64.b64encode(open(input_image, "rb").read()).decode("utf-8")

    result = requests.post(
        url=f"{model_endpoint}{gemma_path}",
        json=GenerateContent(
            prompt=prompt,
            system_prompt=system_prompt,
            temp=temp,
            max_new_tokens=max_new_tokens,
            input_image=input_image,
        ).model_dump(),
    )

    if result.status_code != 200:
        raise Exception(
            f"Error model request, status_code: {result.status_code} error: {result.text}"
        )

    return result.json()["response"]
