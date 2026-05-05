import torch
import io
import base64
from ray import serve
from PIL import Image
from transformers import pipeline
from src.schemas import GenerateContent

@serve.deployment(ray_actor_options={"num_gpus": 0.8})
class Gemma3:
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
    
    async def generate_content(self, input: GenerateContent):
        messages = []

        messages.append(
            {"role": "system", "content": [{"type": "text", "text": input.system_prompt}]}
        )

        user_prompt = {"role": "user", "content": [{"type": "text", "text": input.prompt}]}

        if input.input_image:
            image_data = base64.b64decode(input.input_image)
            raw_image = Image.open(io.BytesIO(image_data)).convert("RGB")
            user_prompt["content"].append({"type": "image"})
            messages.append(user_prompt)

            outputs = self.pipe(
                images=raw_image,
                text=messages,
                max_new_tokens=input.max_new_tokens,
                temperature=input.temp,
            )
        else:
            messages.append(user_prompt)
            outputs = self.pipe(
                text=messages, max_new_tokens=input.max_new_tokens, temperature=input.temp
            )

        return {"response": outputs[0]["generated_text"][-1]["content"].strip()}