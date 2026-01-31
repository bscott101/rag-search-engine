import torch
from PIL import Image
from transformers import pipeline


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

        print("made it here")
        if input_image:
            raw_image = Image.open(input_image).convert("RGB")
            user_prompt["content"].append({"type": "image"})
            messages.append(user_prompt)

            outputs = self.pipe(
                images=raw_image,
                text=messages,
                max_new_tokens=max_new_tokens, 
                temperature=temp
            )
        else:
            messages.append(user_prompt)
            outputs = self.pipe(
                text=messages, max_new_tokens=max_new_tokens, temperature=temp
            )
        
        return outputs[0]["generated_text"][-1]["content"].strip()
 