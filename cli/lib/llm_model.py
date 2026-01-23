import torch
from transformers import pipeline


class LLMModel:
    def __init__(
        self, model_path: str = "data/models/gemma-3-4b-it", complex: bool = False
    ):
        self.device = self._get_device()
        self.pipe = pipeline(
            "text-generation",
            model=model_path,
            dtype=torch.bfloat16,
            device_map=self.device,
        )
        self.complex = complex

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

        outputs = self.pipe(messages, max_new_tokens=300, temperature=temp)
        return outputs[0]["generated_text"][-1]["content"].strip()
