import torch
import io
import numpy as np
import base64
from ray import serve
from PIL import Image
from sentence_transformers import SentenceTransformer
from numpy.typing import NDArray
from src.schemas import Movie
from typing import List


@serve.deployment(ray_actor_options={"num_gpus": 0.2})
class Clip:
    def __init__(self, documents: List[Movie], model_name="clip-ViT-B-32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.documents = documents
        self.texts: List[str] = self._gen_texts()
        self.text_embeddings = self._gen_embs()

    def _gen_texts(self):
        temp = []
        for doc in self.documents:
            temp.append(f"{doc.title}: {doc.description}")
        return temp

    def _gen_embs(self):
        return self.model.encode(self.texts, show_progress_bar=True)

    async def embed_image(self, input_image: str) -> NDArray[np.float32]:
        image_data = base64.b64decode(input_image)
        image = Image.open(io.BytesIO(image_data))
        return self.model.encode(image)

    async def search_with_image(self, input_image: str, limit: int = 3):
        image_emb = await self.embed_image(input_image)
        image_emb = image_emb.tolist()
        res = []
        for i in range(0, len(self.text_embeddings)):
            doc = self.documents[i].model_dump()
            text_emb = self.text_embeddings[i]
            cos_sim = np.dot(image_emb, text_emb) / (
                np.linalg.norm(image_emb) * np.linalg.norm(text_emb)
            )
            doc["score"] = cos_sim
            res.append(doc)

        return sorted(res, key=lambda x: x["score"], reverse=True)[:limit]
