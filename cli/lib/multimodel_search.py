import torch
import numpy as np
import os
import requests
import base64
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from lib.search_utils import load_movies
from lib.schemas import Movie, ImageContent
from typing import List, Literal
from dotenv import load_dotenv

load_dotenv()
MODEL_SERVING = os.environ.get("MODEL_SERVING")


class Clip:
    def __init__(
        self, model_name="clip-ViT-B-32", documents: List[Movie] = load_movies()
    ):
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

    def embed_image(self, image_path: str):
        image = Image.open(image_path)
        return self.model.encode(image)

    def search_with_image(self, input_image: str):
        image_emb = self.embed_image(input_image)

        res = []
        for i in range(0, len(self.text_embeddings)):
            doc = self.documents[i].model_dump()
            text_emb = self.text_embeddings[i]
            cos_sim = np.dot(image_emb, text_emb) / (
                np.linalg.norm(image_emb) * np.linalg.norm(text_emb)
            )
            doc["score"] = cos_sim
            res.append(doc)

        return sorted(res, key=lambda x: x["score"], reverse=True)


def image_emded_search(input_image: str, task: Literal["embed", "search"]):
    if MODEL_SERVING == "LOCAL":
        model = Clip()
        if task == "embed":
            return model.embed_image(input_image)
        return model.search_with_image(input_image)

    model_endpoint = os.environ.get("MODEL_ENDPOINT")
    embed_path = os.environ.get("MODEL_IMAGE_EMBED")
    search_path = os.environ.get("MODEL_IMAGE_SERACH")

    input_image = base64.b64encode(open(input_image, "rb").read()).decode("utf-8")
    if task == "embed":
        result = requests.post(
            url=f"{model_endpoint}{embed_path}",
            json=ImageContent(input_image=input_image).model_dump(),
        )

        if result.status_code != 200:
            raise Exception(
                f"Error model request, status_code: {result.status_code} error: {result.text}"
            )
        return result.json()["embedding"]

    result = requests.post(
        url=f"{model_endpoint}{search_path}",
        json=ImageContent(input_image=input_image).model_dump(),
    )
    if result.status_code != 200:
        raise Exception(
            f"Error model request, status_code: {result.status_code} error: {result.text}"
        )
    return result.json()
