import torch
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from lib.search_utils import load_movies
from lib.schemas import Movie
from typing import List



class Clip:
    def __init__(self, model_name="clip-ViT-B-32", documents: List[Movie] = load_movies()):
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
            cos_sim = (
                np.dot(image_emb, text_emb) / (np.linalg.norm(image_emb) * np.linalg.norm(text_emb))
            )
            doc["score"] = cos_sim
            res.append(doc)
            
        return sorted(res, key=lambda x: x["score"], reverse=True)



