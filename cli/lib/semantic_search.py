import string
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def verify_model(self):
        print(f"Model loaded: {self.model}")
        print(f"Max sequence length: {self.model.max_seq_length}")

    def generate_embedding(self, text: str) -> NDArray:
        if not text or not text.strip():
            raise ValueError("cannot generate embedding for empty text")
        emb = self.model.encode([text])[0]
        return emb


def verify_model():
    model = SemanticSearch()
    model.verify_model()


def embed_text(text: str):
    model = SemanticSearch()
    emb = model.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {emb[:3]}")
    print(f"Dimensions: {emb.shape[0]}")
