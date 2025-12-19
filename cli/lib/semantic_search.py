import numpy as np
import os
from typing import List
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from .search_utils import CACHE_DIR, load_movies

MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="mps")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def verify_model(self):
        print(f"Model loaded: {self.model}")
        print(f"Max sequence length: {self.model.max_seq_length}")

    def generate_embedding(self, text: str | List[str]) -> NDArray:
        if type(text) == str:
            text = [text]

        for t in text:
            if not t or not t.strip():
                raise ValueError("cannot generate embedding for empty text")

        emb = self.model.encode(text, show_progress_bar=True)
        return emb

    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        doc_strings = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            doc_string = f"{doc['title']}: {doc['description']}"
            doc_strings.append(doc_string)

        self.embeddings = self.generate_embedding(doc_strings)

        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(MOVIE_EMBEDDINGS_PATH, "wb") as f:
            np.save(f, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        if os.path.exists(MOVIE_EMBEDDINGS_PATH) == False:
            return self.build_embeddings(documents)

        self.embeddings = np.load(MOVIE_EMBEDDINGS_PATH)
        if len(self.embeddings) != len(documents):
            return self.build_embeddings(documents)

        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        return self.embeddings


def verify_model():
    model = SemanticSearch()
    model.verify_model()


def embed_text(text: str):
    model = SemanticSearch()
    emb = model.generate_embedding(text)
    emb: NDArray = emb[0]
    print(f"Text: {text}")
    print(f"First 3 dimensions: {emb[:3]}")
    print(f"Dimensions: {emb.shape[0]}")


def verify_embeddings():
    model = SemanticSearch()
    documents = load_movies()

    embeddings = model.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query: str):
    model = SemanticSearch()
    emb = model.generate_embedding(query)[0]

    print(f"Query: {query}")
    print(f"First 5 dimensions: {emb[:5]}")
    print(f"Shape: {emb.shape[0]}")
