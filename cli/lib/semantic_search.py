import numpy as np
import os
import re
from typing import List
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from .search_utils import (
    CACHE_DIR,
    load_movies,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)

MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="mps")
        self.embeddings: NDArray = None
        self.documents: List[dict] = None
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

    def search(self, query: str, limit: int = 5) -> List[dict]:
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError(
                "No embeddings loaded. Call 'load_or_create_embeddings' first."
            )

        if self.documents is None or len(self.documents) == 0:
            raise ValueError(
                "No documents loaded. Call 'load_or_create_embeddings' first."
            )

        emb = self.generate_embedding(query)[0]
        scores = []
        for index, doc_emb in enumerate(self.embeddings):
            doc = self.documents[index]
            score = cosine_similarity(emb, doc_emb)
            scores.append((score, doc))

        scores.sort(key=lambda x: x[0], reverse=True)
        result = []
        for score, doc in scores[:limit]:
            result.append(
                {
                    "score": score,
                    "title": doc["title"],
                    "description": doc["description"],
                }
            )

        return result


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


def cosine_similarity(vec1: NDArray[np.float32], vec2: NDArray[np.float32]) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def search_command(query: str, limit: int = 5) -> List[dict]:
    model = SemanticSearch()
    documents = load_movies()
    model.load_or_create_embeddings(documents)
    return model.search(query, limit)


def fixed_size_chunking(
    text: str, chunk_size: int = 200, overlap: int = 0
) -> List[str]:
    words = text.split(" ")
    chunks = []
    for index in range(0, len(words), chunk_size):
        start = index - overlap
        end = index + chunk_size + overlap
        if start < 0:
            start = index
        chunk_slice = words[start:end]

        res = ""
        for w in chunk_slice:
            res += w + " "
        chunks.append(res)

    return chunks


def semantic_chunk(text: str, max_chunk_size: int = 4, overlap: int = 0):
    chunks = re.split(r"(?<=[.!?])\s+", text)
    result = []
    i = 0
    n_sentences = len(chunks)
    while i < n_sentences:
        chunk_sentences = chunks[i : i + max_chunk_size]
        if chunks and len(chunk_sentences) <= overlap:
            break
        result.append(" ".join(chunk_sentences))
        i += max_chunk_size - overlap
    return result


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    chunk_method: str = "chunk",
):
    chunks = []
    match chunk_method:
        case "chunk":
            print(f"Chunking {len(text)} characters")
            chunks = fixed_size_chunking(text, chunk_size, overlap)
        case "semantic_chunk":
            print(f"Semantically chunking {len(text)} characters")
            chunks = semantic_chunk(text, chunk_size, overlap)

    for index, chunk in enumerate(chunks):
        print(f"{index + 1}. {chunk}")
