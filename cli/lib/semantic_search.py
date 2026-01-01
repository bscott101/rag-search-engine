import numpy as np
import os
import re
import torch
import json
from typing import List
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from .schemas import Movies, Movie
from .search_utils import (
    CACHE_DIR,
    load_movies,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    SCORE_PRECISION,
    DOCUMENT_PREVIEW_LENGTH,
    DEFAULT_SEARCH_LIMIT,
)

MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.device = self.__device()
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embeddings: NDArray = None
        self.documents: List[Movie] = None
        self.document_map: dict[int, Movie] = {}

    def __device(self) -> str:
        if torch.mps.is_available():
            return "mps"
        return "cpu"

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

    def build_embeddings(self, documents: list[Movie]):
        self.documents = documents
        doc_strings = []
        for doc in documents:
            self.document_map[doc.id] = doc
            doc_string = f"{doc.title}: {doc.description}"
            doc_strings.append(doc_string)

        self.embeddings = self.generate_embedding(doc_strings)

        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(MOVIE_EMBEDDINGS_PATH, "wb") as f:
            np.save(f, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[Movie]):
        if os.path.exists(MOVIE_EMBEDDINGS_PATH) == False:
            return self.build_embeddings(documents)

        self.embeddings = np.load(MOVIE_EMBEDDINGS_PATH)
        if len(self.embeddings) != len(documents):
            return self.build_embeddings(documents)

        self.documents = documents
        for doc in documents:
            self.document_map[doc.id] = doc
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


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings: NDArray = None
        self.chunk_metadata: List[dict] = None
        self.CHUNK_EMBEDDING_PATH = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
        self.CHUNK_METADATA_PATH = os.path.join(CACHE_DIR, "chunk_metadata.json")

    def build_chunk_embeddings(self, documents: List[Movie]) -> NDArray:
        self.documents = documents
        all_chunks = []
        chunk_metadata = []
        for document in documents:
            doc_id = document.id
            self.document_map[doc_id] = document
            if len(document.description) == 0:
                continue

            chunk_split = semantic_chunk(
                document.description, max_chunk_size=4, overlap=1
            )
            for chunk in chunk_split:
                all_chunks.append(chunk)
                chunk_metadata.append(
                    {
                        "movie_idx": doc_id,
                        "chunk_idx": len(all_chunks) - 1,
                        "total_chunks": len(chunk_split),
                    }
                )
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.CHUNK_EMBEDDING_PATH, "wb") as f:
            np.save(f, self.chunk_embeddings)
        with open(self.CHUNK_METADATA_PATH, "w") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: List[Movie]) -> NDArray:
        if os.path.exists(self.CHUNK_EMBEDDING_PATH) == False:
            return self.build_chunk_embeddings(documents)

        if os.path.exists(self.CHUNK_METADATA_PATH) == False:
            return self.build_chunk_embeddings(documents)

        self.chunk_embeddings = np.load(self.CHUNK_EMBEDDING_PATH)
        with open(self.CHUNK_METADATA_PATH, "r") as f:
            metadata = json.load(f)
            self.chunk_metadata = metadata["chunks"]

        self.documents = documents
        for doc in documents:
            self.document_map[doc.id] = doc
        return self.chunk_embeddings

    def search_chunks(self, query: str, limit: int = 10) -> dict:
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError(
                "No chunk embeddings loaded. Call load_or_create_chunk_embeddings"
            )

        chunk_scores = []
        search_emb = self.generate_embedding(query)[0]
        for index, chunk in enumerate(self.chunk_embeddings):
            score = cosine_similarity(search_emb, chunk)
            chunk_scores.append(
                {
                    "chunk_idx": index,
                    "movie_idx": self.chunk_metadata[index]["movie_idx"],
                    "score": score,
                }
            )

        document_scores = {}
        for score in chunk_scores:
            doc_index = score["movie_idx"]
            if (
                doc_index not in document_scores
                or score["score"] > document_scores[doc_index]
            ):
                document_scores[doc_index] = score["score"]
        sorted_scores = sorted(
            document_scores.items(), key=lambda x: x[1], reverse=True
        )

        result = []
        for doc_id, score in sorted_scores[:limit]:
            doc = self.document_map[doc_id]
            result.append(
                {
                    "id": doc_id,
                    "title": doc.title,
                    "document": doc.description[:DOCUMENT_PREVIEW_LENGTH],
                    "score": round(score, SCORE_PRECISION),
                }
            )

        return {"query": query, "result": result}


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


def embed_chunks() -> NDArray:
    documents = load_movies()
    model = ChunkedSemanticSearch()
    return model.load_or_create_chunk_embeddings(documents)


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


def semantic_chunk(text: str, max_chunk_size: int = 4, overlap: int = 0) -> List[str]:
    strip_text = text.strip()
    if not strip_text:
        return []

    chunks = re.split(r"(?<=[.!?])\s+", strip_text)
    if len(chunks) == 1 and not text.endswith((".", "!", "?")):
        chunks = [text]

    result = []
    i = 0
    n_sentences = len(chunks)
    while i < n_sentences:
        chunk_sentences = chunks[i : i + max_chunk_size]
        if chunks and len(chunk_sentences) <= overlap:
            break

        cleaned_sentences = []
        for chunk_sentence in chunk_sentences:
            cleaned_sentences.append(chunk_sentence.strip())
        if not cleaned_sentences:
            continue

        result.append(" ".join(cleaned_sentences))
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


def semantic_chunk_search(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    model = ChunkedSemanticSearch()
    documents = load_movies()
    model.load_or_create_chunk_embeddings(documents)

    return model.search_chunks(query, limit)
