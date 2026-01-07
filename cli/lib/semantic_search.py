import numpy as np
import os
import re
import torch
import json
from typing import List, Tuple
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
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    format_search_result,
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
        if torch.cuda.is_available():
            return "cuda"
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

        self.embeddings = self.model.encode(doc_strings, show_progress_bar=True)

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

        query_emb = self.generate_embedding(query)[0]

        scores: List[Tuple[int, Movie]] = []
        for index, doc_emb in enumerate(self.embeddings):
            doc = self.documents[index]
            score = cosine_similarity(query_emb, doc_emb)
            scores.append((score, doc))

        scores.sort(key=lambda x: x[0], reverse=True)

        result = []
        for score, doc in scores[:limit]:
            result.append(
                {
                    "score": score,
                    "title": doc.title,
                    "description": doc.description,
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

        self.document_map = {}
        for doc in documents:
            self.document_map[doc.id] = doc

        all_chunks = []
        chunk_metadata = []

        for document in documents:
            doc_id = document.id
            self.document_map[doc_id] = document
            text = document.description
            if not text.strip():
                continue

            chunk_split = semantic_chunk(
                text,
                max_chunk_size=DEFAULT_SEMANTIC_CHUNK_SIZE,
                overlap=DEFAULT_CHUNK_OVERLAP,
            )
            for i, chunk in enumerate(chunk_split):
                all_chunks.append(chunk)
                chunk_metadata.append(
                    {
                        "movie_id": doc_id,
                        "chunk_idx": i,
                        "total_chunks": len(chunk_split),
                    }
                )

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.CHUNK_EMBEDDING_PATH, "wb") as f:
            np.save(f, self.chunk_embeddings)
        with open(self.CHUNK_METADATA_PATH, "w") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: List[Movie]) -> NDArray:
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc.id] = doc

        if os.path.exists(self.CHUNK_EMBEDDING_PATH) and os.path.exists(
            self.CHUNK_METADATA_PATH
        ):
            self.chunk_embeddings = np.load(self.CHUNK_EMBEDDING_PATH)
            with open(self.CHUNK_METADATA_PATH, "r") as f:
                data = json.load(f)
                self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
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
                    "movie_id": self.chunk_metadata[index]["movie_id"],
                    "score": score,
                }
            )

        document_scores = {}
        for chunk_score in chunk_scores:
            movie_id = chunk_score["movie_id"]
            if (
                movie_id not in document_scores
                or chunk_score["score"] > document_scores[movie_id]
            ):
                document_scores[movie_id] = chunk_score["score"]

        sorted_scores = sorted(
            document_scores.items(), key=lambda x: x[1], reverse=True
        )

        result = []
        for doc_id, score in sorted_scores[:limit]:
            doc = self.document_map[doc_id]
            res = format_search_result(
                doc_id,
                doc.title,
                doc.description[:DOCUMENT_PREVIEW_LENGTH],
                score=round(score, SCORE_PRECISION),
            )
            result.append(res)

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


def embed_chunks() -> NDArray:
    documents = load_movies()
    model = ChunkedSemanticSearch()
    return model.load_or_create_chunk_embeddings(documents)


def fixed_size_chunking(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    words = text.split()
    chunks = []

    n_words = len(words)
    i = 0
    while i < n_words:
        chunk_words = words[i : i + chunk_size]
        if chunks and len(chunk_words) <= overlap:
            break

        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap

    return chunks


def semantic_chunk(
    text: str,
    max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)

    if len(sentences) == 1 and not text.endswith((".", "!", "?")):
        sentences = [text]

    chunks = []
    i = 0
    n_sentences = len(sentences)

    while i < n_sentences:
        chunk_sentences = sentences[i : i + max_chunk_size]
        if chunks and len(chunk_sentences) <= overlap:
            break

        cleaned_sentences = []
        for chunk_sentence in chunk_sentences:
            cleaned_sentences.append(chunk_sentence.strip())
        if not cleaned_sentences:
            continue
        chunk = " ".join(cleaned_sentences)
        chunks.append(chunk)
        i += max_chunk_size - overlap

    return chunks


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


def semantic_chunk_search(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> dict:
    model = ChunkedSemanticSearch()
    documents = load_movies()
    model.load_or_create_chunk_embeddings(documents)

    results = model.search_chunks(query, limit)
    return {"query": query, "resutls": results}


def semantic_build():
    documents = load_movies()
    model = ChunkedSemanticSearch()
    model.load_or_create_embeddings(documents)
    model.load_or_create_chunk_embeddings(documents)
