from .search_utils import load_movies, preprocess_text
from typing import Dict, List
from .schemas import MovieModel
import pickle
from collections import defaultdict

INDEX_PATH = "cache/index.pkl"
DOCMAP_PATH = "cache/docmap.pkl"


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap: Dict[int, MovieModel] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = preprocess_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)

    def get_documents(self, term: str) -> List[int]:
        query_tokens = preprocess_text(term)
        hits = set()
        for token in query_tokens:
            hits.update(self.index.get(token, set()))

        return sorted(list(hits))

    def get_document_object(self, doc_id: int) -> MovieModel:
        try:
            return self.docmap[doc_id]
        except:
            raise ValueError(f"{doc_id} was not found")

    def build(self):
        movies = load_movies()
        for movie in movies:
            movie_desc = f"{movie.title} {movie.description}"
            self.__add_document(movie.id, movie_desc)
            self.docmap[movie.id] = movie

    def __write_file(self, file_path, object):
        with open(file_path, "wb") as file:
            file.write(object)

    def save(self):
        self.__write_file(INDEX_PATH, pickle.dumps(self.index))
        self.__write_file(DOCMAP_PATH, pickle.dumps(self.docmap))

    def load(self):
        try:
            with open(INDEX_PATH, "rb") as f:
                self.index = pickle.load(f)
        except:
            raise ValueError(f"{INDEX_PATH} not found")

        try:
            with open(DOCMAP_PATH, "rb") as f:
                self.docmap = pickle.load(f)
        except:
            raise ValueError(f"{DOCMAP_PATH} not found")
