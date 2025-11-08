from .search_utils import load_movies, preprocess_text
from typing import Dict, List
from .schemas import MovieModel
import pickle

INDEX_PATH = "cache/index.pkl"
DOCMAP_PATH = "cache/docmap.pkl"


class InvertedIndex:
    def __init__(self):
        self.index: Dict[str, List[int]] = {}
        self.docmap: Dict[int, MovieModel] = {}

    def __add_document(self, doc_id: int, text: str):
        tokens = preprocess_text(text)
        for token in tokens:
            try:
                self.index[token].append(doc_id)
            except:
                self.index[token] = [doc_id]

    def get_documents(self, term: str):
        query_tokens = preprocess_text(term)
        for token in query_tokens:
            try:
                hits = self.index[token]
            except:
                raise ValueError(f"{token} is not in db")
        return sorted(hits)

    def build(self):
        movies = load_movies()
        for movie in movies:
            self.__add_document(movie.id, f"{movie.title} {movie.description}")
            self.docmap[movie.id] = movie

    def __write_file(self, file_path, object):
        with open(file_path, "wb") as file:
            file.write(object)

    def save(self):
        self.__write_file(INDEX_PATH, pickle.dumps(self.index))
        self.__write_file(DOCMAP_PATH, pickle.dumps(self.docmap))

    def load(self):
        with open(INDEX_PATH, "rb") as f:
            self.index = pickle.load(f)
        with open(DOCMAP_PATH, "rb") as f:
            self.docmap = pickle.load(f)
