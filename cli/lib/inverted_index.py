from .search_utils import load_movies, preprocess_text
from typing import Dict, List
from .schemas import MovieModel
import pickle
from collections import defaultdict, Counter

INDEX_PATH = "cache/index.pkl"
DOCMAP_PATH = "cache/docmap.pkl"
COUNTER_PATH = "cache/term_frequencies.pkl"


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap: Dict[int, MovieModel] = {}
        self.term_frequencies: Dict[int, Counter] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = preprocess_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id] = Counter(tokens)

    def get_documents(self, term: str) -> List[int]:
        hits = self.index.get(term, set())
        return sorted(list(hits))

    def get_document_object(self, doc_id: int) -> MovieModel:
        try:
            return self.docmap[doc_id]
        except:
            raise ValueError(f"{doc_id} was not found")

    def get_tf(self, doc_id, term):
        token = preprocess_text(term)
        if len(token) > 1:
            raise ValueError("Only one word is allowed for word count")
        token = token[0]
        return self.term_frequencies[doc_id][token]

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
        self.__write_file(COUNTER_PATH, pickle.dumps(self.term_frequencies))

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

        try:
            with open(COUNTER_PATH, "rb") as f:
                self.term_frequencies = pickle.load(f)
        except:
            raise ValueError(f"{COUNTER_PATH} not found")
