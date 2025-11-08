import json
from nltk.stem import PorterStemmer
import string

from typing import List
from .schemas import MovieModel

DATA_PATH_MOVIES = "data/movies.json"
DATA_PATH_STOP_WORDS = "data/stopwords.txt"
DEFAULT_SEARCH_LIMIT = 5


def load_movies() -> List[MovieModel]:
    data = json.loads(open(DATA_PATH_MOVIES, "rb").read())
    data = [MovieModel(**x) for x in data["movies"]]
    return data


def load_stop_words() -> set:
    data = open(DATA_PATH_STOP_WORDS, "r").read()
    return set(data.splitlines())


def filter_stop_words(query_tokens: List[str]) -> List[str]:
    stop_words = load_stop_words()
    tokens = []
    for token in query_tokens:
        if token not in stop_words:
            tokens.append(token)

    return tokens


def preprocess_text(text: str) -> List[str]:
    stemmer = PorterStemmer()

    text = text.lower()
    trans_table = str.maketrans("", "", string.punctuation)
    text = text.translate(trans_table)

    tokens = text.split(" ")
    tokens = [x for x in tokens if not None]
    tokens = filter_stop_words(tokens)

    return [stemmer.stem(x) for x in tokens]
