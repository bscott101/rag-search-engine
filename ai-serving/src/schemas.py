import json
from pydantic import BaseModel, ConfigDict

class GenerateContent(BaseModel):
    prompt: str
    system_prompt: str | None = "You are a helpful assistant"
    temp: float | None = 0.3
    max_new_tokens: int | None = 300
    input_image: str | None = None


class ClipSearch(BaseModel):
    input_image: str
    limit: int | None = 3


class Movie(BaseModel):
    id: int
    title: str
    description: str


class Movies(BaseModel):
    movies: list[Movie]


def load_movies(file_path: str) -> list[Movie]:
    with open(file_path, "r") as f:
        data = json.load(f)
        data = Movies(**data)
    return data.movies



