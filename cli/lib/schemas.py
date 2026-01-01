from pydantic import BaseModel
from typing import List, Dict


class Movie(BaseModel):
    id: int
    title: str
    description: str


class Movies(BaseModel):
    movies: List[Movie]
