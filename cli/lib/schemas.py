from pydantic import BaseModel
from typing import List, Optional


class Movie(BaseModel):
    id: int
    title: str
    description: str


class Movies(BaseModel):
    movies: List[Movie]

class GenerateContent(BaseModel):
    prompt: str
    system_prompt: str | None = "You are a helpful assistant"
    temp: float | None = 0.3
    max_new_tokens: int | None = 300
    input_image: str | None = None


class FormattedResults(BaseModel):
    doc_id: int
    title: str
    document: str
    score: float
    norm_score: Optional[float] = None
    rff_score: Optional[float] = 0.0
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    semantic_score: Optional[float] = None
    semantic_rank: Optional[int] = None
    hybird_score: Optional[float] = None
