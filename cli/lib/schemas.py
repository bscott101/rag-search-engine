from pydantic import BaseModel
from typing import List, Optional


class Movie(BaseModel):
    id: int
    title: str
    description: str


class Movies(BaseModel):
    movies: List[Movie]


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
