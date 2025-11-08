from pydantic import BaseModel


class MovieModel(BaseModel):
    id: int
    title: str
    description: str
