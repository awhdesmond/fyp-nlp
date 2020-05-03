from pydantic import BaseModel


class ListArticlesFilter(BaseModel):
    page_size: int
    page: int


class Article(BaseModel):
    id: str
    source: str
    url: str
    imageurl: str = None
    title: str
    author: str
    publishedDate: str
    content: str
