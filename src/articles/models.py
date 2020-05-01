from pydantic import BaseModel


class ListArticlesFilter(BaseModel):
    page_size: int
    page: int


class Article(BaseModel):

    _id: str
    source: str
    url: str
    image_url: str
    title: str
    author: str
    published_date: str
    content: str
