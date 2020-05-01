import falcon

from common import serializers
from articles import service, models, exceptions

class ArticleResource:

    def __init__(self, svc: service.ArticleService):
        self.svc = svc

    def on_get_collection(self, req, resp):
        """
        HTTP GET /api/articles?page=1&page_size=10

        Query Params:
            page (required): current page
            page_size (required): size of one page

        Return 400 if either query params are not present
        """
        page_size = req.get_param("page_size")
        page = req.get_param("page")

        l_filter = models.ListArticlesFilter(
            page=page,
            page_size=page_size,
        )

        articles = self.svc.list_articles(l_filter)

        resp.body = serializers.serialize_json([a.dict() for a in articles])
        resp.content_type = falcon.MEDIA_JSON

    def on_get(self, req, resp, id):
        """
        HTTP GET /api/articles/{id}

        Returns 404 if article with id not found
        """
        try:
            article = self.svc.retrieve_article(id)
        except exceptions.ArticleNotFoundException as e:
            raise falcon.HttpNotFound(str(e))

        resp.body = serializers.serialize_json(article.dict())
        resp.content_type = falcon.MEDIA_JSON
