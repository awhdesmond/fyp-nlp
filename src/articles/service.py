from articles import repository, models


class ArticleService:

    def __init__(self, repo: repository.ArticleRepository):
        self.repo = repo

    def list_articles(self, l_filter: models.ListArticlesFilter):
        """
        Retrieve articles based on the list filter

        Args:
            l_filter: filters used to list articles
        """
        return self.repo.list_articles(l_filter)

    def retrieve_article(self, article_id: str):
        """
        Retrieve an article based on the id

        Args:
            article_id: article id
        """
        return self.repo.retrieve_article(article_id)

    def find_related_articles(self, query: str):
        """
        Retrieve articles that are related to
        """
        return self.repo.find_related_articles(query)
