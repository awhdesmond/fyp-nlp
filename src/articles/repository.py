import elasticsearch
from articles import models, exceptions


MIN_RELATED_SCORE = 30


class ArticleRepository:
    """
    Article store powered by Elasticsearch

    Attributes:
        es_endpoint: elasticsearch endpoint
        client: elasticsearch client
    """

    def __init__(self, es_endpoint: str):
        self.es_endpoint = es_endpoint
        self.client = elasticsearch.Elasticsearch([self.es_endpoint])

    def list_articles(self, l_filter: models.ListArticlesFilter):
        """
        Retrieve articles based on the list filter

        Args:
            l_filter: filters used to list articles
        """

        page_start = l_filter.page * l_filter.page_size
        search_body = {
            "query": {"match_all": {}},
            "sort": {"publishedDate": {"order": "desc"}}
        }

        results = self.client.search(
            index="articles",
            doc_type="_doc",
            body=search_body,
            from_=page_start,
            size=l_filter.page_size
        )
        docs = results.get("hits", {}).get("hits", [])

        articles = []
        for doc in docs:
            doc["_source"]["id"] = doc["_id"]
            articles.append(models.Article(**doc["_source"]))

        return articles

    def retrieve_article(self, article_id: str):
        """
        Retrieve an article based on the id

        Args:
            article_id: article id
        """

        search_body = {
            "query": {"match": {"_id": article_id}},
        }

        results = self.client.search(
            index="articles",
            doc_type="_doc",
            body=search_body
        )

        if not results.get("hits", {}).get("hits", []):
            raise exceptions.ArticleNotFoundException(f"not found: {article_id}")

        doc = results.get("hits", {}).get("hits", [])[0]
        doc["_source"]["id"] = doc["_id"]
        return models.Article(**doc["_source"])

    def find_related_articles(self, query: str, min_score=MIN_RELATED_SCORE):
        """
        Retrieve articles that are related to the query

        Args:
            query: query to find the articles for
            min_score: min score to be considered related

        Returns a list or articles
        """

        search_body = {
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "fields": ["title", "content"],
                            "query": query,
                            "minimum_should_match": "30%",
                        }
                    },
                    "should": [
                        {
                            "match": {
                                "title.shingles": query
                            }
                        },
                        {
                            "match": {
                                "content.shingles": query
                            }
                        },
                        {
                            "match_phrase": {
                                "title": {
                                    "query": query,
                                    "boost": 5,
                                    "slop": 5,
                                }
                            }
                        },
                        {
                            "match_phrase": {
                                "content": {
                                    "query": query,
                                    "boost": 3,
                                    "slop": 50,
                                }
                            }
                        }
                    ]
                }
            }
        }

        results = self.client.search(
            index="articles",
            doc_type="_doc",
            body=search_body,
            size=5,
        )

        docs = [
            d for d in results.get("hits", {}).get("hits", [])
            if d["_score"] > min_score
        ]

        articles = []
        for doc in docs:
            doc["_source"]["id"] = doc["_id"]
            articles.append(models.Article(**doc["_source"]))

        return articles
