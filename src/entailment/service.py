
from typing import List, Dict
from pydantic import BaseModel

import articles
from entailment import exceptions
from entailment.nlp import engine, claim


class ArticleEvidence(BaseModel):
    """
    Wrapper class to hold entailment evidence for an article
    """

    article: Dict
    entailment: List[claim.Claim] = []
    contradiction: List[claim.Claim] = []
    neutral: List[claim.Claim] = []

    def is_empty(self):
        return all([
            self.entailment.count() == 0,
            self.contradiction.count() == 0,
            self.neutral.count() == 0,
        ])


class EntailmentService:
    """EntailmentService evaluates a entailment relationship
    for an input claim by matching claim with news articles

    Attributes:
        entailment_engine: module for analaysing entailment relationship
        article_svc: for retrieving related articles
    """

    def __init__(
        self,
        claim_analyser: claim.ClaimSimilarityAnalyser,
        claim_extractor: claim.ClaimExtractor,
        entailment_engine: engine.EntailmentEngine,
        article_svc: articles.service.ArticleService
    ):
        self.entailment_engine = entailment_engine
        self.article_svc = article_svc

    def entailment_query(self, query: str):
        """
        Retrieves related articles for a query text and
        run entailment model against it to find supporting,
        contradicting or neural evidence.

        Args:
            query: query text by the user

        Returns evidence of articles for the query text
        Raises NoClaimException if query text does not contain a claims
        """

        # 1. Attempt to extract claim from query text
        claims = self.claim_extractor.extract_claims(query)
        if not claims:
            raise exceptions.NoClaimException(f"no claims found for: {query}")

        # 2. Find the related articles for the query text
        query_claim = claims[0]
        related_articles = self.article_svc.find_related_articles(query)

        # 3. Generate entailment evidences from related article claims
        evidences = []
        for article in related_articles:
            evidence = ArticleEvidence(article=article)

            content = article["content"]
            article_claims = self.claim_extractor.extract_claims(content)
            related_claims = [
                c for c in article_claims
                if self.claim_analyser.is_related(c, query_claim)
            ]

            # We are only interested in claims that are related to each other
            for related_claim in related_claims:
                hypothesis = query_claim.sentence
                premise = related_claim.sentence
                pred, score = self.entailment_engine.predict(hypothesis, premise)

                if not pred:
                    continue

                related_claim.score = score
                if pred == engine.EntailmentEngine.ENTAILMENT_INDEX:
                    evidence.entailment.append(related_claim)
                if pred == engine.EntailmentEngine.CONTRADICTION_INDEX:
                    evidence.contradiction.append(related_claim)
                if pred == engine.EntailmentEngine.NEUTRAL_INDEX:
                    evidence.neutral.append(related_claim)

            if not evidence.is_empty():
                evidences.append(evidence)

        return evidences
