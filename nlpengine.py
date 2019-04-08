import spacy
import pydash
import collections
import string

from concurrent.futures import ThreadPoolExecutor

from SPOLTExtractor import SPOLTExtractor
from textualEntailment import TextualEntailmentModel

ENTAILMENT_INDEX = 0
CONTRADICTION_INDEX = 1
NEUTRAL_INDEX = 2

ENTAILMENT_THRESHOLD = 0.55
CONTRADICT_THRESHOLD = 0.55

class NLPEngine(object):

    def __init__(self):
        self.spoltExtractor = SPOLTExtractor()
        self.textEntModel = TextualEntailmentModel()
        self.textEntModel.createModel()

    def _compareQueryWithRelatedArticle(self, queryClaim, article):
        article["evidence"] = {
            "entailment": [],
            "contradiction": [],
            "neutral": []
        }

        titleClaims   = self.spoltExtractor.extractClaims(article["title"].strip())
        contentClaims = self.spoltExtractor.extractClaims(article["content"].strip())
        claims = titleClaims + contentClaims

        relatedClaims = []
        for c in claims:
            if c.isRelatedSPOENT(queryClaim):
                relatedClaims.append(c)

        if len(relatedClaims) <= 0:
            return article

        for claim in relatedClaims:
            if claim.spolt['object'] == "" and claim.spolt['prepPobj'] == "":
                continue

            hypothesis = queryClaim.subPredObj()
            premise = claim.subPredObj()

            textualEntailmentResult = self.textEntModel.predict(premise, hypothesis)
            entailmentProb = round(textualEntailmentResult[ENTAILMENT_INDEX], 2)
            contradictProb = round(textualEntailmentResult[CONTRADICTION_INDEX], 2)
            neutralProb = round(textualEntailmentResult[NEUTRAL_INDEX], 2)

            print(premise)
            print(hypothesis)
            print(entailmentProb)
            print(contradictProb)
            print(neutralProb)
            
            claim.score = textualEntailmentResult.tolist()

            if entailmentProb > contradictProb and entailmentProb > neutralProb:
                article["evidence"]["entailment"].append(claim.serialise())

            if contradictProb > entailmentProb and contradictProb > neutralProb:
                article["evidence"]["contradiction"].append(claim.serialise())

            if neutralProb > entailmentProb and neutralProb > contradictProb:
                article["evidence"]["neutral"].append(claim.serialise())    
        return article

    def handleAnalyseQuery(self, query, relatedArticles):
        queryClaims = self.spoltExtractor.extractClaims(query)
        if len(queryClaims) <= 0:
            return None

        queryClaim = queryClaims[0]
        articlesWithEvidence = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            def fn(article):
                return self._compareQueryWithRelatedArticle(queryClaim, article)
            articlesWithEvidence = executor.map(fn, relatedArticles)
            articlesWithEvidence = list(articlesWithEvidence)

            return articlesWithEvidence

        



