import spacy
import pydash
import collections
import string

from concurrent.futures import ThreadPoolExecutor

from SPOExtractor import SPOExtractor
from textualEntailment import TextualEntailmentModel

ENTAILMENT_INDEX = 0
CONTRADICTION_INDEX = 1
NEUTRAL_INDEX = 2

ENTAILMENT_THRESHOLD = 0.5
CONTRADICT_THRESHOLD = 0.6

class NLPEngine(object):

    def __init__(self):
        self.spoltExtractor = SPOExtractor()
        self.textEntModel = TextualEntailmentModel()
        self.textEntModel.createModel()

    def _compareQueryWithRelatedArticle(self, queryClaim, article):
        article["evidence"] = {
            "entailment": [],
            "contradiction": [],
            "neutral": []
        }

        contentClaims = self.spoltExtractor.extractClaims(article["content"].strip()) 
        relatedClaims = []
        
        for claim in contentClaims:
            print(claim)
            print(claim.isRelatedSPOENT(queryClaim))
            if claim.isRelatedSPOENT(queryClaim):
                relatedClaims.append(claim)        


        print(relatedClaims)

        if len(relatedClaims) <= 0:
            return article

        for claim in relatedClaims:
            hypothesis = queryClaim.sentence
            premise = claim.sentence

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
                if entailmentProb >= ENTAILMENT_THRESHOLD:
                    article["evidence"]["entailment"].append(claim.serialise())

            if contradictProb > entailmentProb and contradictProb > neutralProb:
                if contradictProb >= CONTRADICT_THRESHOLD:
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

        for article in relatedArticles:
            result = self._compareQueryWithRelatedArticle(queryClaim, article)
            articlesWithEvidence.append(result)

        print("Returning")
        return articlesWithEvidence

            

        



