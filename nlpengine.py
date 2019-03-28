import spacy
import pydash
import collections
import string

from concurrent.futures import ThreadPoolExecutor

from cachetools import cached, LRUCache
from thesaurus import Word
from parseTree import ParseTree, ParseTreeNode

from textualEntailment import TextualEntailmentModel

cache = LRUCache(maxsize=1000)

SUBJECTS_COSINE_SIMILARITY_THRESHOLD = 0.6
SYNONYMS_FLAG = 0
ANTONYMS_FLAG = 1
NEUTRAL_FLAG  = 2

ENTAILMENT_INDEX = 0
CONTRADICTION_INDEX = 1
NEUTRAL_INDEX = 2

ENTAILMENT_THRESHOLD = 0.6
CONTRADICT_THRESHOLD = 0.6

class Claim():
    def __init__(self, spolt, score, claimer, sentence):
        self.spolt = spolt
        self.score = score
        self.claimer = claimer
        self.sentence = sentence
        
    def serialise(self):
        return {
            "score": self.score,
            "claimer": self.claimer,
            "sentence": self.sentence,
        }

    def __repr__(self):
        return self.sentence + " (" + str(self.score) + ")"

class NLPEngine(object):

    def __init__(self):
        self.nlp = spacy.load('en_core_web_lg')
        self.textEntModel = TextualEntailmentModel()
        self.textEntModel.createModel()

        

    def sanitizeText(self, text):
        return text.replace("”", "'") \
                    .replace("“", "'") \
                    .replace("’", "'") \
                    .replace("\"", "'") \
                    .replace("\n", "")

    def isPredicate(self, spacyToken):
        return spacyToken.pos_ == "VERB"

    def isNounFamily(self, spacyToken):
        return spacyToken.pos_ in ["NOUN", "PROPN", "NUM", "DET"]

    def removeFirstThat(self, sentence):
        tokens = sentence.split(" ")[1:]
        if len(tokens) > 0 and tokens[0].lower() == "that":
            return " ".join(tokens)
        else:
            return sentence

    def similarityScore(self, subA, subB):
        subADoc = self.nlp(subA)
        subBDoc = self.nlp(subB)
        return subADoc.similarity(subBDoc)

    @cached(cache)
    def areSynonyms(self, stringA, stringB, pos=None):
        if stringA == stringB:
            return True

        try:
            wordA = Word(stringA)
            wordB = Word(stringB)

            synonymsA = pydash.flatten_deep(wordA.synonyms('all', partOfSpeech=pos))
            synonymsB = pydash.flatten_deep(wordB.synonyms('all', partOfSpeech=pos))
            return stringA in synonymsB or stringB in synonymsA
        except:
            return False

    @cached(cache)
    def areAntonyms(self, stringA, stringB, pos=None):
        try:
            wordA = Word(stringA)
            wordB = Word(stringB)

            antonymsA = pydash.flatten_deep(wordA.antonyms('all', partOfSpeech=pos))
            antonymsB = pydash.flatten_deep(wordB.antonyms('all', partOfSpeech=pos))
            return stringA in antonymsB or stringB in antonymsA
        except:
            return False

    def areSimilarSubjects(self, subA, subB, alpha=SUBJECTS_COSINE_SIMILARITY_THRESHOLD):
        subADoc = self.nlp(subA)
        subBDoc = self.nlp(subB)

        # Make a semantic similarity estimate. 
        # The default estimate is cosine similarity using an average of word vectors.
        # Order of words does not matter
        return subADoc.similarity(subBDoc) >= alpha

    def areSimilarPredicates(self, predA, predB):
        predADoc = self.nlp(predA)
        predBDoc = self.nlp(predB)
        predALemma = predADoc[0].lemma_
        predBLemma = predBDoc[0].lemma_

        synonymsFlag = self.areSynonyms(predALemma, predBLemma)
        antonymsFlag = self.areAntonyms(predALemma, predBLemma)

        if synonymsFlag:
            return SYNONYMS_FLAG
        elif antonymsFlag:
            return ANTONYMS_FLAG
        return NEUTRAL_FLAG


    def isClaimerVerb(self, spacyToken):
        return spacyToken.lemma_ in ["say","announce", "answer", "assert", "claim","convey","declare","deliver","disclose","express","mention","reply","report","respond","reveal"]

    def mergeSpacySpansForDoc(self, spacyDoc):
        for span in list(spacyDoc.ents) + list(spacyDoc.noun_chunks):
            span.merge
            

    def extractWhatOtherPeopleClaim(self, spacyToken):
        parseTree = ParseTree(spacyToken)
        ccompChild = parseTree.root.retrieveChildren("ccomp")
        claimerNodes = parseTree.extractSubjectNodes(withCCAndConj=True)

        if len(ccompChild) > 0:
            saying = " ".join([x.text for x in list(ccompChild[0].innerToken.subtree)]).strip()

            subject = ""
            for node in claimerNodes:
                subject = subject + " " + node.innerToken.text
            subject = subject.strip()

            return {
                "saying": saying,
                "claimer": subject
            }
        return None

    def nlpProcessText(self, text, debug=False):
        textDoc = self.nlp(text)

        claims = []
        for spacySentence in textDoc.sents:
            sanitizedSentence = self.sanitizeText(spacySentence.text)
            sentenceDoc = self.nlp(sanitizedSentence)
            self.mergeSpacySpansForDoc(sentenceDoc)
            
            for spacyToken in sentenceDoc:
                if spacyToken.dep_ == "ROOT" and spacyToken.pos_ == "VERB":
                    
                    if self.isClaimerVerb(spacyToken):
                        saying = self.extractWhatOtherPeopleClaim(spacyToken)
                        if saying is None:
                            parseTree = ParseTree(spacyToken)
                            spolt = parseTree.extractData()
                            claim = Claim(spolt, 0, "", sanitizedSentence)
                            claims.append(claim)
                        else:
                            result = self.nlpProcessText(saying["saying"])
                            for claim in result:
                                claim.claimer = saying["claimer"]
                                claims.append(claim)
                            
                    else:
                        parseTree = ParseTree(spacyToken)
                        spolt = parseTree.extractData()
                        claim = Claim(spolt, 0, "", sanitizedSentence)
                        claims.append(claim)
        return claims


    def handleAnalyseQuery(self, query, relatedArticles):
        queryClaims = self.nlpProcessText(query)
        if len(queryClaims) <= 0:
            return None

        queryClaim = queryClaims[0]

        def _compareQueryWithRelatedArticle(article):
            article["evidence"] = {
                "entailment": [],
                "contradiction": [],
                "neutral": []
            }

            def filterHasSimilarField(field, queryClaim, relatedArticleClaims, drop=False):
                results = []
                for claim in relatedArticleClaims:
                    if claim.spolt[field] == "":
                        if drop:
                            continue
                        else:
                            results.append(claim)    
                    elif self.areSimilarSubjects(queryClaim.spolt[field], claim.spolt[field]):
                        results.append(claim)
                return results
        
            claims = self.nlpProcessText(article["content"])

            relatedClaims = filterHasSimilarField("subject", queryClaim, claims, drop=True)

            if queryClaim.spolt["object"] != "":
                relatedClaims = filterHasSimilarField("object", queryClaim, relatedClaims)
            
            if queryClaim.spolt["prepPobj"] != "":
                relatedClaims = filterHasSimilarField("prepPobj", queryClaim, relatedClaims)

            if len(relatedClaims) <= 0:
                return article

            for claim in relatedClaims:
                hypothesis = queryClaim.sentence
                premise = claim.sentence
                
                textualEntailmentResult = self.textEntModel.predict(premise, hypothesis)
                entailmentProb = round(textualEntailmentResult['label_probs'][ENTAILMENT_INDEX], 2)
                contradictProb = round(textualEntailmentResult['label_probs'][CONTRADICTION_INDEX], 2)
                neutralProb = round(textualEntailmentResult['label_probs'][NEUTRAL_INDEX], 2)

                claim.score = textualEntailmentResult['label_probs']
                if entailmentProb >= ENTAILMENT_THRESHOLD:
                    article["evidence"]["entailment"].append(claim.serialise())
                elif contradictProb >= CONTRADICT_THRESHOLD:
                    article["evidence"]["contradiction"].append(claim.serialise())
                else:
                    article["evidence"]["neutral"].append(claim.serialise())
            
            return article


        with ThreadPoolExecutor(max_workers=4) as executor:
            articlesWithEvidence = executor.map(_compareQueryWithRelatedArticle, relatedArticles)
            articlesWithEvidence = list(articlesWithEvidence)

        return articlesWithEvidence



