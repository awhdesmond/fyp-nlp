import spacy
import pydash
import collections
import string

from cachetools import cached, LRUCache
from thesaurus import Word
from parseTree import ParseTree, ParseTreeNode

SpoltScore = collections.namedtuple('SpoltScore', 'spolt score article')
cache = LRUCache(maxsize=1000)

SYNONYMS_FLAG = 0
ANTONYMS_FLAG = 1
NEUTRAL_FLAG  = 2

class NLPEngine(object):

    def __init__(self):
        self.nlp = spacy.load('en_core_web_lg')
        

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
    
        spolts = []
        sayings = []
        
        for spacySentence in textDoc.sents:
            sanitizedSentence = self.sanitizeText(spacySentence.text)
            sentenceDoc = self.nlp(sanitizedSentence)
            
            for span in list(sentenceDoc.ents) + list(sentenceDoc.noun_chunks):
                span.merge()
                
            for spacyToken in sentenceDoc:
                if spacyToken.dep_ == "ROOT" and spacyToken.pos_ == "VERB":
                    if self.areSynonyms("see", spacyToken.lemma_, pos="verb"):
                        saying = self.extractWhatOtherPeopleClaim(spacyToken)
                        if saying is None:
                            parseTree = ParseTree(spacyToken)
                            spolts.append(parseTree.extractData())
                        else:
                            sayings.append(saying)
                    elif self.areSynonyms("say", spacyToken.lemma_, pos="verb"):
                        saying = self.extractWhatOtherPeopleClaim(spacyToken)
                        if saying is None:
                            parseTree = ParseTree(spacyToken)
                            spolts.append(parseTree.extractData())
                        else:
                            r = self.nlpProcessText(saying["saying"])
                            for c in r:
                                c["claimer"] = saying["claimer"]
                                spolts.append(c)

                    else:
                        parseTree = ParseTree(spacyToken)
                        spolts.append(parseTree.extractData())
                        
        return spolts

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
        
    def similarityScore(self, subA, subB):
        subADoc = self.nlp(subA)
        subBDoc = self.nlp(subB)
        return subADoc.similarity(subBDoc)

    def areSimilarSubjects(self, subA, subB, alpha=0.50):
        wordsA = subA.translate(str.maketrans('', '', string.punctuation)).split(" ")
        wordsB = subB.translate(str.maketrans('', '', string.punctuation)).split(" ")
        
        intersectionWords = pydash.arrays.intersection(wordsA, wordsB)
        for word in intersectionWords:
            pydash.arrays.pull(wordsA, word)
            pydash.arrays.pull(wordsB, word)
        
        if len(wordsA) == 0 or len(wordsB) == 0:
            return True
        
        wordsADoc = self.nlp(" ".join(wordsA))
        wordsBDoc = self.nlp(" ".join(wordsB))
        
        # Make a semantic similarity estimate. 
        # The default estimate is cosine similarity using an average of word vectors.
        # Order of words does not matter
        return wordsADoc.similarity(wordsBDoc) >= alpha

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


    # 1: Analyse the query first
    # 2: Analyse the related articles to retrieve their spolts
    # 3: Filter out based on subjects
    def handleAnalyseQuery(self, text, relatedArticles):
        relatedArticlesSpolts = []
        for article in relatedArticles:
            articleSpolts = self.nlpProcessText(article["content"])
            articleSpolts = [(articleSpolt, article) for articleSpolt in articleSpolts]
            relatedArticlesSpolts.extend(articleSpolts)
        
    
        querySpolts = self.nlpProcessText(text)
        if len(querySpolts) <= 0:
            return {
                "relatedArticles": relatedArticles
            }

        querySpolt = querySpolts[0]
        relatedSpoltScores = [SpoltScore(spolt=articleSpolt[0], score=[], article=articleSpolt[1]) for articleSpolt in relatedArticlesSpolts]

        def filterSimilarClaimsOfField(field, inputSpolt, spoltScores):
            results = []
            for spoltScore in spoltScores:
                if spoltScore.spolt[field] == "":
                    continue
                
                 
                if self.areSimilarSubjects(inputSpolt[field], spoltScore.spolt[field]):
                    spoltScore.score.append(self.similarityScore(inputSpolt[field], spoltScore.spolt[field])) 
                    newSpoltScore = SpoltScore(spolt=spoltScore.spolt, score=spoltScore.score, article=spoltScore.article)
                    results.append(newSpoltScore)
            
            return results

        def filterSimilarClaimsOfAction(inputSpolt, spoltScores):
            results = []
            for spoltScore in spoltScores:
                if spoltScore.spolt["action"] == "":
                    continue

                if self.areSimilarPredicates(inputSpolt["action"], spoltScores.spolt["action"]) == 0:
                    spoltScore.score.append(self.similarityScore(inputSpolt["action"], spoltScore.spolt["action"])) 
                    newSpoltScore = SpoltScore(spolt=spoltScore.spolt, score=spoltScore.score, article=spoltScore.article)
                    results.append(newSpoltScore)
            return results
       
        relatedSpoltScores = filterSimilarClaimsOfField("subject", querySpolt, relatedSpoltScores)
        if querySpolt["object"] != "":
            relatedSpoltScores = filterSimilarClaimsOfField("object", querySpolt, relatedSpoltScores)
        elif querySpolt["prepPobj"] != "":
            relatedSpoltScores = filterSimilarClaimsOfField("prepPobj", querySpolt, relatedSpoltScores)

        if querySpolt["action"] != "":
            relatedSpoltScores = filterSimilarClaimsOfAction(querySpolt, relatedSpoltScores)

        evidenceMap = {
            "supporting": [],
            "opposing": [],
            "neutral": [],
            "relatedArticles": relatedArticles
        }

        for spoltScore in relatedSpoltScores:
            finalScore = sum(spoltScore.score) / len(spoltScore.score)
            similarFlag = self.areSimilarPredicates(querySpolt["predicate"], spoltScore.spolt["predicate"])

            if similarFlag == SYNONYMS_FLAG:
                if (querySpolt["predicateInverse"] != "" and spoltScore.spolt["predicateInverse"] == "")\
                    or (querySpolt["predicateInverse"] == "" and spoltScore.spolt["predicateInverse"] != ""):
                        finalScore = -finalScore
                        newSpoltScore = SpoltScore(spolt=spoltScore.spolt, score=finalScore, article=spoltScore.article)
                        evidenceMap["opposing"].append(newSpoltScore)
                else:
                    newSpoltScore = SpoltScore(spolt=spoltScore.spolt, score=finalScore, article=spoltScore.article)
                    evidenceMap["supporting"].append(newSpoltScore)
            elif similarFlag == 1:
                newSpoltScore = SpoltScore(spolt=spoltScore.spolt, score=finalScore, article=spoltScore.article)
                evidenceMap["opposing"].append(newSpoltScore)
            else:
                newSpoltScore = SpoltScore(spolt=spoltScore.spolt, score=finalScore, article=spoltScore.article)
                evidenceMap["neutral"].append(newSpoltScore)            
    
        return evidenceMap









