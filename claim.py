from nlpmodels import nlp_lg
from nlpmodels import nlp_md
from nlpmodels import nlp_sm

SUBJECTS_COSINE_SIMILARITY_THRESHOLD = 0.75

class Claim():
    def __init__(self, spolt, ents, score, claimer, sentence):
        self.spolt = spolt
        self.ents = ents
        self.score = score
        self.claimer = claimer
        self.sentence = sentence
        
    def serialise(self):
        return {
            "score": self.score,
            "claimer": self.claimer,
            "sentence": self.sentence,
            "spolt": self.spolt,
            "ents": self.ents
        }
    
    def cosineSimilarity(self, subA, subB):
        subADoc = nlp_lg(subA)
        subBDoc = nlp_lg(subB)
        return subADoc.similarity(subBDoc)

    def subPredObj(self):
        result = self.spolt["subject"]
        
        if self.spolt["predicateInverse"] != "":
            result = result + " " + self.spolt["predicateInverse"]
            
        if self.spolt["predicate"] != "":
            result = result + " " + self.spolt["predicate"]
        
        if self.spolt["object"] != "":
            result = result + " " + self.spolt["object"]
            
        return result
    
    def spoltString(self, ppFlag=False):
        result = self.spolt["subject"]
        
        if self.spolt["predicateInverse"] != "":
            result = result + " " + self.spolt["predicateInverse"]
            
        if self.spolt["predicate"] != "":
            result = result + " " + self.spolt["predicate"]
        
        if self.spolt["object"] != "":
            result = result + " " + self.spolt["object"]
            
        if self.spolt["action"] != "":
            result = result + " " + self.spolt["action"]
            
        if self.spolt["prepPobj"] != "" and ppFlag:
            result = result + " " + " ".join(self.spolt["prepPobj"])
            
        return result
        
    def isRelatedPure(self, other):
        return self.cosineSimilarity(self.sentence, other.sentence) > SUBJECTS_COSINE_SIMILARITY_THRESHOLD

    def isRelatedSPO(self, other):
        score1 = self.cosineSimilarity(self.spoltString(), other.spoltString())
        score2 = self.cosineSimilarity(self.spoltString(ppFlag=True), other.spoltString())
        score3 = self.cosineSimilarity(self.spoltString(), other.spoltString(ppFlag=True))
        score4 = self.cosineSimilarity(self.spoltString(ppFlag=True), other.spoltString(ppFlag=True))
        return max([score1, score2, score3, score4]) >  SUBJECTS_COSINE_SIMILARITY_THRESHOLD

    
    def isRelatedSPOENT(self, other, threshold=SUBJECTS_COSINE_SIMILARITY_THRESHOLD):        
        entsKeys = [
            "PERSON",
            "NORP",
            "ORG",
            "GPE",
            "LOC",
            "PRODUCT",
            "EVENT",
            "MONEY",
        ]
        
        scores = {}
        totalScore = 0
        for idx, key in enumerate(entsKeys):            
            selfEntString = " ".join(self.ents[key])
            otherEntString = " ".join(other.ents[key])
            scores[key] = self.cosineSimilarity(selfEntString, otherEntString)
            totalScore += scores[key]
        
        gpeScore = scores["GPE"]
        if gpeScore < 0.65:
            return False
        
        locScore = scores["LOC"]
        if locScore < 0.5:
            return False

        if self.isRelatedSPO(other) < 0.5:
            return False
        
        avgScore = (totalScore + self.isRelatedSPO(other)) / (len(entsKeys) + 1)
        return (avgScore > threshold)
    
        
    def __repr__(self):
        return self.sentence + " (" + str(self.score) + ")"