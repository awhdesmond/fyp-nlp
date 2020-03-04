import json

from nlpmodels import nlp_lg
from nlpmodels import nlp_md
from nlpmodels import nlp_sm 
from nlpmodels import rf

class Claim():
    
    def __init__(self, spo, ents, timerange, score, sentence):
        self.spo       = spo
        self.ents      = ents
        self.timerange = timerange
        self.score     = score
        self.sentence  = sentence
        
    def serialise(self):
        data = {
            "score": self.score,
            "sentence": self.sentence,
            "spo": self.spo,
            "ents": self.ents
        }
        return data
        

    def cosineSimilarity(self, subA, subB):
        subADoc = nlp_lg(subA)
        subBDoc = nlp_lg(subB)
        return subADoc.similarity(subBDoc)
    

    def extractNotNamedEntitiesNouns(self):
        sDoc = nlp_lg(self.sentence)
        nouns = []
        
        for token in sDoc:
            if (token.pos_ in ["NOUN", "PROPN", "NUM"]) and token.ent_type_ == "":
                nouns.append(token.text)
                
        return " ".join(nouns)
    
    def extractAdjectives(self):
        sDoc = nlp_lg(self.sentence)
        adjs = []
        
        for token in sDoc:
            if (token.pos_ in ["ADJ"]) and token.ent_type_ == "":
                adjs.append(token.text)
                
        return " ".join(adjs)
    
    def extractEnts(self):
        entsKeys = [
            "PERSON",
            "NORP",
            "ORG",
            "GPE",
            "LOC",
        ]
        
        entString = ""
        for idx, key in enumerate(entsKeys):            
            entString += " " + " ".join(self.ents[key])

        return entString
    
    def extractVerbs(self):
        sDoc = nlp_lg(self.sentence)
        verbs = []
        
        for token in sDoc:
            if token.pos_ == "VERB":
                verbs.append(token.text)
                
        return " ".join(verbs)
        
    def generateFeatureVector(self, other):
        
        gpeCosine      = self.cosineSimilarity(" ".join(self.ents["GPE"]), " ".join(other.ents["GPE"]))
        norpCosine     = self.cosineSimilarity(" ".join(self.ents["NORP"]), " ".join(other.ents["NORP"]))
        personCosine   = self.cosineSimilarity(" ".join(self.ents["PERSON"]), " ".join(other.ents["PERSON"]))
        orgCosine      = self.cosineSimilarity(" ".join(self.ents["ORG"]), " ".join(other.ents["ORG"]))
        entsCosine     = self.cosineSimilarity(self.extractEnts(), other.extractEnts())
        nounsCosine    = self.cosineSimilarity(self.extractNotNamedEntitiesNouns(), other.extractNotNamedEntitiesNouns())
        adjsCosine     = self.cosineSimilarity(self.extractAdjectives(), other.extractAdjectives())
        verbCosine     = self.cosineSimilarity(self.extractVerbs(), other.extractVerbs())
        sentenceCosine = self.cosineSimilarity(self.sentence, other.sentence)
    
        return [gpeCosine, entsCosine, nounsCosine, verbCosine, sentenceCosine]
    

    def spoString(self, ppFlag=False):
        result = self.spo["subject"]
        
        if self.spo["predicateInverse"] != "":
            result = result + " " + self.spo["predicateInverse"]
            
        if self.spo["predicate"] != "":
            result = result + " " + self.spo["predicate"]
        
        if self.spo["object"] != "":
            result = result + " " + self.spo["object"]
            
        if self.spo["action"] != "":
            result = result + " " + self.spo["action"]
            
        if self.spo["prepPobj"] != "" and ppFlag:
            result = result + " " + " ".join(self.spo["prepPobj"])
            
        return result
        
    
    def isRelatedPure(self, other, threshold):
        return self.cosineSimilarity(self.sentence, other.sentence) >= threshold
    

    def isRelatedMagicSPOENT(self, other, time=False, threshold=0.5):        
        if time:
            overlap = (self.timerange["high"] >= other.timerange["low"]) or (self.timerange["low"] <= other.timerange["high"])
            if not overlap:
                return False
              
        entScore = self.cosineSimilarity(self.extractEnts(), other.extractEnts())
        
        if entScore < 0.55:
            return False
                
        nounScore = self.cosineSimilarity(self.extractNotNamedEntitiesNouns(), other.extractNotNamedEntitiesNouns())
        
        return nounScore >= 0.6
        
    def isRelatedSPOENT(self, other):
        fv = self.generateFeatureVector(other)
        return rf.predict([fv])[0] == 1


    def __repr__(self):
        return self.sentence + " (" + str(self.score) + ")"