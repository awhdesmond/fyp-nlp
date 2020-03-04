import spacy
import pydash

from claim import Claim
from parseTree import ParseTree, ParseTreeNode

import datetime
from natty import DateParser

from nlpmodels import nlp_lg
from nlpmodels import nlp_md
from nlpmodels import nlp_sm 

class SPOExtractor(object):

    def sanitizeText(self, text):
        return text.replace("”", "'") \
                    .replace("“", "'") \
                    .replace("’", "'") \
                    .replace("\"", "'") \
                    .replace("\n", "") \
                    .replace(u'\u200b', ' ').strip()

    def isRootPredicate(self, spacyToken):
        return spacyToken.pos_ == "VERB" and spacyToken.dep_ == "ROOT"

    def isClaimerVerb(self, spacyToken):
        return spacyToken.lemma_ in ["say","announce", "answer", "assert", "claim","convey","declare","deliver","disclose","express","mention","reply","report","respond","reveal"]
    
    def mergeSpacySpansForDoc(self, spacyDoc):
        for span in list(spacyDoc.ents) + list(spacyDoc.noun_chunks):
            span.merge()

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

    def extractSentenceEntities(self, sentence):
        entMap = {
            "PERSON": [],
            "NORP": [],
            "ORG": [],
            "GPE": [],
            "LOC": []
        }
        
        sDoc = nlp_lg(sentence)
        for t in sDoc:
            if t.ent_type_ in entMap.keys():
                entMap[t.ent_type_].append(t.text)
        return entMap

    
    def extractTimeRange(self, sentence):
        dp = DateParser(sentence)
        matches = dp.result()

        if matches == None:
            return {
                "low": datetime.datetime.now(),
                "high": datetime.datetime.now()
            }
        
        matches.sort()
        
        if len(matches) == 1:
            return {
                "low": matches[0],
                "high": matches[0]
            }
        
        return {
            "low": matches[0],
            "high": matches[-1]
        } 
    
    def extractClaims(self, text, debug=False):
        textDoc = nlp_lg(text)

        claims = []

        for spacySentence in textDoc.sents:
            sanitizedSentence = self.sanitizeText(spacySentence.text)
            isParsed = False
            
            for nlp in [nlp_lg, nlp_md, nlp_sm]:
                if isParsed:
                    break
                    
                sentenceDoc = nlp(sanitizedSentence)
                self.mergeSpacySpansForDoc(sentenceDoc)

                for spacyToken in sentenceDoc:
                    if spacyToken.dep_ == "ROOT" and spacyToken.pos_ == "VERB":
                        isParsed = True
                        
                        if self.isClaimerVerb(spacyToken):
                            saying = self.extractWhatOtherPeopleClaim(spacyToken)
                            if saying is None:
                                parseTree = ParseTree(spacyToken)
                                spo = parseTree.extractData()
                                if spo != None:
                                    entMap = self.extractSentenceEntities(sanitizedSentence)
                                    #timerange = self.extractTimeRange(sanitizedSentence)
                                    claim = Claim(spo, entMap, None, 0, sanitizedSentence)
                                    claims.append(claim)
                            else:
                                result = self.extractClaims(saying["saying"])
                                for claim in result:
                                    claim.claimer = saying["claimer"]
                                    claims.append(claim)            
                        else:
                            parseTree = ParseTree(spacyToken)
                            spo = parseTree.extractData()
                            if spo != None:    
                                entMap = self.extractSentenceEntities(sanitizedSentence)
                                #timerange = self.extractTimeRange(sanitizedSentence)
                                claim = Claim(spo, entMap, None, 0, sanitizedSentence)
                                claims.append(claim)
        return claims
        
