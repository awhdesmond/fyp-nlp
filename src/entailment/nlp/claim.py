from itertools import chain
from typing import Dict

import spacy
from pydantic import BaseModel
from entailment.nlp import pos_tree

class Claim(BaseModel):
    """
    Represents a natural sentence with SPOL metadata
    """
    entities: Dict
    score: str
    sentence: str


class ClaimSimilarityAnalyser:
    """
    Analyse claims similarity using logreg and cosine similarity

    Attributes:
        nlp (spacy.model)
        logreg (scikit-learn model): logistic regression model
    """

    def __init__(self, nlp, logreg):
        self.nlp = nlp
        self.logreg = logreg

    def consine_similarity(self, sentence_a: str, sentence_b: str):
        doc_a = self.nlp(sentence_a)
        doc_b = self.nlp(sentence_b)

        return doc_a.similarity(doc_b)

    def extract_not_named_entities_nouns(self, claim: Claim):
        """
        Extract entities that aren't recognised by the model's
        named entity recognition

        Returns a string made up of the nouns
        """
        s_doc = self.nlp(claim.sentence)

        nouns = [
            t.text for t in s_doc
            if t.pos_ in ["NOUN", "PROPN", "NUM"] and t.ent_type_ == ""
        ]
        return " ".join(nouns)

    def extract_entities(self, claim: Claim):
        """
        Extract entities that are recognised by the model's
        named entity recognition

        Returns a string made up of the entities
        """

        entity_keys = [
            "PERSON",
            "NORP",
            "ORG",
            "GPE",
            "LOC"
        ]

        entities = [
            claim.entities[key] for key in entity_keys
        ]
        return " ".join(chain.from_iterable(entities))

    def extract_verbs(self, claim: Claim):
        """
        Extract verbs from the claim's sentence

        Returns a string made up of the verbs
        """
        s_doc = self.nlp(claim.sentence)
        verbs = [t.text for t in s_doc if t.pos_ == "VERB"]
        return " ".join(verbs)

    def extract_adjectives(self, claim: Claim):
        """
        Extract adjectives that exists in the claim

        Returns a string made up of the adjectives
        """
        s_doc = self.nlp(claim.sentence)

        adjs = [
            t.text for t in s_doc
            if t.pos_ in ["ADJ"] and t.ent_type_ == ""
        ]
        return " ".join(adjs)

    def generate_feature_vector(self, claim_one: Claim, claim_two: Claim):
        """
        Generate feature vector for the log reg model using cosine similarity
        for the different entities, verbs and nouns

        Returns a list of floats (i.e feature values)
        """
        gpe_cosine = self.consine_similarity(
            " ".join(claim_one.entities["GPE"]),
            " ".join(claim_two.entities["GPE"])
        )

        entities_cosine = self.consine_similarity(
            self.extract_entities(claim_one),
            self.extract_entities(claim_two)
        )

        nouns_cosine = self.consine_similarity(
            self.extract_not_named_entities_nouns(claim_one),
            self.extract_not_named_entities_nouns(claim_two)
        )
        verb_cosine = self.consine_similarity(
            self.extract_verbs(claim_one),
            self.extract_verbs(claim_two)
        )
        sentence_cosine = self.consine_similarity(
            claim_one.sentence,
            claim_two.sentence
        )

        return [gpe_cosine, entities_cosine, nouns_cosine, verb_cosine, sentence_cosine]

    def is_related(self, claim_one, claim_two):
        """
        Returns if two claims are related
        """
        fv = self.generate_feature_vector(claim_one, claim_two)
        return self.logreg.predict([fv])[0] == 1


class ClaimExtractor:
    """
    Extracts SPO claims from natural sentences

    Attributes:
        nlp (spacy.model)
    """

    def __init__(self, nlp):
        self.nlp = nlp

    def sanitext_text(self, text: str):
        """
        Cleans the input text and remove unparsable characters
        """
        sanitise_map = {
            "”": "'",
            "“": "'",
            "’": "'",
            "\"": "'",
            "\n": "",
            u"\u200b": ""
        }

        char_list = [c for c in text.strip()]
        sanitized_list = [
            sanitise_map[c] if c in sanitise_map else c
            for c in char_list
        ]

        return "".join(sanitized_list).strip()

    def is_root_predicate(self, token: spacy.tokens.token.Token):
        """
        Returns if a token is the root predicate of a sentence
        """
        return token.pos_ == "VERB" and token.dep_ == "ROOT"

    def is_claimer_verb(self, token: spacy.tokens.token.Token):
        claimer_verbs = set([
            "say",
            "announce",
            "answer",
            "assert",
            "claim",
            "convey",
            "declare",
            "deliver",
            "disclose",
            "express",
            "mention",
            "reply",
            "report",
            "respond",
            "reveal"
        ])

        return token.lemma_ in claimer_verbs

    def merge_spans_in_doc(self, doc: spacy.tokens.Doc):
        """
        Merge token spans in a doc
        """
        spans = list(doc.ents) + list(doc.noun_chunks)
        for span in spans:
            span.merge()

    def extract_sentence_entities(self, sentence: str):
        """
        Extracts the named entities present in the sentence
        """
        entities_map = {
            "PERSON": [],
            "NORP": [],
            "ORG": [],
            "GPE": [],
            "LOC": []
        }

        s_doc = self.nlp(sentence)

        for t in s_doc:
            if t.ent_type_ not in entities_map.keys():
                continue
            entities_map[t.ent_type_].append(t.text)

        return entities_map

    def extract_claimer_saying(self, token: spacy.tokens.token.Token):
        """Extract the claims made by a claimer in the original
        sentence using ccomp (clausal complement)

        Returns the saying claimed by the claimer
        """
        tree = pos_tree.PartOfSpeechTree(token)
        ccomp_child = tree.root.retrieve_children("ccomp")

        if not ccomp_child:
            return None

        saying = " ".join([x.text for x in ccomp_child[0].token.subtree])
        saying = saying.strip()
        return saying

    def extract_claim(
        self,
        sentence: spacy.tokens.Span,
        token: spacy.tokens.token.Token
    ):
        """
        Return the claim by parsing the token's tree
        """
        tree = pos_tree.PartOfSpeechTree(token)

        if not tree.has_spo_structure():
            return None

        entities_map = self.extract_sentence_entities(sentence.text)
        claim = Claim(entities=entities_map, score=0, sentence=sentence.text)
        return claim

    def extract_claims_from_sentence(self, sentence: spacy.tokens.Span):
        """
        Extract claims from the sentence in the following scenarios:

        1. The root predicate could be a claimer verb (e.g say) which
        means the actual claim root token is a child of this token.
            1.1: ccomp is not present, we just try to parse the current token
            1.2: ccomp is present, and we parse the actual ccomp

        2. The root verb is the actual root predicate of the claim.

        Returns a list of claims that are present in the sentence
        """
        claims = []
        for token in sentence:
            if token.dep_ != "ROOT" or token.pos_ != "VERB":
                continue

            # 1: root verb is a claimer verb
            if self.is_claimer_verb(token):
                saying = self.extract_claimer_saying(token)

                # 1.1: if clausal complement is not present, attempt
                # to parse the tree it using this token
                if saying is None:
                    claim = self.extract_claim(sentence, token)
                    if claim:
                        claims.append(claim)
                    continue

                # 1.2: clausal complement is present, parse the complement itself
                result = self.extract_claims(saying)
                for claim in result:
                    claims.append(claim)
                continue

            # 2: root verb is the start of the predicate
            claim = self.extract_claim(sentence, token)
            if claim:
                claims.append(claim)

        return claims

    def extract_claims(self, text: str):
        """
        Breaks up the text into sentences and extracts claims
        that are present in the sentences

        Returns a list of claims
        """

        sanitized_text = self.sanitext_text(text)
        text_doc = self.nlp(sanitized_text)
        self.merge_spans_in_doc(text_doc)

        claims = []
        for sentence in text_doc.sents:
            claims.extend(self.extract_claims_from_sentence(sentence))

        return claims
