import pydash
from collections import deque

class ParseTreeNode(object):

    def __init__(self, spacyToken): #spacyToken is the root
         self.innerToken = spacyToken
         self.children = {}

    def insertChild(self, spacyToken, relationship):
        childNode = ParseTreeNode(spacyToken)
        if relationship in self.children:
            self.children[relationship].append(childNode)
        else:
            self.children[relationship] = [childNode]
        return childNode

    def retrieveChildren(self, relationship):
        if relationship not in self.children:
            return []
        return self.children[relationship]
    
class ParseTreeNode(object):

    def __init__(self, spacyToken): #spacyToken is the root
         self.innerToken = spacyToken
         self.children = {}

    def insertChild(self, spacyToken, relationship):
        childNode = ParseTreeNode(spacyToken)
        if relationship in self.children:
            self.children[relationship].append(childNode)
        else:
            self.children[relationship] = [childNode]
        return childNode

    def retrieveChildren(self, relationship):
        if relationship not in self.children:
            return []
        return self.children[relationship]
    
class ParseTree(object):
    def __init__(self, rootToken):
        self.root = ParseTreeNode(rootToken)

        nodeQueue = deque()
        nodeQueue.append(self.root)
        while len(nodeQueue) > 0:
            node = nodeQueue.popleft()
            for child in node.innerToken.children:
                if child.dep_ == "punct":
                    continue
                if child.dep_ == "appos":
                    for apposChild in child.children:
                        if apposChild.dep_ == "punct":
                            continue
                        childNode = node.insertChild(apposChild, apposChild.dep_)
                        nodeQueue.append(childNode)
                    continue
                childNode = node.insertChild(child, child.dep_)
                nodeQueue.append(childNode)

    def printTreeLevelOrder(self):
        nodeQueue = deque()
        nodeQueue.append(self.root)
        
        level = 1
        levelCount = len(nodeQueue)
        while len(nodeQueue) > 0:
            node = nodeQueue.popleft()

            for (key, childNodeList) in node.children.items():
                for childNode in childNodeList:
                    print(key, "-", childNode.innerToken.text, "-- ", node.innerToken.text)
                    nodeQueue.append(childNode)

            levelCount = levelCount - 1
            if levelCount == 0:
                level = level + 1
            levelCount = len(nodeQueue)
    

    def extracePredicateInverse(self):
        inverseRelationship = ["neg"]
        inverseNodes = self.root.retrieveChildren("neg")
        return inverseNodes
        
    
    def extractContext(self):
        contextRelationships = ['advcl']
        contextNodes = []
        for relationship in contextRelationships:
            contextNodes.extend(self.root.retrieveChildren(relationship))

        if len(contextNodes) <= 0:
            return None
        
        contextNode = contextNodes[0]
        context = " ".join([x.text for x in list(contextNode.innerToken.subtree)]).strip()

        return context


    def extractSubjectNodes(self, withCCAndConj=False):
        subjectRelationships = ["nsubj", "nsubjpass"]
        subjectNodes = []
        for relationship in subjectRelationships:
            subjectNodes.extend(self.root.retrieveChildren(relationship))
    
        if len(subjectNodes) <= 0:
            return []

        subjectNode = subjectNodes[0]
        leftTreeModifiers = ["poss", "npadvmod" ,"det", "compound", "amod", "advmod", "nummod", "nmod"]
        rightTreeModifiers = ["poss", "det", "compound", "amod", "nummod", "case", "appos", "prep", "pobj"]

        leftTreeNodes = []
        rightTreeNodes = []

        def _extractLeftTreeCompoundsDFS(node):
            leftTreeNodes.insert(0, node)

            if len(pydash.intersection(leftTreeModifiers, list(node.children.keys()))) <= 0:
                return

            for modifier in leftTreeModifiers:
                if modifier not in node.children:
                    continue
                for childNode in node.children[modifier]:
                    if childNode.innerToken.i < node.innerToken.i: # left
                        _extractLeftTreeCompoundsDFS(childNode)
        
        def _extractRightTreeCompoundsDFS(node):
            rightTreeNodes.append(node)
            
            if len(pydash.intersection(rightTreeModifiers, list(node.children.keys()))) <= 0:
                return

            for modifier in rightTreeModifiers:
                if modifier not in node.children:
                    continue
                for childNode in node.children[modifier]:

                    if childNode.innerToken.i > node.innerToken.i: # right
                        _extractRightTreeCompoundsDFS(childNode)

        _extractLeftTreeCompoundsDFS(subjectNode)
        _extractRightTreeCompoundsDFS(subjectNode)

        if withCCAndConj and len(subjectNodes) > 0:
            ccAndConjRelationships = ["cc", "conj"]
            ccAndConjQueue = deque()
            
            ccAndConjQueue.append(subjectNodes[0])
            while len(ccAndConjQueue) > 0:
                node = ccAndConjQueue.popleft()
                subjectNodes.append(node)

                for (rel, childNodeList) in node.children.items():
                    if rel in ccAndConjRelationships:
                        for childNode in childNodeList:
                            ccAndConjQueue.append(childNode)
            subjectNodes = subjectNodes[1:]

        subjectNodes = leftTreeNodes[:-1] + subjectNodes + rightTreeNodes[1:]    
        return subjectNodes


    def extractObjectNodes(self, withCCAndConj=False):
        objectRelationship = ["dobj", "acomp"]
        objectNodes = []
        for relationship in objectRelationship:
            objectNodes.extend(self.root.retrieveChildren(relationship))
    

        if len(objectNodes) <= 0:
            return []
        
        objectNode = objectNodes[0]
        leftTreeModifiers = ["poss", "npadvmod" ,"det", "compound", "amod", "advmod", "nummod", "nmod"]
        rightTreeModifiers = ["poss", "det", "compound", "amod", "nummod", "case", "appos", "prep", "pobj"]

        leftTreeNodes = []
        rightTreeNodes = []

        def _extractLeftTreeCompoundsDFS(node):
            leftTreeNodes.insert(0, node)

            if len(pydash.intersection(leftTreeModifiers, list(node.children.keys()))) <= 0:
                return

            for modifier in leftTreeModifiers:
                if modifier not in node.children:
                    continue
                for childNode in node.children[modifier]:
                    if childNode.innerToken.i < node.innerToken.i: # left
                        _extractLeftTreeCompoundsDFS(childNode)
        
        def _extractRightTreeCompoundsDFS(node):
            rightTreeNodes.append(node)
            
            if len(pydash.intersection(rightTreeModifiers, list(node.children.keys()))) <= 0:
                return

            for modifier in rightTreeModifiers:
                if modifier not in node.children:
                    continue
                for childNode in node.children[modifier]:

                    if childNode.innerToken.i > node.innerToken.i: # right
                        _extractRightTreeCompoundsDFS(childNode)

        _extractLeftTreeCompoundsDFS(objectNode)
        _extractRightTreeCompoundsDFS(objectNode)


        if withCCAndConj and len(objectNodes) > 0:
            ccAndConjRelationships = ["cc", "conj"]
            ccAndConjQueue = deque()
            
            ccAndConjQueue.append(objectNodes[0])
            while len(ccAndConjQueue) > 0:
                node = ccAndConjQueue.popleft()
                objectNodes.append(node)

                for (rel, childNodeList) in node.children.items():
                    if rel in ccAndConjRelationships:
                        for childNode in childNodeList:
                            ccAndConjQueue.append(childNode)
            objectNodes = objectNodes[1:]
        
        objectNodes = leftTreeNodes[:-1] + objectNodes + rightTreeNodes[1:]                
        return objectNodes
    
    # the plane is going to land on the ground with 214 passengers
    def extractActionNodesData(self):
        actionRelationships = ["xcomp"]
        actionNodes = []
        
        for relationship in actionRelationships:
            actionNodes.extend(self.root.retrieveChildren(relationship))

        if len(actionNodes) <= 0:
            return None
        
        actionNode = actionNodes[0]
        actionNodeParseTree = ParseTree(actionNode.innerToken)
        actionNodeData = actionNodeParseTree.extractData()
        return actionNodeData
    
    def extractPrepAndPobjNodes(self):
        lefts = []
        rights = []
        prepPobjModifiers = ["prep", "pobj", "pcomp"]
        def _extractPrepPobjDFS(node):
            if node.innerToken.i < self.root.innerToken.i:
                lefts.append(node)
            else:
                rights.append(node)
            
            if len(pydash.intersection(prepPobjModifiers, node.children.keys)) <= 0:
                return
            
            for modifier in prepPobjModifiers:
                if modifier not in node.children:
                    continue
                for childNode in node.children[modifier]:
                    _extractPrepPobjDFS(childNode)

        _extractPrepPobjDFS(self.root)
        return {
            "lefts": lefts,
            "rights": rights[1:]
        }
    
    
    def extractData(self):
        predicate = self.root.innerToken.text.strip()
        predicateInverseNodes = self.extracePredicateInverse()
        predicateContex = self.extractContext()


        subjectNodes = self.extractSubjectNodes(withCCAndConj=True)
        dobjectNodes = self.extractObjectNodes(withCCAndConj=True)
        actionNodesData = self.extractActionNodesData()
        prepPobjNodes = self.extractPrepAndPobjNodes()
        
        predicateInverse = ""
        for node in predicateInverseNodes:
            predicateInverse = predicateInverse + " " + node.innerToken.text
        predicateInverse = predicateInverse.strip()

        subject = ""
        for node in subjectNodes:
            subject = subject + " " + node.innerToken.text
        subject = subject.strip()
        
        dobject = ""
        for node in dobjectNodes:
            dobject = dobject + " " + node.innerToken.text
        dobject = dobject.strip()
        
        ppLefts = []
        ppRights = []
        
        skipFlag = False
        temp = ""        
        for node in prepPobjNodes['lefts']:
            if not skipFlag:            
                if len(node.innerToken.text.split(" ")) > 1:
                    ppLefts.append(node.innerToken.text)
                else:
                    temp = node.innerToken.text
                    skipFlag = True
            else:
                temp = temp + " " + node.innerToken.text
                ppLefts.append(temp)
                skipFlag = False
                temp = ""
                
        skipFlag = False
        temp = ""        
        for node in prepPobjNodes['rights']:
            if not skipFlag:            
                if len(node.innerToken.text.split(" ")) > 1:
                    ppRights.append(node.innerToken.text)
                else:
                    temp = node.innerToken.text
                    skipFlag = True
            else:
                temp = temp + " " + node.innerToken.text
                ppRights.append(temp)
                skipFlag = False
                temp = ""
        
        if dobject == "" and len(ppRights) > 0:
            dobject = ppRights[0]
            ppRights = ppRights[1:]
        elif dobject == "" and len(ppLefts) > 0:
            dobject = ppLefts[0]
            ppLefts = ppLefts[1:]
        
        action = ""
        if actionNodesData is not None:
            action = actionNodesData["predicate"]
        
        if subject == "" or object == "":
            return None

        return {
            "predicate": predicate,
            "predicateInverse": predicateInverse,
            "predicateContext": predicateContex,
            "subject": subject,
            "object": dobject,
            "action": action,
            "prepPobj": ppLefts + ppRights
        }