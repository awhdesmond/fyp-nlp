import spacy
from typing import List
from collections import deque

class PartOfSpeechTreeNode:
    """
    Part-of-speech tree node
    """

    def __init__(self, token: spacy.tokens.token.Token):
        self.token = token
        self.children = {}

    def insert_child(self, token: spacy.tokens.token.Token):
        """
        Insert a token as a child to the node
        """
        child_node = PartOfSpeechTreeNode(token)
        self.children.setdefault(token.dep_, [])
        self.children[token.dep_].append(child_node)
        return child_node

    def retrieve_children(self, relationship: str):
        """
        Retrieves the children node of this node for
        the specified relationship
        """
        return self.children.get(relationship, [])


class PartOfSpeechTree:
    """
    Part of Speech (POS) Tree constructed using spacy's POS tagging

    Attributes:
        root (PartOfSpeechTreeNode): the root node of the tree
    """

    def __init__(self, token: spacy.tokens.token.Token):
        self.root = PartOfSpeechTreeNode(token)
        self._init_tree()

    def _process_node(self, node: PartOfSpeechTreeNode, node_queue: deque):
        """
        Recursively add the nodes into the tree
        """
        for child in node.token.children:
            if child.dep_ == "punct":
                continue

            if child.dep_ == "appos":
                for appos_child in child.children:
                    child_node = PartOfSpeechTreeNode(appos_child)
                    self._process_node(child_node, node_queue)
                continue

            child_node = node.insert_child(child)
            node_queue.append(child_node)

    def _init_tree(self):
        """
        Initialise the part-of-speech tagged tree for SPO extraction
        """
        node_queue = deque()
        node_queue.append(self.root)

        while node_queue:
            node = node_queue.popleft()
            self._process_node(node, node_queue)

    def nodes_of_relationship(self, relationships: List[str]):
        """
        Returns if the POS tree contains any subject nodes
        from the root predicate
        """
        nodes = []
        for rel in relationships:
            nodes.extend(self.root.retrieve_children(rel))

        return nodes

    def has_spo_structure(self):
        """
        Returns if the tree has an SPO structure
        """
        has_subject_nodes = self.nodes_of_relationship(["nsubj", "nsubjpass"])
        has_object_nodes = self.nodes_of_relationship(["dobj", "acomp"])
        has_action_nodes = self.nodes_of_relationship(["xcomp"])
        has_prep_prop_nodes = self.nodes_of_relationship(["prep", "pobj", "pcomp"])

        return has_subject_nodes and (has_object_nodes or has_prep_prop_nodes or has_action_nodes)
