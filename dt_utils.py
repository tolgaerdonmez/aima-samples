from math import log2
from random import randint

from graphviz import Digraph


def select_label(examples, example_goals, labels):
    """
    Returns the attrib index with most importance
    """

    n = sum([1 for x in example_goals if x == 0])
    p = sum([1 for x in example_goals if x == 1])

    def boolean_entropy(q):

        b = 0
        if q != 0:
            b += q * log2(q)
        if (1 - q) != 0:
            b += (1 - q) * log2(1 - q)
        return -b

    def remainder(label):
        r = 0

        for d in label.values:
            nk = sum([
                1 for idx, x in enumerate(examples)
                if example_goals[idx] == 0 and x[label.id] == d
            ])

            pk = sum([
                1 for idx, x in enumerate(examples)
                if example_goals[idx] == 1 and x[label.id] == d
            ])

            if pk + nk != 0:
                r += ((pk + nk) / (n + p)) * boolean_entropy(pk / (pk + nk))
        return r

    def gain(label):
        return boolean_entropy(p / (p + n)) - remainder(label)

    return max(
        [(label, gain(label)) for label in labels],
        key=lambda x: x[1],
    )[0]


def have_same_classification(goals):
    return all([x == goals[0] for x in goals])


def seperate_examples(examples) -> (list, list):
    """
        seperates goals and labels from examples
        returns (label values for examples, goal values of examples)
    """
    return [ex[:len(ex) - 1]
            for ex in examples], [ex[len(ex) - 1] for ex in examples]


def plurality_value(goals):
    from dt import DecisionLeafNode
    no = sum([1 for x in goals if x == 0])
    yes = sum([1 for x in goals if x == 1])
    if no > yes:
        return DecisionLeafNode(0)
    elif yes > no:
        return DecisionLeafNode(1)
    else:
        # break ties
        return DecisionLeafNode(randint(0, 1))


def create_decision_tree_graph(node, dot=None):
    from dt import DecisionLeafNode
    if dot is None:
        dot = Digraph(comment='Decision Tree')

    if isinstance(node, type(DecisionLeafNode(0))):
        print("leaf node drawn")
        dot.node(str(node), label=str(node.value))
    else:
        dot.node(str(node), label=str(node.label))
        if node.children is not None:
            for decision, child_node in node.children.items():
                dot.edge(str(node), str(child_node), label=str(decision))
                create_decision_tree_graph(child_node, dot)

    return dot
