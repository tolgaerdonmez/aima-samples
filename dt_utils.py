from math import log2
from random import randint

from graphviz import Digraph


def boolean_entropy(q):
    b = 0
    if q != 0:
        b += q * log2(q)
    if (1 - q) != 0:
        b += (1 - q) * log2(1 - q)
    return -b


def entropy(vP: list):
    e = 0
    for p in vP:
        e += (p * log2(p)) if p != 0 else 0
    return -e


def select_label(examples, example_goals, labels, classifications):
    """
    Returns the attrib index with most importance
    """
    C = len(example_goals)
    # probability of each classifications occurance in example_goals
    pC = [
        sum([1 for x in example_goals if x == c]) / C for c in classifications
    ]

    def remainder(label):
        r = 0

        for d in label.values:
            # count of examples
            # whose label has the d'th value of label's values
            C_a_k = sum([1 for x in examples if x[label.id] == d])
            if C_a_k == 0:
                continue
            # probability of each classification by the d'th value of label
            pC_a = []
            for c in classifications:
                p = sum([
                    1 for idx, x in enumerate(examples)
                    if example_goals[idx] == c and x[label.id] == d
                ]) / C_a_k
                pC_a.append(p)

            r += (C_a_k / C) * entropy(pC_a)

        return r

    def gain(label):
        return entropy(pC) - remainder(label)

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


def plurality_value(goals, classifications):
    from dt import DecisionLeafNode
    max_class = goals[0]
    count = 0
    for classification in classifications:
        c = sum([1 for x in goals if x == classification])
        if c > count:
            count = c
            max_class = classification
        elif c == count:
            # break ties
            i = randint(0, 1)
            count = [count, c][i]
            max_class = [max_class, classification][i]

    return DecisionLeafNode(max_class)


def create_decision_tree_graph(node, dot=None):
    from dt import DecisionLeafNode
    if dot is None:
        dot = Digraph(comment='Decision Tree')

    if isinstance(node, DecisionLeafNode):
        dot.node(str(node), label=str(node.value))
    else:
        dot.node(str(node), label=str(node.label.label))
        if node.children is not None:
            for decision, child_node in node.children.items():
                dot.edge(str(node), str(child_node), label=str(decision))
                create_decision_tree_graph(child_node, dot)

    return dot
