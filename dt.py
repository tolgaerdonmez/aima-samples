from dataclasses import dataclass

import dt_utils


@dataclass()
class Label:
    id: int
    label: str | bool | int
    values: list


class DecisionLeafNode:
    __id = 0

    def __init__(self, value):
        self.id = DecisionLeafNode.__id
        DecisionLeafNode.__id += 1
        self.value = value

    def __str__(self):
        return f"DecisionLeafNode({self.id}, {self.value})"


class DecisionTreeNode:

    def __init__(self, label, children):
        self.label = label
        self.children = children


class DecisionTreeLearner:

    def __init__(self, labels: [Label], examples):
        self.labels = labels
        self.examples = examples

    def train(self):
        return self.__train(self.examples, self.examples, self.labels)

    def __train(
        self,
        examples_to_learn,
        parent_examples_to_learn,
        labels_to_match,
    ):
        (examples, goals) = dt_utils.seperate_examples(examples_to_learn)

        if len(examples_to_learn) == 0:
            return dt_utils.plurality_value(
                dt_utils.seperate_examples(parent_examples_to_learn)[1])
        elif dt_utils.have_same_classification(goals):
            return dt_utils.plurality_value(goals)
        elif len(labels_to_match) == 0:
            return dt_utils.plurality_value(goals)

        selected_label = dt_utils.select_label(examples, goals,
                                               labels_to_match)
        node = DecisionTreeNode(selected_label.label, {})
        labels = [
            label for label in labels_to_match if label.id != selected_label.id
        ]
        for v in selected_label.values:
            exs = list(
                filter(
                    lambda ex: ex[selected_label.id] == v,
                    examples_to_learn,
                ))
            child_node = self.__train(
                exs,
                examples_to_learn,
                labels,
            )
            node.children[v] = child_node

        return node
