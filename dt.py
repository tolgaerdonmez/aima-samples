import pickle

import dt_utils


class Label:

    def __init__(self, id: int, label, values: list):
        self.id = id
        self.label = label
        self.values = values


class DecisionLeafNode:
    __id = 0

    def __init__(self, value):
        self.id = DecisionLeafNode.__id
        DecisionLeafNode.__id += 1
        self.value = value

    def __str__(self):
        return f"DecisionLeafNode({self.id}, {self.value})"


class DecisionTreeNode:

    def __init__(self, label: Label, children):
        self.label = label
        self.children = children


class DecisionTreeLearner:

    def __init__(self, labels: [Label], examples):
        self.labels = labels
        self.examples = examples
        self.classifications = list(
            set(dt_utils.seperate_examples(examples)[1]))
        self.tree = None

    def train(self):
        self.tree = self.__train(self.examples, self.examples, self.labels)

    def __train(
        self,
        examples_to_learn,
        parent_examples_to_learn,
        labels_to_match,
    ):
        (examples, goals) = dt_utils.seperate_examples(examples_to_learn)

        if len(examples_to_learn) == 0:
            return dt_utils.plurality_value(
                dt_utils.seperate_examples(parent_examples_to_learn)[1],
                self.classifications)
        elif dt_utils.have_same_classification(goals):
            return dt_utils.plurality_value(goals, self.classifications)
        elif len(labels_to_match) == 0:
            return dt_utils.plurality_value(goals, self.classifications)

        selected_label = dt_utils.select_label(examples, goals,
                                               labels_to_match,
                                               self.classifications)
        node = DecisionTreeNode(selected_label, {})
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

    def decide(self, x: list | dict):
        """
            x: list of attributes values in matching
               order of trained examples |
               dict of attributes values each key
               corresponding the trained example's label index

            Looks up the decision tree to decide on classification

            returns classification
        """
        curr = self.tree
        while (not isinstance(curr, DecisionLeafNode)):
            decision = x[curr.label.id]
            # move among the tree by each label test
            curr = curr.children[decision]

        return curr.value

    def test(self, test_examples):
        """
            Returns the accuracy of the
            trained tree tried on given test_examples
            a value from interval [0,1]
        """
        (examples, goals) = dt_utils.seperate_examples(test_examples)
        accuracy = sum([
            1 for idx, x in enumerate(examples) if self.decide(x) == goals[idx]
        ]) / len(examples)
        return accuracy

    def export_to(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def export_image(self, filename, format="png"):
        if self.tree is None:
            print("No tree to export")
            return

        dt_utils.create_decision_tree_graph(self.tree).render(
            filename,
            format=format,
            cleanup=True,
        )
        print(f"Successfully exported tree as {format}")

    @staticmethod
    def import_from(filename):
        with open(filename, "rb") as f:
            obj = pickle.load(f)
            if isinstance(obj, DecisionTreeLearner):
                return obj
            else:
                raise Exception(
                    "given file is not a valid DecisionTreeLearner")
