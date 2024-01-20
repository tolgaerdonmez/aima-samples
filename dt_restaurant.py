import csv

from dt import Label, DecisionTreeLearner
import dt_utils

data = None
with open(
        "aima-data/restaurant.csv",
        "r",
        encoding="utf-8",
) as csvfile:
    data = [[s.strip() for s in row]
            for row in csv.reader(csvfile, delimiter=',')]

raw_labels = data[0]
examples = data[1:]
for e in examples:
    e[len(e) - 1] = 1 if e[len(e) - 1] == 'Yes' else 0

labels = []
for i in range(len(raw_labels) - 1):
    labels.append(
        Label(
            i,
            raw_labels[i],
            list(set([e[i] for e in examples])),
        ))

print("Training...")
tree = DecisionTreeLearner(labels, examples).train()
print("Complete.")

dt_utils.create_decision_tree_graph(tree).render('trained_tree',
                                                 format="png",
                                                 cleanup=True)
