import csv

from dt import Label, DecisionTreeLearner

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

labels = []
for i in range(len(raw_labels) - 1):
    labels.append(
        Label(
            i,
            raw_labels[i],
            list(set([e[i] for e in examples])),
        ))

print("Training...")
dt_learner = DecisionTreeLearner(labels, examples)
dt_learner.train()

print("Complete.")

dt_learner.export_image("trained_restaurant_tree")
