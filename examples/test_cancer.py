import pybnc as pb
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(123)

data = np.genfromtxt("examples/cancer.csv", delimiter=",", skip_header=True)
# train, test = train_test_split(data)
train = data
test = data

target = 0
train_X = np.delete(train, target, axis=1)
train_Y = train[:, target]
test_X = np.delete(test, target, axis=1)
test_Y = test[:, target]

bn = pb.BayesNetClassifier()
bn.fit(train_X, train_Y)

sample = train_X[0]
print("sample: ", sample)

for node in bn.bn_nodes:
    print("\n")
    print("node.parents", node.parents)
    node_sample = np.array([int(train_Y[0]) if p == 'Y' else int(sample[p])
                            for p in node.parents])
    print("node_sample", sample)
    print(f"P({node.target} | {', '.join(str(x) for x in node.parents)}): ",
          node.hdp.query(node_sample))
    node.hdp.print_tree()

query = target
# event = {column: 0 for column in train_X.columns}
# print("likelihood")
# print(bn.query(target, event=event, algorithm="likelihood", n_iterations=1000))
# print("rejection")
# print(bn.query(target, event=event, algorithm="rejection", n_iterations=1000))
# print("gibbs")
# print(bn.query(target, event=event, algorithm="gibbs", n_iterations=1000))
# print("exact")
# print(bn.query(target, event=event, algorithm="exact", n_iterations=1000))

bn.graphviz().render('cancer_kdb2', directory="examples/figures",
                     format='png', cleanup=True)

# query = target
# event = {column: False for column in train_X.columns}
# print(bn.query(target, event=event, algorithm="rejection", n_iterations=10000))
##
#
# print("TRAIN ACCURACY:")
# predictions = bn.predict(train_X)
#
#
# print("TEST ACCURACY:")
# bn.predict(test_X)
