import pybnc as pb
import pandas as pd

data = pd.read_csv("examples/cancer.csv")

target = 'Class'
X = data.drop(target, axis=1)
Y = data[target]

edges = pb.structure.kdb(X, Y)

bn = pb.BayesNet(*edges)
bn.fit(data, Y)
sample = data.iloc[0]
print("sample: ", sample)
#
for node in bn.bn_nodes:
    print("\n")
    print(f"P({node.target} | {', '.join(node.parents)}): ",
          node.hdp.query(sample[node.parents]))
    node.hdp.print_tree()

query = target
event = {column: 0 for column in X.columns}
print("likelihood")
print(bn.query(target, event=event, algorithm="likelihood", n_iterations=1000))
print("rejection")
print(bn.query(target, event=event, algorithm="rejection", n_iterations=1000))
# print("gibbs")
# print(bn.query(target, event=event, algorithm="gibbs", n_iterations=1000))
# print("exact")
# print(bn.query(target, event=event, algorithm="exact", n_iterations=1000))

bn.graphviz().render('cancer_kdb', directory="examples/figures",
                     format='png', cleanup=True)

query = target
event = {column: False for column in X.columns}
# print(bn.query(target, event=event, algorithm="exact", n_iterations=10000))
