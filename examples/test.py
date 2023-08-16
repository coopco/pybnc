import click
import pybnc as pb
import numpy as np
from sklearn.model_selection import train_test_split

# np.random.seed(12321122)


def print_network(bn, train_X, train_Y):
    sample = train_X[0]
    print("sample: ", sample)

    for node in bn.bn_nodes:
        print("\n")
        print("node.parents", node.parents)
        print(f"P({node.target} | {', '.join(str(x) for x in node.parents)})")
        print(node.distribution)


def fit(train_X, train_Y, structure, parameter):
    bn = pb.BayesNetClassifier()
    bn.fit(train_X, train_Y, structure=structure,
           parameter=parameter)

    return bn


def accuracy(test_X, test_Y):
    pass


@click.command()
@click.argument('filename', type=click.Path(exists=True))
@click.option('--target', default=0)
@click.option('--structure', default="kdb")
@click.option('--parameter', default="hdp")
def test(filename, target, structure, parameter):
    """Print FILENAME if the file exists."""
    click.echo(click.format_filename(filename))
    data = np.genfromtxt(filename,
                         delimiter=",", skip_header=True)

    # train, test = train_test_split(data)
    train = data
    test = data

    # target: index of target variable
    train_X = np.delete(train, target, axis=1)
    train_Y = train[:, target]
    test_X = np.delete(test, target, axis=1)
    test_Y = test[:, target]

    bn = fit(train_X, train_Y, structure, parameter)

    print_network(bn, train_X, train_Y)
    probs = np.array([bn.predict_proba(x, method="rejection", n_iterations=3000)
                      for x in test_X])
    predictions = np.argmax(probs, axis=1)
    print(predictions)
    print(test_Y)
    print(sum(predictions == test_Y)/len(test_Y))
    print()

    # print(bn.predict_proba(np.array([0]),
    #      method="rejection", n_iterations=3000))
    # print(bn.predict_proba(np.array([1]),
    #      method="rejection", n_iterations=3000))
    return bn


if __name__ == "__main__":
    bn = test()
