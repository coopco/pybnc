import collections
import itertools
import graphlib
import numpy as np

from .parameter.hdp import estimate_prob_HDP


class BNNode():
    def __init__(self, target, parents):
        self.target = target
        self.parents = parents

    def fit(self, x_child: np.array, x_parents: np.ndarray, target=None, method="hdp", **kwargs):
        """

        Fit CPT to data

        X: The conditioning variables (parents)
        Xc: The conditioned variable (child)
        target: The target variable for classification if required

        """
        # TODO error check that parents are correct
        if method == "hdp":
            assert target is not None, "Target needs to be specified for HDPs"
            self.method = "hdp"
            self.hdp = estimate_prob_HDP(x_parents, x_child, target, **kwargs)

    def query(self, sample: np.array):
        """

        Get the probability estimated by the HDP process

        sample: a datapoint (without the target variable)

        """
        if self.method == "hdp":
            return self.hdp.query(sample)

    def sample(self, n=1):
        if self.method == "hdp":
            return self.hdp.sample(n)


# BELOW code from sorobn
class BNGraph():
    def __init__(self, *structure):
        def coerce_list(obj):
            if isinstance(obj, list):
                return obj
            return [obj]

        # BELOW CODE FROM sorobn
        # The structure is made up of nodes (scalars) and edges (tuples)
        edges = (e for e in structure if isinstance(e, tuple))
        nodes = set(e for e in structure if not isinstance(e, tuple))

        # Convert edges into children and parent connections
        self.parents = collections.defaultdict(set)
        self.children = collections.defaultdict(set)

        for parents, children in edges:
            for parent, child in itertools.product(
                coerce_list(parents), coerce_list(children)
            ):
                self.parents[child].add(parent)
                self.children[parent].add(child)

        # collections.defaultdict(set) -> dict(list)
        print(self.parents)
        self.parents = {node: sorted(parents, key=lambda i: ord(str(i)))
                        for node, parents in self.parents.items()}
        self.children = {
            node: sorted(children) for node, children in self.children.items()
        }

        # The nodes are sorted in topological order. Nodes of the same level
        # are sorted in lexicographic order.
        ts = graphlib.TopologicalSorter()
        all_nodes = {*self.parents.keys(), *self.children.keys(), *nodes}
        for node in sorted(all_nodes, key=lambda i: ord(str(i))):
            ts.add(node, *self.parents.get(node, []))
        self.nodes = list(ts.static_order())

        self.bn_nodes = [BNNode(node, self.parents.get(node, []))
                         for node in self.nodes]

    def ancestors(self, node):
        """Return a node's ancestors."""
        parents = self.parents.get(node, ())
        if parents:
            parent_ancestors = set.union(*[self.ancestors(p) for p in parents])
            return set(parents) | parent_ancestors
        return set()

    def roots(self):
        """Return the network's roots.

        A root is a node that doesn't have any parents.

        """
        return [node for node in self.nodes if node not in self.parents]

    def leaves(self):
        """Return the network's leaves.

        A root is a node that doesn't have any children.

        """
        return [node for node in self.nodes if node not in self.children]

    def is_tree(self):
        """Indicate whether or not the network is a tree.

        Each node in a tree has at most one parent. Therefore, the network is
        not a tree if any of its nodes has two or more parents.
        # TODO not sufficient

        """
        return not any(len(parents) > 1 for parents in self.parents.values())
