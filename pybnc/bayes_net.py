# TODO change sample.py to inference.py?
import typing
import random

import numpy as np

from .graph import BNGraph
from .structure import kdb


__all__ = ["BayesNetClassifier"]


class BayesNetClassifier(BNGraph):
    """
    Bayesian network.
    """

    def __init__(self, seed: int = None):
        self.seed = seed
        self._rng = random.Random(seed)

    def partial_fit(self, X: np.ndarray, Y: np.ndarray):
        """Update the parameters of each conditional distribution."""
        raise NotImplementedError
        # TODO incremental update

    # TODO args for structure and parameter methods
    # TODO np.ndarray
    def fit(self,
            X: np.ndarray,
            Y: np.ndarray,
            structure="kdb",
            parameter="hdp",
            **kwargs):
        """Find the values of each conditional distribution."""

        # Compute mutual information between all varaibles (excluding target I think)
        # Compute mutual information between all varaibles conditioned on target
        # TODO need dataloader class
        self.target_cardinality = len(np.unique(Y))

        # Learn structure
        if structure == "kdb":
            edges = kdb(X, Y)
        else:
            raise NotImplementedError

        BNGraph.__init__(self, *edges)

        for node in self.bn_nodes:
            if "Y" == node.target:
                x_child = Y
            else:
                x_child = X[:, node.target]

            if "Y" in node.parents:
                # Append Y to X[parents]
                parents = [item for item in node.parents if item != "Y"]
                x_parents = X[:, parents]
                x_parents = np.hstack((x_parents, Y[:, np.newaxis]))
            else:
                x_parents = X[:, node.parents]

            # TODO n_iters arg
            node.fit(x_child, x_parents, Y, method=parameter, **kwargs)

        return self

    def variable_elimination(self, X):
        """
        Variable elimination.
        """
        raise NotImplementedError

    def rejection_sampling(self, event, n_iterations):
        """Answer a query using rejection sampling."""
        sampler = self._forward_sample()
        # TODO preallocate array size
        # samples = {var: [] for var in query}
        counts = np.zeros(self.target_cardinality)

        for _ in range(n_iterations):
            sample, _ = next(sampler)

            # Reject if sample is not consistent with event
            # TODO ugly
            reject = False
            for i, value in enumerate(event):
                if sample[i] != value:
                    reject = True

            if reject:
                continue

            # Accept sample
            counts[sample['Y']] += 1
            # for var in query:
            #     samples[var].append(sample[var])

        num_samples = sum(counts)
        return counts / sum(counts) if num_samples > 0 else counts

    def predict(self, X):
        return np.argmax(X, axis=1)

    def predict_proba(self, X, method="exact", n_iterations=100):
        # TODO vectorize
        # Compute P(Y | X)
        if method == "exact":
            return self.variable_elimination(X)
        elif method == "rejection":
            return self.rejection_sampling(X, n_iterations)
        else:
            raise ValueError

    def _forward_sample(self, init=None):
        """
        Perform forward sampling.

        init:

        """

        if init is not None:
            raise NotImplementedError

        init = init or {}
        # TODO advantages/disadvantages of generator
        while True:
            sample = {}
            likelihood = 1.0

            for _, node in enumerate(self.nodes):
                # TODO ugly
                bn_node = None
                for bn_node_i in self.bn_nodes:
                    if bn_node_i.target == node:
                        bn_node = bn_node_i

                # Access P(node | parents(node))
                if node in self.parents:
                    # Doesn't throw error since nodes are in topological order
                    condition = {
                        idx: sample[parent]
                        for idx, parent in enumerate(self.parents[node])}
                else:
                    condition = {}

                node_value, p = bn_node.sample(condition=condition)
                node_value = node_value[-1]

                sample[node] = node_value
                likelihood *= p

            yield sample, likelihood

    def sample(self, n=1, init={}):
        pass

    # BELOW CODE from sorobn
    # Computes P(query | event)
    def query(
        self,
        *query: typing.Tuple[str],
        event: dict,
        algorithm="exact",
        n_iterations=100,
    ) -> np.ndarray:
        """
        Computes P(query | event)
        """

        if not query:
            raise ValueError("At least one query variable has to be specified")

        for q in query:
            if q in event:
                raise ValueError(
                    "A query variable cannot be part of the event")

        if algorithm == "exact":
            answer = self._variable_elimination(*query, event=event)

        elif algorithm == "gibbs":
            answer = self.gibbs_sampling(
                *query, event=event, n_iterations=n_iterations
            )

        elif algorithm == "likelihood":
            answer = self.llh_weighting(
                *query, event=event, n_iterations=n_iterations)

        elif algorithm == "rejection":
            answer = self.rejection_sampling(
                *query, event=event, n_iterations=n_iterations
            )

        else:
            raise ValueError(
                "Unknown algorithm, must be one of: exact, gibbs, likelihood, "
                + "rejection"
            )

        answer = answer.rename(f'P({", ".join(query)})')

        # We sort the index levels if there are multiple query variables
        if isinstance(answer.index, pd.MultiIndex):
            answer = answer.reorder_levels(sorted(answer.index.names))

        return answer.sort_index()

    # Computes P(event)

    # BELOW CODE from sorobn

    def importance_sampling(self, *query, event, n_iterations):
        """Importance sampling."""

        samples = {var: [None] * n_iterations for var in query}
        likelihoods = [None] * n_iterations

        sampler = self._forward_sample(init=event)

        for i in range(n_iterations):

            # Sample by using the events as fixed values
            sample, likelihood = next(sampler)

            # Compute the likelihood of this sample
            for var in query:
                samples[var][i] = sample[var]
            likelihoods[i] = likelihood

        # Now we aggregate the resulting samples according to their
        # associated likelihoods
        results = pd.DataFrame({"likelihood": likelihoods, **samples})
        results = results.groupby(list(query))["likelihood"].mean()
        results /= results.sum()

        return results

    def gibbs_sampling(self, *query, event, n_iterations):
        """Gibbs sampling."""
        pass

    def graphviz(self):
        """Export to Graphviz.
        """

        import graphviz

        G = graphviz.Digraph()

        for node in self.nodes:
            G.node(str(node))

        for node, children in self.children.items():
            for child in children:
                G.edge(str(node), str(child))

        return G
