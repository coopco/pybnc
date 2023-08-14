# TODO change sample.py to inference.py?
import typing
import random

import numpy as np
import pandas as pd

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

    def partial_fit(self, X: pd.DataFrame, Y: pd.Series):
        """Update the parameters of each conditional distribution."""
        raise NotImplementedError
        # TODO incremental update

    # TODO args for structure and parameter methods
    # TODO np.ndarray
    def fit(self,
            X: pd.DataFrame,
            Y: pd.Series,
            structure="kdb",
            parameter="hdp"):
        """Find the values of each conditional distribution."""

        # Compute mutual information between all varaibles (excluding target I think)
        # Compute mutual information between all varaibles conditioned on target
        # TODO need dataloader class

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

            node.fit(x_child, x_parents, Y, method=parameter, n_iters=100)

        return self

    def variable_elimination(self, X):
        """
        Variable elimination.
        """
        raise NotImplementedError

    def predict(self, X):
        return np.argmax(X, axis=1)

    def predict_proba(self, X, method="exact"):
        # Compute P(Y | X)
        if method == "exact":
            return self.variable_elimination(X)
        else:
            raise ValueError

    def _forward_sample(self, init=None):
        """
        Perform forward sampling.

        init:

        """

        init = init or {}

        while True:

            sample = {}
            likelihood = 1.0

            for idx, node in enumerate(self.nodes):
                bn_node = None
                for bn_node_i in self.bn_nodes:
                    if bn_node_i.target == node:
                        bn_node = bn_node_i

                # Access P(node | parents(node))
                # P = self.P[node]
                # If node has a parent
                if node in self.parents:
                    # Doesn't throw error since nodes are in topological order
                    condition = {
                        idx: sample[parent]
                        for idx, parent in enumerate(self.parents[node])}
                else:
                    condition = {}

                # if node value fixed
                if node in init:
                    node_value = init[node]
                    # TODO ensure condition nodes in correct order
                    p = bn_node.query(condition)[node_value]
                else:
                    # TODO what if node is a root?
                    # node_value = P.cpt.sample(rng=self._rng)
                    node_value, p = bn_node.sample(condition)
                    node_value = node_value[-1]

                sample[node] = node_value
                likelihood *= p

            yield sample, likelihood

    def sample(self, n=1, init={}):
        pass

    # BELOW CODE from sorobn
    # Computes P(query | event)
    # TODO how to write this without pandas column namees?
    def query(
        self,
        *query: typing.Tuple[str],
        event: dict,
        algorithm="exact",
        n_iterations=100,
    ) -> pd.Series:
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

    def sample(self, n=1, init: dict = None, method="forward"):
        """Generate a new sample at random by using forward sampling.

        Parameters
        ----------
        n
            Number of samples to produce. A DataFrame is returned if `n > 1`.
            A dictionary is returned if not.
        init
            Allows forcing certain variables to take on given values.
        method
            The sampling method to use. Possible choices are: forward.

        """

        if method == "forward":
            sampler = (sample for sample, _ in self._forward_sample(init))

        else:
            raise ValueError("Unknown method, must be one of: forward")

        if n > 1:
            return pd.DataFrame(next(sampler) for _ in range(n)).sort_index(
                axis="columns"
            )
        return next(sampler)

    def rejection_sampling(self, *query, event, n_iterations):
        """Answer a query using rejection sampling."""

        # We don't know many samples we won't reject, therefore we cannot
        # preallocate arrays
        samples = {var: [] for var in query}
        sampler = (sample for sample, _ in self._forward_sample())

        for _ in range(n_iterations):
            sample = next(sampler)

            # Reject if the sample is not consistent with the specified events
            if any(sample[var] != val for var, val in event.items()):
                continue

            for var in query:
                samples[var].append(sample[var])

        # Aggregate and normalize the obtained samples
        samples = pd.DataFrame(samples)
        return samples.groupby(list(query)).size() / len(samples)

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
