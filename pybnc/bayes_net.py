# TODO change sample.py to inference.py?
import typing

import numpy as np
import pandas as pd

from .sample import BNSampling
from .graph import BNGraph


__all__ = ["BayesNet"]


class BayesNet(BNGraph, BNSampling):
    """
    Bayesian network.
    """

    def __init__(self, *structure, prior_count: int = None, seed: int = None):
        BNGraph.__init__(self, *structure)
        BNSampling.__init__(self, seed)

    def prepare(self) -> "BayesNet":
        # TODO does anything need to be done here?
        pass

    def partial_fit(self, X: pd.DataFrame, target=None):
        """Update the parameters of each conditional distribution."""
        # TODO incremental update
        for node in self.bn_nodes:
            x_child = X[node.target]
            x_parents = X[node.parents]

            node.fit(x_child, x_parents, target, n_iters=100)

        self.prepare()
        return self

    # TODO kind of weird passing target series when it is already apart of X
    def fit(self, X: pd.DataFrame, target=None):
        """Find the values of each conditional distribution."""
        return self.partial_fit(X, target)

    def _variable_elimination(self, *query, event):
        """
        Variable elimination.
        """
        raise NotImplementedError

    # BELOW CODE from sorobn
    # Computes P(query | event)
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
            answer = self._gibbs_sampling(
                *query, event=event, n_iterations=n_iterations
            )

        elif algorithm == "likelihood":
            answer = self._llh_weighting(
                *query, event=event, n_iterations=n_iterations)

        elif algorithm == "rejection":
            answer = self._rejection_sampling(
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
    def predict_proba(self, X: typing.Union[dict, pd.DataFrame]):
        """Return likelihood estimates.

        The probabilities are obtained by first computing the full joint
        distribution. Then, the likelihood of a sample is retrieved by
        accessing the relevant row in the full joint distribution.

        This method is a stepping stone for other functionalities, such as
        computing the log-likelihood. The latter can in turn be used for
        structure learning.

        Parameters
        ----------
        X
            One or more samples.

        """

        # Convert dict to DataFrame
        if isinstance(X, dict):
            return self.predict_proba(pd.DataFrame([X])).iloc[0]

        fjd = self.full_joint_dist()

        # For partial events
        if unobserved := set(fjd.index.names) - set(X.columns):
            fjd = fjd.droplevel(list(unobserved))
            fjd = fjd.groupby(fjd.index.names).sum()

        # For multiple events
        if len(fjd.index.names) > 1:
            return fjd[pd.MultiIndex.from_frame(X[fjd.index.names])]

        return fjd

    def predict_log_proba(self, X: typing.Union[dict, pd.DataFrame]):
        """Return log-likelihood estimates.

        Parameters
        ----------
        X
            One or more samples.

        """
        return np.log(self.predict_proba(X))

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

    def _repr_svg_(self):
        return self.graphviz()
