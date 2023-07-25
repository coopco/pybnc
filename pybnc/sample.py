import random
import typing

import pandas as pd


class BNSampling():

    def __init__(self, seed: int = None):
        self.seed = seed
        self._rng = random.Random(seed)

    def _forward_sample(
        self, init: dict = None
    ) -> typing.Iterator[typing.Tuple[dict, float]]:
        """Perform forward sampling.

        This is also known as "ancestral sampling" or "prior sampling".

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

    def _rejection_sampling(self, *query, event, n_iterations):
        """Answer a query using rejection sampling.

        This is probably the easiest approximate inference method to
        understand. The idea is simply to produce a random sample and keep it
        if it satisfies the specified event. The sample is rejected if any part
        of the event is not consistent with the sample. The downside of this
        method is that it can potentially reject many samples, and therefore
        requires a large `n` in order to produce reliable estimates.

        """

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

    def _llh_weighting(self, *query, event, n_iterations):
        """Likelihood weighting.

        Likelihood weighting is a particular instance of importance sampling.
        The idea is to produce random samples, and weight each sample according
        to its likelihood.

        """

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

    def _gibbs_sampling(self, *query, event, n_iterations):
        """Gibbs sampling.

        The mathematical details of why this works are quite involved, but the
        idea is quite simple. We start with a random sample where the event
        variables are specified. Every iteration, we pick a random variable
        that is not part of the event variables, and sample it randomly. The
        sampling is conditionned on the current state of the sample, which
        requires computing the conditional distribution of each variable with
        respect to it's Markov blanket. Every time a random value is sampled,
        we update the current state and record it.

        """
        pass
