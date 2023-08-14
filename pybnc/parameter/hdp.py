"""
Petitjean, F., Buntine, W., Webb, G. I., & Zaidi, N. (2018). Accurate parameter
    estimation for Bayesian network classifiers using hierarchical Dirichlet
    processes. Machine Learning, 107(8–10), 1303–1331.
    https://doi.org/10.1007/s10994-018-5718-0
"""
import numpy as np
import pybnc.utils as utils


class HDPNode:
    """

    """

    def __init__(self, tree, parent, xc_card):
        # K number of Xc classes
        self.tree = tree
        self.parent = parent
        self.xc_card = xc_card
        self.children = []
        self.n = np.zeros(xc_card)
        self.marginal_n = 0
        self.t = np.zeros(xc_card)
        self.marginal_t = 0
        self.concentration_idx = None  # depth of node at time of sampling

        self.p = np.zeros(xc_card)
        self.p_averaged = np.zeros(xc_card)
        self.np_accumulated = 1

    def get_nodes_at_depth(self, depth):
        if depth == 0:
            return [self]

        nodes = []
        for child in self.children:
            nodes = nodes + child.get_nodes_at_depth(depth - 1)

        return nodes

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None

    def get_concentration(self):
        if self.concentration_idx is None:
            return 2.0

        return self.tree.ctab[self.concentration_idx]

    def update_concentration(self, alpha):
        if alpha <= 0:
            raise Exception
        self.tree.ctab[self.concentration_idx] = alpha

# /**
# * This function computes the values of the smoothed conditional probabilities
# * as a function of (nk,tk,c,d) and of the parent probability. <br/>
# * p_k = ( ( nk - tk*d ) / (N + c) ) ) + ( ( c + T*d ) / (N + c) ) ) *
# * p^{parent}_k
# */
    def compute_probabilities(self):
        alpha = self.get_concentration()
        sum = 0.0

        for k in range(self.xc_card):
            # Uniform parent if root
            if self.is_root():
                parent_prob = 1.0 / self.xc_card
            else:
                parent_prob = self.parent.p[k]

            # TODO check where this formula comes from
            self.p[k] = self.n[k] / (self.marginal_n + alpha) \
                + alpha * parent_prob / (self.marginal_n + alpha)
            sum += self.p[k]

        # normalize
        self.p /= sum

        for child in self.children:
            child.compute_probabilities()

# /**
# * This method accumulates the pks so that the final result is averaged over
# * several successive iterations of the Gibbs sampling process in log space to
# * avoid underflow
# */
    def record_probability_recursively(self):
        # p_averaged stores the log sum
        sum = 0.0

        for k in range(self.xc_card):
            self.p_averaged[k] += (self.p[k] -
                                   self.p_averaged[k]) / self.np_accumulated
            sum += self.p_averaged[k]

        # normalize
        for k in range(self.xc_card):
            self.p_averaged[k] /= sum

        self.np_accumulated += 1

        for child in self.children:
            child.record_probability_recursively()


class HDPTree:
    """

    """

    def __init__(self, X, xc, target):
        # Assumes class values are integers
        self.xc_card = int(xc.max()) + 1  # if starting at zero
        self.root = HDPNode(self, None, self.xc_card)

        # TODO make sure column indexes don't get mixed up with full X dataset
        #     e.g. sample requires x_sorted indexes

        # Calculate MI(X_i; X_j | Y) from dataset for all attributes i != j
        mi_cond = {j:
                   utils.conditional_mutual_information(
                       xc, X[:, j], target)
                   for j in range(X.shape[1])}

        # TODO test that mutual information calculations are actually correct
        # Sort attributes on MI(X_i; Y)
        # TODO if parent is in X
        self.x_sorted = sorted(range(X.shape[1]),
                               key=lambda x: mi_cond[x],
                               # if x != target.name else np.inf,
                               reverse=True)
        self.X = X[:, self.x_sorted]
        print(self.x_sorted)

        # Depth tying
        self.ctab = np.full(self.X.shape[1]+1, 2, dtype=float)
        self.depth = self.X.shape[1]+1

        self.init_tree_with_dataset(self.X, xc)

    def print_tree(self, node=None, level=0):
        """Prints nodes of tree recursively."""
        if node is None:
            node = self.root
            n_marg = node.marginal_n
            t_marg = node.marginal_t
            n = node.n
            t = node.t
            p = node.p_averaged
            text = f"  n*={n_marg}, n={n}, t*={t_marg}, t={t}, p={p}"
            print("  " * level + "|_" + text)

        for i, child in enumerate(node.children):
            n_marg = child.marginal_n
            t_marg = child.marginal_t
            n = child.n
            t = child.t
            p = child.p_averaged
            text = f"{i}, n*={n_marg}, n={n}, t*={t_marg}, t={t}, p={p}"
            print("  " * (level+1) + "|_" + text)
            self.print_tree(child, level+1)

    def get_nodes_at_depth(self, depth):
        """Returns all nodes at a given depth of the tree."""
        return self.root.get_nodes_at_depth(depth)

    def add_observation(self, x, xc):
        """
        Assumes data is an integer from 0 to (nValues - 1) representing a
        categorical value.

        x: value of conitioning variables
        xc: value of conditioned variable
        """
        # TODO Assumes x is in x_sorted order (like in init_tree_with_dataset)
        node = self.root
        for xi in x:
            node = node.children[int(xi)]

        node.n[int(xc)] += 1
        node.marginal_n += 1

    def init_tree_with_dataset(self, X, xc):
        """
        Creates the tree branches down to the leaves for which data exists
        TODO if data doesn't exist?
        Assumes data is an integer from 0 to (nValues - 1) representing a
        categorical value.

        X: A dataframe containing the conditioning variables
        xc: A series containing values for the conditioned variables

        """
        # create appropriate branches
        node = self.root
        for i, xi_name in enumerate(range(X.shape[1])):
            # Assumes integers representing categorical values
            xi_card = int(X[:, xi_name].max())+1  # if starting at zero
            for node in self.get_nodes_at_depth(i):
                node.children = [HDPNode(self, node, self.xc_card)
                                 for _ in range(xi_card)]

        for idx, row in enumerate(X):
            self.add_observation(row, xc[idx])

    def query(self, sample):
        """

        Get the probability estimated by the HDP process

        sample: a datapoint (without the target variable)

        """
        sample = sample[self.x_sorted]
        node = self.root
        for i in range(len(sample)):
            node = node.children[int(sample[i])]

        return node.p_averaged

    def sample(self, condition={}, n=1):
        """

        Samples from the CPT

        condition: specify fixed values for parents

        """
        # TODO shouldnt be multiplying p down tree?
        # TODO error check that tree has been fitted
        fixed_parents = condition.keys()
        p = 1.0
        depth = 0
        point = np.zeros(self.depth, dtype=int)
        node = self.root
        while True:
            if depth in fixed_parents:
                idx = condition[depth]
                # p *= node.p_averaged[idx] ?
            else:
                # Pick child, weighted by p
                idx = int(utils.multinomial(node.p_averaged, n=n))
                # Update likelihood
                p *= node.p_averaged[idx]

            point[depth] = idx

            if node.is_leaf():
                break

            node = node.children[idx]
            depth += 1

        return point, p


def record_probability_recursively(root):
    """

    Averages the estimates for all nodes in the tree

    """
    root.compute_probabilities()
    root.record_probability_recursively()


def estimate_prob_HDP(X, xc, target, n_iters, n_burn_in=None):
    """

    Algorithm 1

    data: the dataset
    n_iters: number of iterations to run the sampler fo
    n_burn_in: number of iterations before starting to average out thetas

    """
    if n_burn_in is None:
        n_burn_in = min(1000, int(n_iters/10))

    tree = HDPTree(X, xc, target)
    init_parameters_recursively(tree.root)

    # Assign each node a tied concentration
    for depth in range(tree.depth):
        # Per depth tying
        for node in tree.get_nodes_at_depth(depth):
            node.concentration_idx = depth

    for iter in range(n_iters):  # Gibbs sampler
        # sampling parameters for all nodes bottom-up
        for depth in range(tree.depth-1, -1, -1):
            sample_node(node, 10, tree.ctab[depth])

        # sample concentrations
        for level in range(1, tree.depth):
            sample_concentration(
                tree.ctab[depth], tree.get_nodes_at_depth(level))

        if iter > n_burn_in:
            record_probability_recursively(tree.root)

    return tree


def init_parameters_recursively(node):
    """

    Algorithm 2

    node: node of which we want to intialise the parameters

    """
    # init. children and collect stats
    if not node.is_leaf():
        for child in node.children:
            init_parameters_recursively(child)
            for k in range(node.xc_card):
                node.n[k] = node.n[k] + child.t[k]
                node.marginal_n = node.marginal_n + child.t[k]  # marginal

    if node.is_root():
        # forall k
        for k in range(len(node.t)):
            node.t[k] = min(1, node.n[k])
    else:
        for k in range(node.xc_card):
            if node.n[k] <= 1:
                node.t[k] = node.n[k]
            else:
                alpha = node.get_concentration()
                tinit = alpha * \
                    (utils.digamma(alpha + node.marginal_n) -
                     utils.digamma(alpha))
                node.t[k] = max(1, np.floor(tinit))

    node.marginal_t = np.sum(node.t)  # marginal


def sample_node(node, w, alpha):
    """

    Algorithm 3: Sample psuedo-counts for node

    node: node of which we want to sample the parameters
    w: window for sampling
    alpha: concentration to assign to node

    """
    if alpha <= 0:
        raise Exception
    if node.is_root():
        # no sampling
        # forall k
        for k in range(len(node.t)):
            node.t[k] = min(1, node.n[k])

        return

    node.update_concentration(alpha)  # assign concentration to node

    for k in range(node.xc_card):
        if node.n[k] <= 1:
            node.t[k] = node.n[k]  # value fixed
        else:
            min_tk = int(max(1, node.t[k] - w))
            max_tk = int(min(node.t[k] + w, node.n[k]))

            # Constructing a vector to sample node.t[k] from
            v = np.zeros(int(node.n[k] + 1))
            for t in range(min_tk, max_tk+1):
                v[t] = change_tk_and_get_probability(node, k, t)

            v = np.exp(utils.normalize_log_space(v))

            t = utils.multinomial(v)
            change_tk_and_get_probability(node, k, t)


def change_tk_and_get_probability(node, k, new_value):
    """

    Algorithm 4

    node: node of which we want to sample the parameters
    k: index of the value we want to change in t
    new_value: value to replace t_k by, if possible

    """
    inc = new_value - node.t[k]
    if inc < 0:  # check if valid for parent
        if not node.is_root() and (node.parent.n[k]+inc) < node.parent.t[k]:
            return 0

    node.t[k] = node.t[k] + inc
    node.marginal_t = node.marginal_t + inc  # marginal

    if not node.is_root():  # update statistics at the parent
        node.parent.n[k] = node.parent.n[k] + inc
        node.parent.marginal_n = node.parent.marginal_n + inc  # marginal

    log_s1 = utils.log_stirling(node.parent.n[k], node.parent.t[k])
    log_s2 = utils.log_stirling(node.n[k], node.t[k])
    log_rf = utils.log_rising_factorial(
        node.parent.get_concentration(), node.parent.n[k])

    log_p = log_s1 + log_s2 - log_rf + \
        node.t[k] * np.log(node.get_concentration())
    return log_p


def sample_concentration(alpha, nodes):
    """

    Algorithm 5: Sampling of concentration parameters

    alpha: concentration to sample
    nodes: nodes sharing this concentration parameter (tying)

    """
    # TODO make sure this makes sense ( for marginal_n in beta as well)
    rate = 1  # 1 instead of zero to work with missing data
    for node in nodes:
        # change of variable, sample q
        q = utils.beta(alpha, max(node.marginal_n, 1))
        rate = rate - np.log(q)

    alpha = utils.gamma(sum(n.marginal_t for n in nodes), rate)  # sample alpha

    for node in nodes:
        node.update_concentration(alpha)
