import numpy as np
import pybnc.utils as utils


def kdb(X, Y, k=1):
    """
    X: A dataframe
    Y: A series
    k: k
    """
    num_col = X.shape[1]
    # Calculate MI(X_i; Y) from dataset for all attributes
    miy = {column: utils.mutual_information(
        X[:, column], Y) for column in range(num_col)}

    # Calculate MI(X_i; X_j | Y) from dataset for all attributes i != j
    mi_cond = {frozenset((column_i, column_j)):
               utils.conditional_mutual_information(
                   X[:, column_i], X[:, column_j], Y)
               for i, column_i in enumerate(range(num_col))
               for column_j in range(num_col)[i+1:]}

    # Sort attributes on MI(X_i; Y)
    x_sorted = sorted(range(num_col), key=lambda x: miy[x], reverse=True)

    edges = []
    for i, xi in enumerate(x_sorted):
        # Add target Y as parent
        edges.append(("Y", xi))
        # Add k attributes with highest MI(X_i; X_j | Y) as parents
        for vk in range(min(i-1, k), 0, -1):
            m = np.argmax([mi_cond[frozenset((xi, x_sorted[j]))]
                          for j in range(i) if (xi, x_sorted[j]) not in edges])
            edges.append((x_sorted[m], xi))

    return edges
