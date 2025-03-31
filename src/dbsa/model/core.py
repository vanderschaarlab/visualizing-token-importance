import numpy as np
from scipy.spatial.distance import cdist


# In the experiments, energy dist is used to reproduce the AISTATS results
def compute_energy_distance(X, Y, distance="cosine"):
    n = len(X)
    m = len(Y)
    # Compute pairwise distances
    if distance == "cosine":
        dists_XY = cdist(X, Y, distance)
        dists_XX = cdist(X, X, distance)
        dists_YY = cdist(Y, Y, distance)
    elif distance == "l1":
        dists_XY = cdist(X, Y, "minkowski", p=1)
        dists_XX = cdist(X, X, "minkowski", p=1)
        dists_YY = cdist(Y, Y, "minkowski", p=1)
    elif distance == "l2":
        dists_XY = cdist(X, Y, "minkowski", p=2)
        dists_XX = cdist(X, X, "minkowski", p=2)
        dists_YY = cdist(Y, Y, "minkowski", p=2)
    else:
        raise ValueError(f"Invalid distance metric: {distance}")

    # Compute the terms
    term1 = (2.0 / (n * m)) * np.sum(dists_XY)
    term2 = (1.0 / n**2) * np.sum(dists_XX)
    term3 = (1.0 / m**2) * np.sum(dists_YY)

    energy_distance = term1 - term2 - term3
    return energy_distance, dists_XY, dists_XX, dists_YY


def permutation_test_energy(X, Y, num_permutations=1000, distance="cosine"):
    combined = np.vstack((X, Y))
    n = len(X)
    E_values = []
    for _ in range(num_permutations):
        np.random.shuffle(combined)
        perm_X = combined[:n]
        perm_Y = combined[n:]
        E_perm, dists_XY, dists_XX, dists_YY = compute_energy_distance(
            perm_X, perm_Y, distance=distance
        )
        E_values.append(E_perm)
    return np.array(E_values)


def compute_energy_distance_fn(
    baseline_embeddings1, baseline_embeddings2, distance="cosine"
):
    E_n, dists_XY, dists_XX, dists_YY = compute_energy_distance(
        baseline_embeddings1, baseline_embeddings2, distance=distance
    )
    E_values = permutation_test_energy(
        baseline_embeddings1,
        baseline_embeddings2,
        num_permutations=500,
        distance=distance,
    )
    p_value = np.mean(E_values >= E_n)
    return E_n, p_value
