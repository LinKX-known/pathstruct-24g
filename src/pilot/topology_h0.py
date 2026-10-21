"""Exact H0 persistence (Vietoris-Rips) without a TDA dependency.

The finite death times of H0 are the edge weights of the Euclidean minimum
spanning tree, and the MST is a subgraph of the Delaunay triangulation, so the
deaths are obtained by sorting the MST weights. Exact, not approximate.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import Delaunay, QhullError, cKDTree


def _delaunay_edges(x: np.ndarray) -> np.ndarray:
    tri = Delaunay(x)
    edges = np.vstack(
        [tri.simplices[:, [0, 1]], tri.simplices[:, [1, 2]], tri.simplices[:, [0, 2]]]
    )
    return np.unique(np.sort(edges, axis=1), axis=0)


def _knn_edges(x: np.ndarray, k: int = 10) -> np.ndarray:
    n = len(x)
    k = max(1, min(k, n - 1))
    _, idx = cKDTree(x).query(x, k=k + 1)
    edges = np.stack([np.repeat(np.arange(n), k), idx[:, 1:].ravel()], axis=1)
    return np.unique(np.sort(edges, axis=1), axis=0)


def h0_deaths(points: np.ndarray) -> np.ndarray:
    x = np.asarray(points, dtype=np.float64)
    n = len(x)
    if n < 2:
        return np.zeros(0)
    try:
        edges = _delaunay_edges(x)
    except (QhullError, ValueError):
        edges = _knn_edges(x)
    if len(edges) == 0:
        return np.zeros(0)
    weights = np.linalg.norm(x[edges[:, 0]] - x[edges[:, 1]], axis=1)
    graph = csr_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(n, n))
    mst = minimum_spanning_tree(graph).tocoo()
    return np.sort(mst.data)
